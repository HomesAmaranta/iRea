"""
TIGER: Recommender Systems with Generative Retrieval.

Uses HuggingFace T5ForConditionalGeneration directly (matching the reference
implementation). Items are represented as flat offset-encoded codebook tokens:
    token = code + codebook_idx * codebook_size + 1
(0 is reserved for padding.)

Reference:
    TIGER: https://arxiv.org/abs/2305.05065
"""

import torch
import torch.nn as nn
from modules.t5 import T5ForConditionalGeneration, T5Config
from typing import Dict, Any
from models.sid import SIDModelMixin, SIDOutput


class Tiger(SIDModelMixin, nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5config = T5Config(
            num_layers=config.get('num_layers', 4),
            num_decoder_layers=config.get('num_decoder_layers', 4),
            d_model=config.get('d_model', 128),
            d_ff=config.get('d_ff', 1024),
            num_heads=config.get('num_heads', 6),
            d_kv=config.get('d_kv', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
            vocab_size=config.get('vocab_size', 769),
            pad_token_id=config.get('pad_token_id', 0),
            eos_token_id=config.get('eos_token_id', 0),
            decoder_start_token_id=config.get('pad_token_id', 0),
            feed_forward_proj=config.get('feed_forward_proj', 'relu'),
        )
        self.model = T5ForConditionalGeneration(t5config)
        self.sem_id_dim = config.get('sem_id_dim', 3)
        self.configure_sid(
            config.get('codebook_size', (t5config.vocab_size - 1) // self.sem_id_dim),
            self.sem_id_dim,
            config.get('max_history_tokens', 20 * self.sem_id_dim),
        )

    @property
    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.model.get_input_embeddings().parameters() if p.requires_grad)
        return total, emb

    def forward(self, input_ids, attention_mask=None, labels=None):
        if isinstance(input_ids, dict):
            return self.forward_sid(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def forward_sid(self, inp):
        self._validate_inp(inp)
        labels = inp.get("labels")
        if labels is not None and labels.shape != (inp["input_ids"].shape[0], self.sem_id_dim):
            raise ValueError("labels must contain one complete target SID per sample")
        decoder_ids = None
        if labels is None:
            decoder_ids = inp["input_ids"].new_zeros((inp["input_ids"].shape[0], 1))
        outputs = self.model(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            labels=labels, decoder_input_ids=decoder_ids,
        )
        return SIDOutput(outputs.loss, outputs.logits, outputs.last_hidden_state)

    def encode_item_hidden(self, inp):
        """Last encoder layer for one complete SID, pooled before any normalization."""
        self._validate_inp(inp)
        if inp["input_ids"].shape[1] != self.sem_id_dim or not inp["attention_mask"].all():
            raise ValueError("Encoder item mean requires one complete unpadded SID")
        hidden = self.model.encoder(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
        )[0]
        return hidden.float().mean(dim=1)

    def _generation_context(self, inp):
        encoder_hidden = self.model.encoder(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
        )[0]
        return encoder_hidden, inp["attention_mask"]

    def _next_sid_token(self, context, prefix):
        start = prefix.new_zeros((prefix.shape[0], 1))
        outputs = self.model(
            encoder_outputs=(context[0],), attention_mask=context[1],
            decoder_input_ids=torch.cat((start, prefix), dim=1),
        )
        return outputs.logits[:, -1, :], outputs.last_hidden_state[:, -1, :]

    def _next_sid_token_cached(self, context, prefix, cache):
        # The encoder output and cross-attention KV stay fixed during decoding.
        decoder_ids = prefix.new_zeros((prefix.shape[0], 1)) if cache is None else prefix[:, -1:]
        outputs = self.model(
            encoder_outputs=(context[0],), attention_mask=context[1],
            decoder_input_ids=decoder_ids, past_key_values=cache, use_cache=True,
        )
        return (outputs.logits[:, -1, :], outputs.last_hidden_state[:, -1, :],
                outputs.past_key_values)

    def generate(self, input_ids, attention_mask=None, num_beams=20, constrain_sid=False,
                 use_cache=True):
        if isinstance(input_ids, dict):
            return SIDModelMixin.generate(
                self, input_ids, num_beams=num_beams, constrain_sid=constrain_sid,
                use_cache=use_cache,
            )
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.sem_id_dim + 1,  # +1 for decoder_start_token
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
