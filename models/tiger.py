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
from modules.fusion import ItemFusion
from modules.nitp import configure_nitp, nitp_loss


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
        self.fusion_mode = config.get("fusion_mode", "none")
        self.fusion_position = config.get("fusion_position", "out")
        if self.fusion_mode not in ("none", "add", "concat"):
            raise ValueError("fusion_mode must be none, add or concat")
        if self.fusion_position not in ("out", "attn"):
            raise ValueError("fusion_position must be out or attn")
        if self.fusion_mode != "none":
            self.item_fusion = ItemFusion(
                t5config.d_model, config.get("fusion_dim", 0), self.fusion_mode
            )
        configure_nitp(self, t5config.d_model, t5config.num_decoder_layers,
                       config.get("nitp_weight", 0.0), config.get("nitp_target_layer", 1))

    def _fusion_embedding(self, inp):
        embedding = inp.get("fusion_emb")
        if (embedding is None or embedding.ndim != 2
                or embedding.shape != (inp["input_ids"].shape[0], self.item_fusion.embedding_dim)
                or not torch.isfinite(embedding).all()):
            raise ValueError("Fusion requires a finite current-item fusion_emb of shape [B, fusion_dim]")
        return embedding.detach()

    def _fusion_kwargs(self, embedding):
        if self.fusion_mode == "none":
            return {}
        return dict(
            first_token_fusion=self.item_fusion, fusion_emb=embedding,
            fusion_position=self.fusion_position,
        )

    @property
    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.model.get_input_embeddings().parameters() if p.requires_grad)
        return total, emb

    def forward(self, input_ids, attention_mask=None, labels=None):
        if isinstance(input_ids, dict):
            return self.forward_sid(input_ids)
        if self.fusion_mode != "none":
            raise ValueError("Fusion requires the SID dictionary interface with fusion_emb")
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
        if labels is not None and (
            (labels <= 0).any() or (labels > self.codebook_size * self.sem_id_dim).any()
        ):
            raise ValueError("Target SID tokens must be offset-encoded, without padding")
        use_nitp = self.training and labels is not None and self.nitp_weight > 0
        decoder_ids = None
        if labels is None:
            decoder_ids = inp["input_ids"].new_zeros((inp["input_ids"].shape[0], 1))
        elif use_nitp:
            decoder_ids = torch.cat((labels.new_zeros((labels.shape[0], 1)), labels), dim=1)
        outputs = self.model(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            labels=None if use_nitp else labels, decoder_input_ids=decoder_ids,
            nitp_target_layer=self.nitp_target_layer if use_nitp else 0,
            **self._fusion_kwargs(
                self._fusion_embedding(inp) if self.fusion_mode != "none" else None
            ),
        )
        if use_nitp:
            hidden = outputs.last_hidden_state[:, :-1, :]
            logits = outputs.logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            auxiliary = nitp_loss(self.nitp_projector, hidden, outputs.nitp_target)
            return SIDOutput(loss, logits, hidden, auxiliary)
        return SIDOutput(outputs.loss, outputs.logits, outputs.last_hidden_state)

    def _generation_context(self, inp):
        encoder_hidden = self.model.encoder(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]
        )[0]
        context = (encoder_hidden, inp["attention_mask"])
        if self.fusion_mode != "none":
            context += (self._fusion_embedding(inp),)
        return context

    def _next_sid_token(self, context, prefix):
        start = prefix.new_zeros((prefix.shape[0], 1))
        outputs = self.model(
            encoder_outputs=(context[0],), attention_mask=context[1],
            decoder_input_ids=torch.cat((start, prefix), dim=1),
            **self._fusion_kwargs(context[2] if self.fusion_mode != "none" else None),
        )
        return outputs.logits[:, -1, :], outputs.last_hidden_state[:, -1, :]

    def _next_sid_token_cached(self, context, prefix, cache):
        # The encoder output and cross-attention KV stay fixed during decoding.
        decoder_ids = prefix.new_zeros((prefix.shape[0], 1)) if cache is None else prefix[:, -1:]
        outputs = self.model(
            encoder_outputs=(context[0],), attention_mask=context[1],
            decoder_input_ids=decoder_ids, past_key_values=cache, use_cache=True,
            **self._fusion_kwargs(context[2] if self.fusion_mode != "none" else None),
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
        if self.fusion_mode != "none":
            raise ValueError("Fusion requires the SID dictionary interface with fusion_emb")
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.sem_id_dim + 1,  # +1 for decoder_start_token
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
