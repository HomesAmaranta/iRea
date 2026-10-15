"""Shared SID next-token interfaces, without changing the model backbones."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SIDOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


@dataclass
class SIDGenerationOutput:
    sequences: torch.Tensor  # [B, beams, sid_length], no start token
    scores: torch.Tensor  # [B, beams], summed token log probabilities
    last_hidden_state: torch.Tensor  # [B, beams, D], predicts the final SID token


class SIDModelMixin:
    """Common target-only NTP and fixed-length autoregressive beam search."""

    def configure_sid(self, codebook_size, sem_id_dim, max_history_tokens):
        if min(codebook_size, sem_id_dim, max_history_tokens) <= 0:
            raise ValueError("SID dimensions and history length must be positive")
        self.codebook_size = codebook_size
        self.sem_id_dim = sem_id_dim
        self.max_history_tokens = max_history_tokens

    def _validate_inp(self, inp):
        ids = inp["input_ids"]
        mask = inp["attention_mask"]
        if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] == 0:
            raise ValueError("input_ids and attention_mask must have shape [B, L]")
        if ids.shape[1] > self.max_history_tokens:
            raise ValueError("History exceeds max_history_tokens")
        if not torch.equal(mask.bool(), ids.ne(0)):
            raise ValueError("attention_mask must identify non-padding SID tokens")
        if not mask[:, -1].bool().all() or (mask[:, 1:] < mask[:, :-1]).any():
            raise ValueError("SID history must be nonempty and left-padded")

    def forward_sid(self, inp):
        self._validate_inp(inp)
        ids = inp["input_ids"]
        labels = inp.get("labels")
        if labels is not None:
            if labels.shape != (ids.shape[0], self.sem_id_dim):
                raise ValueError("labels must contain one complete target SID per sample")
            if (labels <= 0).any() or (labels > self.codebook_size * self.sem_id_dim).any():
                raise ValueError("Target SID tokens must be offset-encoded, without padding")
            # The last history token predicts c0; each target prefix predicts the next code.
            ids = torch.cat((ids, labels[:, :-1]), dim=1)
            hidden = self._encode_sid_sequence(ids, inp)[:, -self.sem_id_dim:, :]
        else:
            hidden = self._encode_sid_sequence(ids, inp)[:, -1:, :]
        # Deliberately no d_model scaling or normalization of the output head.
        logits = hidden @ self.item_embedding.weight.T
        loss = None if labels is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        return SIDOutput(loss, logits, hidden)

    def _encode_sid_sequence(self, ids, inp):
        return self._encode(ids)

    def _generation_context(self, inp):
        return inp["input_ids"], inp["attention_mask"]

    def _next_sid_token(self, context, prefix):
        ids = torch.cat((context[0], prefix), dim=1)
        hidden = self._encode(ids)[:, -1, :]
        return hidden @ self.item_embedding.weight.T, hidden

    @torch.no_grad()
    def generate(self, inp, num_beams=20, constrain_sid=False):
        """Generate complete SIDs; inference never reads inp['labels']."""
        self._validate_inp(inp)
        vocab_size = self.codebook_size * self.sem_id_dim + 1
        first_step_options = self.codebook_size if constrain_sid else vocab_size - 1
        if not 1 <= num_beams <= first_step_options:
            raise ValueError("num_beams exceeds the available first-token candidates")
        batch_size = inp["input_ids"].shape[0]
        device = inp["input_ids"].device
        base_context = self._generation_context(inp)
        sequences = torch.empty(batch_size, 1, 0, device=device, dtype=torch.long)
        scores = torch.zeros(batch_size, 1, device=device)

        for step in range(self.sem_id_dim):
            beam_count = sequences.shape[1]
            context = tuple(
                value.repeat_interleave(beam_count, dim=0) for value in base_context
            )
            logits, hidden = self._next_sid_token(
                context, sequences.reshape(batch_size * beam_count, step)
            )
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs[:, 0] = -torch.inf  # 0 is padding/start, never a SID code.
            if constrain_sid:
                start = step * self.codebook_size + 1
                stop = start + self.codebook_size
                log_probs[:, :start] = -torch.inf
                log_probs[:, stop:] = -torch.inf
            candidates = scores.unsqueeze(-1) + log_probs.reshape(
                batch_size, beam_count, vocab_size
            )
            scores, indices = candidates.reshape(batch_size, -1).topk(num_beams, dim=-1)
            parents = indices // vocab_size
            tokens = indices % vocab_size
            sequences = sequences.gather(
                1, parents.unsqueeze(-1).expand(-1, -1, step)
            )
            sequences = torch.cat((sequences, tokens.unsqueeze(-1)), dim=-1)
            last_hidden = hidden.reshape(batch_size, beam_count, -1).gather(
                1, parents.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )

        return SIDGenerationOutput(sequences, scores, last_hidden)
