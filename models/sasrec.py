"""SASRec with a decoder-only, relative-position causal attention backbone."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sid import SIDModelMixin


class SASRec(SIDModelMixin, nn.Module):
    """A single decoder stack, retaining the SASRec training and SID interfaces.

    Each of ``num_blocks`` decoder blocks contains one causal self-attention
    sublayer and one FFN. Only the first block has relative position bias.
    Each sublayer uses pre-RMSNorm; final RMSNorm and dropout run once after
    the complete decoder stack. Checkpoints from the old backbone are incompatible.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 20,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        loss_type: str = "bce",  # "bce" (original paper) or "ce" (cross-entropy)
        codebook_size: Optional[int] = None,
        sem_id_dim: int = 3,
        d_kv: Optional[int] = None,
        feed_forward_proj: str = "relu",
        attention_window: Optional[int] = None,
    ):
        super().__init__()
        if min(num_items, max_seq_len, embed_dim, num_heads, num_blocks, ffn_dim) <= 0:
            raise ValueError("Model dimensions and number of decoder blocks must be positive")
        if d_kv is None:
            if embed_dim % num_heads:
                raise ValueError("embed_dim must be divisible by num_heads when d_kv is unset")
            d_kv = embed_dim // num_heads
        if d_kv <= 0:
            raise ValueError("d_kv must be positive")
        if attention_window is not None and attention_window <= 0:
            raise ValueError("attention_window must be positive")
        if loss_type not in ("bce", "ce"):
            raise ValueError("loss_type must be bce or ce")
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.loss_type = loss_type
        if codebook_size is not None:
            if num_items != codebook_size * sem_id_dim:
                raise ValueError("num_items must equal the offset SID vocabulary size")
            self.loss_type = "ce"
            self.configure_sid(codebook_size, sem_id_dim, max_seq_len - sem_id_dim + 1)

        # Padding is excluded by the attention mask, not by zeroing embeddings.
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([
            SASRecBlock(
                embed_dim, num_heads, ffn_dim, dropout, d_kv,
                feed_forward_proj=feed_forward_proj,
                has_relative_attention_bias=(index == 0),
                attention_window=attention_window,
            )
            for index in range(num_blocks)
        ])
        self.decoder_final_layer_norm = RMSNorm(embed_dim)

    def _encode(self, input_ids: torch.Tensor, cache=None, use_cache=False):
        """Encode input sequence into hidden representations [B, L, D]."""
        if cache is not None and not use_cache:
            raise ValueError("cache requires use_cache=True")
        mask = input_ids.ne(0)
        key_mask = mask if cache is None else torch.cat((cache[1], mask), dim=1)
        x = self.dropout(self.item_embedding(input_ids))
        present = []
        for index, block in enumerate(self.decoder_blocks):
            if use_cache:
                past = None if cache is None else cache[0][index]
                x, kv = block(x, key_mask, past_key_value=past, use_cache=True)
                present.append(kv)
            else:
                x = block(x, mask)
        x = self.dropout(self.decoder_final_layer_norm(x))

        if use_cache:
            return x, (tuple(present), key_mask)
        return x

    def _lm_logits(self, hidden_states):
        return (hidden_states * self.embed_dim ** -0.5) @ self.item_embedding.weight.T

    def _next_sid_token_cached(self, context, prefix, cache):
        # Prefill the history once, then only encode the newly selected token.
        ids = context[0] if cache is None else prefix[:, -1:]
        hidden, cache = self._encode(ids, cache=cache, use_cache=True)
        hidden = hidden[:, -1, :]
        return self._lm_logits(hidden), hidden, cache

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        targets: Optional[torch.Tensor] = None,  # [B, L]
        negatives: Optional[torch.Tensor] = None,  # [B, L] for BCE loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item IDs sequence [B, L], 0 is padding
            targets: Target item IDs [B, L] for computing loss
            negatives: Negative sample IDs [B, L], required when loss_type="bce"

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Training loss if targets provided, else None
        """
        if isinstance(input_ids, dict):
            return self.forward_sid(input_ids)
        hidden = self._encode(input_ids)  # [B, L, D]

        loss = None
        if targets is not None:
            if self.loss_type == "bce":
                assert negatives is not None, "negatives required for BCE loss"
                pos_emb = self.item_embedding(targets)    # [B, L, D]
                neg_emb = self.item_embedding(negatives)  # [B, L, D]
                scaled_hidden = hidden * self.embed_dim ** -0.5
                pos_logits = (scaled_hidden * pos_emb).sum(dim=-1)  # [B, L]
                neg_logits = (scaled_hidden * neg_emb).sum(dim=-1)  # [B, L]

                valid_mask = (targets != 0).float()
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction='none'
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction='none'
                )
                loss = ((pos_loss + neg_loss) * valid_mask).sum() / valid_mask.sum()
            else:
                logits_all = self._lm_logits(hidden)
                loss = F.cross_entropy(
                    logits_all.view(-1, self.num_items + 1),
                    targets.view(-1),
                    ignore_index=0,
                )

        # Full logits only when needed (evaluation or CE loss already computed)
        if targets is not None and self.loss_type == "bce":
            return None, loss
        logits = self._lm_logits(hidden)  # [B, L, num_items+1]
        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item."""
        logits, _ = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        last_logits[:, 0] = float('-inf')  # Exclude padding
        _, top_k_items = torch.topk(last_logits, top_k, dim=-1)
        return top_k_items


class RMSNorm(nn.Module):
    """RMS normalization without mean subtraction or bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(self.weight.dtype)


class RelativePositionBias(nn.Module):
    """Causal relative distances, exact nearby and logarithmically bucketed afar."""

    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        distance = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = self.num_buckets // 2
        large_bucket = max_exact + (
            torch.log(distance.float() / max_exact + 1e-6)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).to(torch.long)
        large_bucket = large_bucket.clamp_max(self.num_buckets - 1)
        return torch.where(distance < max_exact, distance, large_bucket)

    def forward(self, query_length, key_length, device, query_offset=0):
        # Cached queries retain their absolute offsets within the full prefix.
        query_positions = torch.arange(
            query_offset, query_offset + query_length, device=device
        )[:, None]
        key_positions = torch.arange(key_length, device=device)[None, :]
        buckets = self._relative_position_bucket(key_positions - query_positions)
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class CausalMultiHeadAttention(nn.Module):
    """Unscaled causal QK attention with bias-free Q/K/V/O projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dropout_rate: float,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        attention_window: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = num_heads * d_kv
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.attention_window = attention_window
        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.position_bias = (
            RelativePositionBias(
                relative_attention_num_buckets, relative_attention_max_distance, num_heads
            )
            if has_relative_attention_bias else None
        )

    def _shape(self, states):
        batch_size, seq_len, _ = states.shape
        return states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)

    def forward(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self._shape(self.q(hidden_states))
        key_states = self._shape(self.k(hidden_states))
        value_states = self._shape(self.v(hidden_states))
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[0].shape[2]
            key_states = torch.cat((past_key_value[0], key_states), dim=2)
            value_states = torch.cat((past_key_value[1], value_states), dim=2)
        key_length = key_states.shape[2]

        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.position_bias is not None:
            scores = scores + self.position_bias(
                seq_len, key_length, hidden_states.device, query_offset=past_length
            )

        query_positions = torch.arange(
            past_length, past_length + seq_len, device=hidden_states.device
        )[:, None]
        key_positions = torch.arange(key_length, device=hidden_states.device)[None, :]
        distance = query_positions - key_positions
        causal_mask = distance >= 0
        if self.attention_window is not None:
            causal_mask = causal_mask & (distance < self.attention_window)
        scores = scores.masked_fill(
            ~causal_mask.view(1, 1, seq_len, key_length), torch.finfo(scores.dtype).min
        )
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, key_length)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_kv
        )
        output = self.o(attn_output)
        return (output, (key_states, value_states)) if use_cache else output


class FeedForward(nn.Module):
    """Bias-free ReLU or gated GELU feed-forward sublayer."""

    def __init__(self, d_model, d_ff, dropout_rate, feed_forward_proj):
        super().__init__()
        self.feed_forward_proj = feed_forward_proj
        self.dropout = nn.Dropout(dropout_rate)
        if feed_forward_proj.startswith("gated"):
            self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
            self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, hidden_states):
        if self.feed_forward_proj.startswith("gated"):
            hidden_states = F.gelu(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        else:
            hidden_states = F.relu(self.wi(hidden_states))
        return self.wo(self.dropout(hidden_states))


class SASRecBlock(nn.Module):
    """One pre-RMSNorm causal self-attention and one pre-RMSNorm FFN."""

    def __init__(
        self, embed_dim, num_heads, ffn_dim, dropout, d_kv,
        feed_forward_proj="relu", has_relative_attention_bias=False, attention_window=None,
    ):
        super().__init__()
        self.self_attn_norm = RMSNorm(embed_dim)
        self.self_attn = CausalMultiHeadAttention(
            embed_dim, num_heads, d_kv, dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            attention_window=attention_window,
        )
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, feed_forward_proj)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        attn_output = self.self_attn(
            self.self_attn_norm(hidden_states), attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        if use_cache:
            attn_output, present = attn_output
        hidden_states = hidden_states + self.dropout(attn_output)
        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(ffn_output)
        return (hidden_states, present) if use_cache else hidden_states
