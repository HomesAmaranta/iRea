"""Reference SASRec backbone with SID next-token training and cached generation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sid import SIDModelMixin


class SASRec(SIDModelMixin, nn.Module):
    """Absolute positions, query-only LayerNorm attention and residual ReLU FFNs."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 20,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 512,
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
            raise ValueError("Model dimensions and number of blocks must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        # Keep existing gin bindings, but reject options outside reference SASRec.
        if d_kv is not None and d_kv != embed_dim // num_heads:
            raise ValueError("SASRec requires d_kv = embed_dim // num_heads")
        if feed_forward_proj != "relu":
            raise ValueError("SASRec requires feed_forward_proj=relu")
        if attention_window is not None:
            raise ValueError("Reference SASRec does not use an attention_window")
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

        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-8)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(self, input_ids: torch.Tensor, cache=None, use_cache=False):
        """Encode input sequence into hidden representations [B, L, D]."""
        if cache is not None and not use_cache:
            raise ValueError("cache requires use_cache=True")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [B, L] with L > 0")
        past_length = 0 if cache is None else cache[1].shape[1]
        length = input_ids.shape[1]
        if past_length + length > self.max_seq_len:
            raise ValueError("Sequence exceeds the absolute position embedding capacity")
        mask = input_ids.ne(0)
        key_mask = mask if cache is None else torch.cat((cache[1], mask), dim=1)
        x = self.item_embedding(input_ids)
        if self.loss_type != "bce":
            x = x * self.embed_dim ** 0.5
        positions = torch.arange(past_length, past_length + length, device=input_ids.device)
        x = self.emb_dropout(x + self.position_embedding(positions)[None])
        query_mask = mask.unsqueeze(-1).to(x.dtype)
        x = x * query_mask
        present = []
        for index, block in enumerate(self.blocks):
            if use_cache:
                past = None if cache is None else cache[0][index]
                x, kv = block(x, key_mask.unsqueeze(-1), past_key_value=past, use_cache=True)
                present.append(kv)
            else:
                x = block(x, mask.unsqueeze(-1))
            x = x * query_mask
        x = self.final_norm(x)

        if use_cache:
            return x, (tuple(present), key_mask)
        return x

    def _lm_logits(self, hidden_states):
        return hidden_states @ self.item_embedding.weight.T

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
                pos_logits = (hidden * pos_emb).sum(dim=-1)  # [B, L]
                neg_logits = (hidden * neg_emb).sum(dim=-1)  # [B, L]

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


class MultiHeadAttention(nn.Module):
    """Scaled QK attention with a normalized-query residual and optional KV cache."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, states):
        batch_size, seq_len, _ = states.shape
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query, key_value, mask, past_key_value=None, use_cache=False
    ):
        batch_size, seq_len, _ = query.shape
        query_states = self._shape(self.q_proj(query))
        key_states = self._shape(self.k_proj(key_value))
        value_states = self._shape(self.v_proj(key_value))
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[0].shape[2]
            key_states = torch.cat((past_key_value[0], key_states), dim=2)
            value_states = torch.cat((past_key_value[1], value_states), dim=2)
        key_length = key_states.shape[2]

        scores = (query_states @ key_states.transpose(-1, -2)) * self.scale
        query_positions = torch.arange(
            past_length, past_length + seq_len, device=query.device
        )[:, None]
        key_positions = torch.arange(key_length, device=query.device)[None, :]
        causal_mask = key_positions <= query_positions
        key_mask = mask.squeeze(-1).bool()[:, None, None, :]
        # FP32 masking keeps the reference -1e9 sentinel finite under autocast.
        masked_scores = scores.float().masked_fill(~key_mask, -1e9).masked_fill(
            ~causal_mask[None, None], -1e9
        )
        attn_weights = F.softmax(masked_scores, dim=-1).to(scores.dtype)
        query_mask = mask[:, -seq_len:, 0][:, None, :, None].to(attn_weights.dtype)
        attn_weights = attn_weights * query_mask
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = attn_output + query
        return (output, (key_states, value_states)) if use_cache else output


class PointWiseFeedForward(nn.Module):
    """Reference ReLU FFN with two dropouts and an unnormalized residual."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        out = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return self.dropout(out) + residual


class SASRecBlock(nn.Module):
    """Normalize only attention queries, then apply a pre-LayerNorm residual FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PointWiseFeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(self, x, mask, past_key_value=None, use_cache=False):
        x = self.attention(
            self.norm1(x), x, mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        if use_cache:
            x, present = x
        x = self.ffn(self.norm2(x), x)
        return (x, present) if use_cache else x
