"""Detached representation diagnostics; never reuse CKA clipping or centering."""

import math

import torch


GEOMETRY_DESCRIPTION = {
    "effective_rank": (
        "exp(entropy(normalized eigenvalues of centered raw-hidden covariance)); "
        "NITP Appendix D / Roy-Vetterli; no clipping or row normalization"
    ),
    "item_cosine": "Exact mean cosine over all distinct nonzero item pairs; no centering",
    "sequence_cosine": (
        "Uniform random nonzero-token pairs from different batch sequences, with replacement; "
        "L2 normalization only; private RNG"
    ),
    "sequence_hidden": "All positions of training output.last_hidden_state, before projection",
    "epoch_reduction": "Arithmetic mean of valid per-batch metrics, not pooled epoch covariance",
}


def _features(hidden):
    if hidden.ndim != 2 or hidden.shape[1] == 0:
        raise ValueError("Geometry requires a [samples, features] matrix")
    values = hidden.detach().to(dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("Geometry requires finite hidden states")
    return values


@torch.no_grad()
def effective_rank(hidden):
    """Covariance spectral entropy: use squared singular values, not singular values."""
    values = _features(hidden)
    if len(values) < 2:
        return None
    # Subtract a reference first so identical rows center exactly to zero.
    centered = values - values[:1]
    centered = centered - centered.mean(dim=0, keepdim=True)
    scale = centered.abs().max()
    if scale == 0:
        return None
    centered = centered / scale
    # Both Grams have the same nonzero spectrum; use the smaller one.
    gram = (centered.T @ centered if centered.shape[1] <= len(centered)
            else centered @ centered.T)
    spectrum = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = spectrum / spectrum.sum()
    positive = probabilities[probabilities > 0]
    return float(torch.exp(-(positive * positive.log()).sum()))


def _unit_vectors(values):
    norms = torch.linalg.vector_norm(values, dim=-1)
    valid = norms > 0
    return values[valid] / norms[valid, None], valid


@torch.no_grad()
def item_geometry(hidden):
    values = _features(hidden)
    unit, valid = _unit_vectors(values)
    n = len(unit)
    cosine = None
    if n >= 2:
        # Subtract the actual diagonal sum, including normalization roundoff.
        cosine = float(
            ((unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1))).clamp(-1, 1)
        )
    rank = effective_rank(values)
    return {
        "effective_rank": rank, "avg_cosine": cosine,
        "effective_rank_status": "valid" if rank is not None else "degenerate",
        "cosine_status": "valid" if cosine is not None else "insufficient_nonzero_items",
        "items": len(values), "nonzero_items": int(valid.sum()),
    }


@torch.no_grad()
def sequence_batch_geometry(hidden, mask, *, num_pairs, generator):
    """Statistics on the unified [batch, prediction positions, dim] training output."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Sequence geometry requires [B, T, D] hidden and a boolean [B, T] mask")
    if isinstance(num_pairs, bool) or not isinstance(num_pairs, int) or num_pairs <= 0:
        raise ValueError("num_pairs must be a positive integer")
    values = _features(hidden[mask])
    rank = effective_rank(values)
    groups = torch.arange(len(hidden), device=hidden.device)[:, None].expand_as(mask)[mask]
    unit, nonzero = _unit_vectors(values)
    groups = groups[nonzero]
    counts = torch.bincount(groups, minlength=len(hidden))
    cosine, pairs = None, 0
    if (counts > 0).sum() >= 2:
        # Each first token has N - sequence_length eligible partners. Weighting
        # by that count samples uniformly over all ordered cross-sequence pairs.
        partner_counts = len(unit) - counts[groups]
        first = torch.multinomial(
            partner_counts.double(), num_pairs, replacement=True, generator=generator
        )
        starts = counts.cumsum(0) - counts
        second = (torch.rand(
            num_pairs, device=hidden.device, dtype=torch.float64, generator=generator
        ) * partner_counts[first]).long()
        first_groups = groups[first]
        second += (second >= starts[first_groups]) * counts[first_groups]
        cosine = float((unit[first] * unit[second]).sum(-1).mean().clamp(-1, 1))
        pairs = num_pairs
    return {
        "effective_rank": rank, "avg_cosine": cosine,
        "tokens": len(values), "nonzero_tokens": len(unit), "sampled_pairs": pairs,
    }


def geometry_log_metrics(record, prefix):
    return {f"{prefix}/{key}": record[key] for key in ("effective_rank", "avg_cosine")
            if record[key] is not None}


class SequenceGeometryEpoch:
    """Keep only scalar batch aggregates; no graphs or epoch-sized hidden buffers."""

    def __init__(self, device, num_pairs=4096, seed=42):
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.num_pairs = num_pairs
        self.sums = {"effective_rank": 0., "avg_cosine": 0.}
        self.counts = dict.fromkeys(self.sums, 0)
        self.batches = self.tokens = self.nonzero_tokens = self.sampled_pairs = 0

    def update(self, hidden, mask):
        record = sequence_batch_geometry(
            hidden, mask, num_pairs=self.num_pairs, generator=self.generator
        )
        self.batches += 1
        self.tokens += record["tokens"]
        self.nonzero_tokens += record["nonzero_tokens"]
        self.sampled_pairs += record["sampled_pairs"]
        for key in self.sums:
            if record[key] is not None and math.isfinite(record[key]):
                self.sums[key] += record[key]
                self.counts[key] += 1

    def result(self):
        return {
            **{key: self.sums[key] / self.counts[key] if self.counts[key] else None
               for key in self.sums},
            "batches": self.batches, "tokens": self.tokens,
            "nonzero_tokens": self.nonzero_tokens, "sampled_pairs": self.sampled_pairs,
            "valid_batches": dict(self.counts),
            "skipped_batches": {key: self.batches - count for key, count in self.counts.items()},
        }
