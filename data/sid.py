"""Model-independent batching of the existing TIGER sliding-window samples."""

import torch


def sid_collate(batch, codebook_size=256, sem_id_dim=3, max_seq_len=20,
                require_timestamps=False, fusion=False):
    if not batch or min(codebook_size, sem_id_dim, max_seq_len) <= 0:
        raise ValueError("A nonempty batch and positive SID dimensions are required")
    max_tokens = max_seq_len * sem_id_dim
    input_ids = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    labels = torch.empty(len(batch), sem_id_dim, dtype=torch.long)
    has_timestamps = [getattr(sample, "timestamps", None) is not None for sample in batch]
    if (require_timestamps or any(has_timestamps)) and not all(has_timestamps):
        raise ValueError("History timestamps are missing from the SID batch")
    timestamps = torch.zeros_like(input_ids) if all(has_timestamps) else None

    for row, sample in enumerate(batch):
        history = torch.as_tensor(sample.item_ids, dtype=torch.long)
        target = torch.as_tensor(sample.target_ids, dtype=torch.long)
        if history.numel() == 0 or history.numel() % sem_id_dim:
            raise ValueError("History must contain complete, nonempty item SIDs")
        if target.shape != (sem_id_dim,):
            raise ValueError("Target SID length does not match sem_id_dim")
        if ((history < 0) | (history >= codebook_size)).any() or (
            (target < 0) | (target >= codebook_size)
        ).any():
            raise ValueError("Raw SID codes must be in [0, codebook_size)")
        if timestamps is not None:
            item_timestamps = torch.as_tensor(sample.timestamps)
            if item_timestamps.ndim != 1 or item_timestamps.numel() != history.numel() // sem_id_dim:
                raise ValueError("Each history item must have exactly one timestamp")
            if item_timestamps.dtype not in (torch.int32, torch.int64):
                raise ValueError("History timestamps must be integer Unix timestamps")
            token_timestamps = item_timestamps.repeat_interleave(sem_id_dim)[-max_tokens:]
            timestamps[row, -token_timestamps.numel():] = token_timestamps
        history = history[-max_tokens:]
        offsets = torch.arange(history.numel()) % sem_id_dim * codebook_size + 1
        input_ids[row, -history.numel():] = history + offsets
        labels[row] = target + torch.arange(sem_id_dim) * codebook_size + 1

    inp = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": labels,
    }
    if timestamps is not None:
        inp["timestamps"] = timestamps
    targets = [getattr(sample, "align_target_emb", None) for sample in batch]
    if fusion and any(target is None for target in targets):
        raise ValueError("Fusion requires current-item embeddings for every sample")
    if any(target is not None for target in targets):
        if any(target is None for target in targets):
            raise ValueError("Alignment targets are missing from the SID batch")
        targets = torch.stack([torch.as_tensor(target, dtype=torch.float32) for target in targets])
        if targets.ndim != 2 or targets.shape[1] == 0 or not torch.isfinite(targets).all():
            raise ValueError("Alignment targets must be finite vectors")
        # Fusion features are available at inference, unlike next-item alignment targets.
        inp["fusion_emb" if fusion else "align_target_emb"] = targets
    return inp
