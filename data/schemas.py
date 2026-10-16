"""
schemas
"""
from typing import NamedTuple, List, Optional
from torch import Tensor

FUT_SUFFIX = "_fut"

class SeqData(NamedTuple):
    """
    SeqData
    """
    user_id: int
    item_ids: List[int]
    target_ids: List[int]
    timestamps: Optional[List[int]] = None  # One timestamp per history item, before SID expansion.
    align_target_emb: Optional[Tensor] = None

class SeqBatch(NamedTuple):
    """
    SeqBatch
    """
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor

class TokenizedSeqBatch(NamedTuple):
    """
    TokenizedSeqBatch
    """
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor
