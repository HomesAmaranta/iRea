"""
Data loading and preprocessing utilities for iRea.

This module provides dataset implementations for various recommendation scenarios:

Datasets:
    - AmazonItemDataset: Item-level data for RQVAE training
    - AmazonSeqDataset: Sequential data for TIGER training
    - AmazonSASRecDataset: Leave-one-out data for SASRec
    - AmazonHSTUDataset: Temporal data for HSTU

Data Schemas:
    - SeqData: Named tuple for sequence data
    - SeqBatch: Batched sequence data
    - TokenizedSeqBatch: Tokenized batch for transformer models
"""

from data.schemas import SeqData, SeqBatch, TokenizedSeqBatch
from data.utils import cycle

__all__ = [
    "SeqData",
    "SeqBatch",
    "TokenizedSeqBatch",
    "cycle",
]
