"""
Model implementations for iRea.

This module provides implementations of various recommendation models:

Baseline Models:
    - SASRec: Self-attention with softmax for sequential recommendation
    - HSTU: SiLU attention + update gate + temporal bias

Generative Models:
    - RQVAE: Vector quantized VAE for semantic ID generation
    - TIGER: Generative retrieval with trie-based constrained decoding
"""

from models.rqvae import RqVae, QuantizeForwardMode
from models.tiger import Tiger
from models.sasrec import SASRec
from models.hstu import HSTU

__all__ = [
    "RqVae",
    "QuantizeForwardMode",
    "Tiger",
    "SASRec",
    "HSTU",
]
