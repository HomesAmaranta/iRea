"""
Reusable modules for iRea models.

This module provides common building blocks:

Encoders:
    - LightT5Encoder: Lightweight T5 encoder (random initialization)
    - SentenceT5Encoder: Pretrained sentence-t5 encoder

Attention & Transformers:
    - T5Attention: T5-style attention with relative position bias
    - TransformerEncoderDecoder: Encoder-decoder transformer

Losses:
    - ReconstructionLoss: MSE/Cosine reconstruction loss
    - CategoricalReconstructionLoss: Cross-entropy for categorical features
    - QuantizeLoss: Commitment loss for VQ-VAE

Metrics:
    - TopKAccumulator: Recall@K and NDCG@K computation

Normalization:
    - L2Norm, RMSNorm, SwishLayerNorm, RootMeanSquareLayerNorm

Embeddings:
    - SemIdEmbedding: Multi-codebook semantic ID embedding
    - UserIdEmbedding: User ID embedding with hashing

Utilities:
    - Gumbel-Softmax sampling functions
    - K-means clustering for codebook initialization
    - Learning rate schedulers
"""

from modules.loss import ReconstructionLoss, CategoricalReconstructionLoss, QuantizeLoss
from modules.metrics import TopKAccumulator
from modules.normalize import L2Norm, RMSNorm
from modules.embedding import SemIdEmbedding, UserIdEmbedding
from modules.utils import parse_config, setup_logger

__all__ = [
    "ReconstructionLoss",
    "CategoricalReconstructionLoss",
    "QuantizeLoss",
    "TopKAccumulator",
    "L2Norm",
    "RMSNorm",
    "SemIdEmbedding",
    "UserIdEmbedding",
    "parse_config",
    "setup_logger",
]
