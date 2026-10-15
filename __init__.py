"""
iRea: SID-based recommendation experiments

This package provides implementations of various generative recommendation models:

Baseline Models:
    - SASRec: Self-Attentive Sequential Recommendation
    - HSTU: Hierarchical Sequential Transduction Unit

Generative Models:
    - RQVAE: Residual Quantized VAE for semantic ID generation
    - TIGER: Generative Retrieval with semantic IDs
"""

__version__ = "0.1.0"

from . import models
from . import modules
from . import data
from . import trainers

__all__ = ["models", "modules", "data", "trainers", "__version__"]
