"""
Federated Learning Client Package

This package contains the client-side implementation for federated medical QA.
The client handles layers 0-1 (initial) and 30-31 (final) of the transformer model.
"""

from .federated_client import (
    FederatedMedicalClient,
    OneBitQuantizer,
    ClientGaLoreOptimizer,
    EnhancedMedicalDataLoader,
    # Transformer components
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer
)

__version__ = "1.0.0"
__all__ = [
    # Main client class
    "FederatedMedicalClient",
    
    # Optimization components
    "OneBitQuantizer",
    "ClientGaLoreOptimizer",
    
    # Data handling
    "EnhancedMedicalDataLoader",
    
    # Model components
    "LlamaRMSNorm",
    "LlamaRotaryEmbedding", 
    "LlamaMLP",
    "LlamaAttention",
    "LlamaDecoderLayer"
]

def create_client(config=None):
    """Create a federated medical client"""
    if config is None:
        from config import get_config
        config = get_config()
    
    return FederatedMedicalClient(config)

def start_training(config=None, **kwargs):
    """Start federated training with a client"""
    client = create_client(config)
    
    # Apply any configuration overrides
    for key, value in kwargs.items():
        if hasattr(client.config.training, key):
            setattr(client.config.training, key, value)
    
    return client.train_until_convergence()

__all__.extend(["create_client", "start_training"])
