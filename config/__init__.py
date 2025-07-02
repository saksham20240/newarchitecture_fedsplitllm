"""
Configuration package for Federated Medical QA System
"""

from .config import (
    FederatedConfig,
    ModelConfig,
    TrainingConfig,
    get_config,
    DEFAULT_CONFIG
)

__all__ = [
    'FederatedConfig',
    'ModelConfig', 
    'TrainingConfig',
    'get_config',
    'DEFAULT_CONFIG'
]

__version__ = '1.0.0'
