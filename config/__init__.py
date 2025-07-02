"""
Configuration Package for Federated Medical QA System

This package handles all configuration management for the federated learning system,
including model parameters, training settings, and system configuration.
"""

from .config import (
    FederatedConfig,
    ModelConfig,
    TrainingConfig,
    QuantizationConfig,
    GaLoreConfig,
    DataConfig,
    ServerConfig,
    ClientConfig,
    LoggingConfig,
    DeviceConfig,
    EvaluationConfig,
    get_config,
    DEFAULT_CONFIG,
    create_config_file
)

__version__ = "1.0.0"
__all__ = [
    # Main configuration classes
    "FederatedConfig",
    "ModelConfig",
    "TrainingConfig", 
    "QuantizationConfig",
    "GaLoreConfig",
    "DataConfig",
    "ServerConfig",
    "ClientConfig",
    "LoggingConfig",
    "DeviceConfig",
    "EvaluationConfig",
    
    # Configuration functions
    "get_config",
    "DEFAULT_CONFIG",
    "create_config_file"
]

# Convenience function
def get_default_config():
    """Get the default federated learning configuration"""
    return get_config()

def create_custom_config(**kwargs):
    """Create a custom configuration with overrides"""
    config = get_config()
    
    # Apply any keyword arguments as overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

__all__.extend(["get_default_config", "create_custom_config"])

