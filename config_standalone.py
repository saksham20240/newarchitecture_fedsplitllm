# config_standalone.py
"""
Standalone configuration file for Federated Medical QA System - VERSION 2
No dependencies on existing config structure
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 512  # REDUCED for faster training
    intermediate_size: int = 2048  # REDUCED
    num_attention_heads: int = 8  # REDUCED
    num_key_value_heads: int = 8  # REDUCED
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    
    # Federated split configuration - OPTIMIZED
    client_initial_layers: List[int] = None
    server_middle_layers: List[int] = None
    client_final_layers: List[int] = None
    
    def __post_init__(self):
        if self.client_initial_layers is None:
            self.client_initial_layers = [0, 1]
        if self.server_middle_layers is None:
            # MINIMAL: Only 2 layers for maximum speed
            self.server_middle_layers = [10, 20]
        if self.client_final_layers is None:
            self.client_final_layers = [30, 31]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_epochs: int = 5  # REDUCED default
    convergence_threshold: float = 0.2  # INCREASED for faster convergence
    patience: int = 2
    gradient_clip_norm: float = 0.5
    warmup_steps: int = 10  # REDUCED
    weight_decay: float = 0.01
    
    # Loss function weights
    language_modeling_weight: float = 1.0
    length_penalty_weight: float = 0.0
    
    # Early stopping
    min_delta: float = 0.02
    restore_best_weights: bool = True

@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    enabled: bool = True
    epsilon: float = 1e-8
    bits: int = 1
    compression_ratio: float = 0.95

@dataclass
class GaLoreConfig:
    """GaLore gradient compression configuration"""
    enabled: bool = True
    rank: int = 4  # MINIMAL rank
    update_proj_gap: int = 10  # REDUCED
    scale: float = 0.01  # REDUCED
    min_param_size: int = 25  # REDUCED

@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: str = "./data"
    dataset_file: str = "medical_qa_dataset.json"
    max_sequence_length: int = 100  # REDUCED for speed
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    
    # Data processing
    shuffle: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    timeout: int = 20  # REDUCED
    max_content_length: int = 50 * 1024 * 1024  # 50MB
    
    # Performance settings
    max_processing_time: int = 15  # REDUCED
    enable_threading: bool = True
    thread_timeout: int = 20
    
    # API endpoints
    initialize_endpoint: str = "/initialize"
    process_hidden_states_endpoint: str = "/process_hidden_states"
    process_gradients_endpoint: str = "/process_gradients"
    status_endpoint: str = "/status"

@dataclass
class ClientConfig:
    """Client configuration"""
    server_url: str = "http://localhost:5000"
    request_timeout: int = 20  # REDUCED
    max_retries: int = 2
    retry_delay: float = 0.5
    
    # Communication settings
    use_compression: bool = True
    verify_ssl: bool = False
    
    # Fallback settings
    max_server_failures: int = 2
    server_communication_interval: int = 10  # INCREASED to reduce communication
    gradient_accumulation_steps: int = 12  # INCREASED

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_dir: str = "./logs"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5

@dataclass
class DeviceConfig:
    """Device configuration"""
    device: str = "auto"
    mixed_precision: bool = False
    compile_model: bool = False
    
    # Memory management
    memory_fraction: float = 0.7  # Use 70% of GPU memory
    clear_cache_frequency: int = 3  # Clear cache every 3 steps
    
    def get_device(self) -> torch.device:
        """Get the appropriate device"""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.device)

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    eval_frequency: int = 5
    save_best_model: bool = True
    compute_bleu: bool = False
    compute_rouge: bool = False
    compute_perplexity: bool = True
    
    # Generation settings
    max_new_tokens: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Memory management
    clear_cache_frequency: int = 3
    gradient_checkpointing: bool = False
    
    # Communication optimization
    max_gradient_size: int = 50_000  # VERY SMALL
    compression_level: int = 6
    
    # Processing limits
    max_batch_processing_time: int = 15
    max_server_processing_time: int = 10
    
    # Fallback settings
    enable_local_fallback: bool = True
    fallback_threshold: int = 2

@dataclass
class FederatedConfig:
    """Complete federated learning configuration - STANDALONE"""
    model: ModelConfig
    training: TrainingConfig
    quantization: QuantizationConfig
    galore: GaLoreConfig
    data: DataConfig
    server: ServerConfig
    client: ClientConfig
    logging: LoggingConfig
    device: DeviceConfig
    evaluation: EvaluationConfig
    performance: PerformanceConfig
    
    # System settings
    seed: int = 42
    experiment_name: str = "federated_medical_qa_v2"
    save_dir: str = "./checkpoints"
    
    @classmethod
    def get_default_config(cls) -> 'FederatedConfig':
        """Get default configuration"""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            quantization=QuantizationConfig(),
            galore=GaLoreConfig(),
            data=DataConfig(),
            server=ServerConfig(),
            client=ClientConfig(),
            logging=LoggingConfig(),
            device=DeviceConfig(),
            evaluation=EvaluationConfig(),
            performance=PerformanceConfig()
        )
    
    @classmethod
    def from_runtime_config(cls, config_file: str) -> 'FederatedConfig':
        """Load configuration from runtime config file"""
        if not os.path.exists(config_file):
            return cls.get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            return cls(
                model=ModelConfig(**config_dict.get('model', {})),
                training=TrainingConfig(**config_dict.get('training', {})),
                quantization=QuantizationConfig(**config_dict.get('quantization', {})),
                galore=GaLoreConfig(**config_dict.get('galore', {})),
                data=DataConfig(**config_dict.get('data', {})),
                server=ServerConfig(**config_dict.get('server', {})),
                client=ClientConfig(**config_dict.get('client', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                device=DeviceConfig(**config_dict.get('device', {})),
                evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
                performance=PerformanceConfig(**config_dict.get('performance', {}))
            )
        except Exception as e:
            print(f"Warning: Failed to load runtime config: {e}")
            return cls.get_default_config()
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        import dataclasses
        
        def asdict_recursive(obj):
            if dataclasses.is_dataclass(obj):
                return {k: asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        return asdict_recursive(self)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        from pathlib import Path
        
        config_dict = self.to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

# Global configuration management
_DEFAULT_CONFIG = None
_RUNTIME_CONFIG = None

def get_config() -> FederatedConfig:
    """Get the default configuration"""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = FederatedConfig.get_default_config()
    return _DEFAULT_CONFIG

def get_optimized_config() -> FederatedConfig:
    """Get an optimized configuration"""
    config = get_config()
    
    # Apply even more aggressive optimizations
    config.training.max_epochs = 3  # VERY SHORT
    config.training.convergence_threshold = 0.5  # VERY HIGH
    config.model.server_middle_layers = [15]  # ONLY 1 LAYER
    config.server.timeout = 15
    config.client.request_timeout = 15
    config.client.gradient_accumulation_steps = 20
    config.galore.rank = 2  # MINIMAL rank
    config.data.max_sequence_length = 50  # VERY SHORT
    config.model.hidden_size = 256  # VERY SMALL
    config.model.intermediate_size = 1024  # VERY SMALL
    config.model.num_attention_heads = 4  # VERY SMALL
    config.model.num_key_value_heads = 4  # VERY SMALL
    
    return config

def get_runtime_config() -> FederatedConfig:
    """Get configuration from runtime file or default"""
    global _RUNTIME_CONFIG
    
    # Check for runtime config file
    config_file = os.environ.get('FEDERATED_CONFIG_FILE')
    if config_file and os.path.exists(config_file):
        if _RUNTIME_CONFIG is None:
            _RUNTIME_CONFIG = FederatedConfig.from_runtime_config(config_file)
        return _RUNTIME_CONFIG
    
    # Fall back to optimized config
    return get_optimized_config()

if __name__ == "__main__":
    # Test configuration
    config = get_optimized_config()
    print("🚀 STANDALONE CONFIGURATION V2:")
    print("=" * 50)
    print(f"✓ Model size: {config.model.hidden_size}d")
    print(f"✓ Server layers: {len(config.model.server_middle_layers)}")
    print(f"✓ Max epochs: {config.training.max_epochs}")
    print(f"✓ Sequence length: {config.data.max_sequence_length}")
    print(f"✓ Timeout: {config.server.timeout}s")
    print("=" * 50)
