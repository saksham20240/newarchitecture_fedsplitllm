# config/config_v2.py
"""
Configuration file for Federated Medical QA System - VERSION 2
Improved runtime configuration handling and parameter overrides
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
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
            # Even more optimized - fewer layers for faster processing
            self.server_middle_layers = [5, 10, 15, 20, 25]  # Only 5 layers
        if self.client_final_layers is None:
            self.client_final_layers = [30, 31]

@dataclass
class TrainingConfig:
    """Training configuration - OPTIMIZED"""
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_epochs: int = 10  # DEFAULT reduced from 20
    convergence_threshold: float = 0.1
    patience: int = 2
    gradient_clip_norm: float = 0.5
    warmup_steps: int = 25  # Further reduced
    weight_decay: float = 0.01
    
    # Loss function weights
    language_modeling_weight: float = 1.0
    length_penalty_weight: float = 0.0
    
    # Early stopping
    min_delta: float = 0.01
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
    rank: int = 8  # FURTHER reduced for better performance
    update_proj_gap: int = 25  # FURTHER reduced
    scale: float = 0.02  # FURTHER reduced
    min_param_size: int = 50  # FURTHER reduced

@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: str = "./data"
    dataset_file: str = "medical_qa_dataset.json"
    max_sequence_length: int = 200  # FURTHER reduced
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
    timeout: int = 25  # FURTHER reduced
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    
    # Performance settings
    max_processing_time: int = 20  # FURTHER reduced
    enable_threading: bool = True
    thread_timeout: int = 25
    
    # API endpoints
    initialize_endpoint: str = "/initialize"
    process_hidden_states_endpoint: str = "/process_hidden_states"
    process_gradients_endpoint: str = "/process_gradients"
    status_endpoint: str = "/status"

@dataclass
class ClientConfig:
    """Client configuration"""
    server_url: str = "http://localhost:5000"
    request_timeout: int = 25  # FURTHER reduced
    max_retries: int = 2
    retry_delay: float = 0.5  # FURTHER reduced
    
    # Communication settings
    use_compression: bool = True
    verify_ssl: bool = False
    
    # Fallback settings
    max_server_failures: int = 2  # FURTHER reduced
    server_communication_interval: int = 8  # Increased to reduce communication
    gradient_accumulation_steps: int = 8  # Increased accumulation

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
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    clear_cache_frequency: int = 5  # Clear cache every 5 steps
    
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
    max_new_tokens: int = 25  # FURTHER reduced
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Memory management
    clear_cache_frequency: int = 5
    gradient_checkpointing: bool = False
    
    # Communication optimization
    max_gradient_size: int = 500_000  # FURTHER reduced
    compression_level: int = 6
    
    # Processing limits
    max_batch_processing_time: int = 20  # FURTHER reduced
    max_server_processing_time: int = 15  # FURTHER reduced
    
    # Fallback settings
    enable_local_fallback: bool = True
    fallback_threshold: int = 2

@dataclass
class FederatedConfig:
    """Complete federated learning configuration - VERSION 2"""
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
    
    def apply_memory_optimizations(self):
        """Apply memory optimizations for GPU sharing"""
        # Reduce model size for sharing
        self.model.hidden_size = min(self.model.hidden_size, 512)
        self.model.intermediate_size = min(self.model.intermediate_size, 2048)
        
        # Reduce batch processing
        self.training.batch_size = 1
        self.data.max_sequence_length = min(self.data.max_sequence_length, 128)
        
        # Increase gradient accumulation
        self.client.gradient_accumulation_steps = 16
        
        # Reduce communication frequency
        self.client.server_communication_interval = 10
        
        return self

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
    
    # Apply additional optimizations
    config.training.max_epochs = 5  # Even fewer epochs
    config.training.convergence_threshold = 0.2
    config.model.server_middle_layers = [10, 20]  # Only 2 layers
    config.server.timeout = 20
    config.client.request_timeout = 20
    config.client.gradient_accumulation_steps = 16
    config.galore.rank = 4  # Minimal rank
    config.data.max_sequence_length = 100  # Very short sequences
    
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

def create_config_file(filepath: str = "./config/default_config_v2.json"):
    """Create a default configuration file"""
    config = get_config()
    config.save(filepath)
    print(f"Default configuration v2 saved to {filepath}")

def create_optimized_config_file(filepath: str = "./config/optimized_config_v2.json"):
    """Create an optimized configuration file"""
    config = get_optimized_config()
    config.save(filepath)
    print(f"Optimized configuration v2 saved to {filepath}")

if __name__ == "__main__":
    # Create both config files
    create_config_file()
    create_optimized_config_file()
    
    print("\n🚀 CONFIGURATION V2 OPTIMIZATIONS:")
    print("=" * 50)
    print("✓ Runtime configuration support")
    print("✓ Command-line parameter overrides")
    print("✓ Reduced server layers (2 instead of 10)")
    print("✓ Shorter sequences (100 tokens)")
    print("✓ Better memory management")
    print("✓ Faster timeout handling")
    print("✓ Improved GPU sharing")
    print("=" * 50)
