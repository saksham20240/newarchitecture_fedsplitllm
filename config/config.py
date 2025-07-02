# config/config.py
"""
Configuration file for Federated Medical QA System
"""

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
    
    # Federated split configuration
    client_initial_layers: List[int] = None
    server_middle_layers: List[int] = None
    client_final_layers: List[int] = None
    
    def __post_init__(self):
        if self.client_initial_layers is None:
            self.client_initial_layers = [0, 1]
        if self.server_middle_layers is None:
            self.server_middle_layers = list(range(3, 31))
        if self.client_final_layers is None:
            self.client_final_layers = [30, 31]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 2
    learning_rate: float = 1e-4
    max_epochs: int = 50
    convergence_threshold: float = 0.01
    patience: int = 3
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Loss function weights
    language_modeling_weight: float = 1.0
    length_penalty_weight: float = 0.1
    
    # Early stopping
    min_delta: float = 0.001
    restore_best_weights: bool = True

@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    enabled: bool = True
    epsilon: float = 1e-8
    bits: int = 1
    compression_ratio: float = 0.97  # Expected compression ratio

@dataclass
class GaLoreConfig:
    """GaLore gradient compression configuration"""
    enabled: bool = True
    rank: int = 64
    update_proj_gap: int = 200
    scale: float = 0.25
    min_param_size: int = 1000  # Minimum parameter size to apply GaLore

@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: str = "./data"
    dataset_file: str = "medical_qa_dataset.json"
    max_sequence_length: int = 512
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
    timeout: int = 60
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    
    # API endpoints
    initialize_endpoint: str = "/initialize"
    process_hidden_states_endpoint: str = "/process_hidden_states"
    process_gradients_endpoint: str = "/process_gradients"
    status_endpoint: str = "/status"

@dataclass
class ClientConfig:
    """Client configuration"""
    server_url: str = "http://localhost:5000"
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Communication settings
    use_compression: bool = True
    verify_ssl: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_dir: str = "./logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class DeviceConfig:
    """Device configuration"""
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    
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
    eval_frequency: int = 5  # Evaluate every N epochs
    save_best_model: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_perplexity: bool = True
    
    # Generation settings for evaluation
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class FederatedConfig:
    """Complete federated learning configuration"""
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
    
    # System settings
    seed: int = 42
    experiment_name: str = "federated_medical_qa"
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
            evaluation=EvaluationConfig()
        )
    
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
        import json
        from pathlib import Path
        
        config_dict = self.to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FederatedConfig':
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            quantization=QuantizationConfig(**config_dict['quantization']),
            galore=GaLoreConfig(**config_dict['galore']),
            data=DataConfig(**config_dict['data']),
            server=ServerConfig(**config_dict['server']),
            client=ClientConfig(**config_dict['client']),
            logging=LoggingConfig(**config_dict['logging']),
            device=DeviceConfig(**config_dict['device']),
            evaluation=EvaluationConfig(**config_dict['evaluation'])
        )

# Default configuration instance
DEFAULT_CONFIG = FederatedConfig.get_default_config()

def get_config() -> FederatedConfig:
    """Get the default configuration"""
    return DEFAULT_CONFIG

def create_config_file(filepath: str = "./config/default_config.json"):
    """Create a default configuration file"""
    config = get_config()
    config.save(filepath)
    print(f"Default configuration saved to {filepath}")

if __name__ == "__main__":
    # Create default config file
    create_config_file()
