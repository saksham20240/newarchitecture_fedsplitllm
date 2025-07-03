# client/federated_client.py
"""
Complete Federated Learning Client for Medical QA
CSV-ONLY VERSION - Uses only the provided CSV dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import io
import numpy as np
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import LlamaConfig
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FederatedConfig, get_config
from dataset.medical_qa_downloader import MedicalQADownloader
from utils.metrics import MedicalQAMetrics
from utils.tokenizer import SimpleTokenizer

class OneBitQuantizer:
    """
    Enhanced 1-bit quantization implementation
    """
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def round_clip(self, x, a=-1, b=1):
        """RoundClip(x, a, b) = max(a, min(b, round(x)))"""
        return torch.clamp(torch.round(x), min=a, max=b)
    
    def quantize(self, tensor):
        """Apply 1-bit quantization to tensor"""
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0)
        
        # Ensure tensor is contiguous
        tensor = tensor.contiguous()
        
        # Calculate γ = mean absolute value
        gamma = torch.mean(torch.abs(tensor))
        
        # Avoid division by zero
        if gamma < self.epsilon:
            gamma = torch.tensor(1.0)
        
        # Apply quantization formula
        normalized = tensor / (gamma + self.epsilon)
        quantized = self.round_clip(normalized, -1, 1)
        
        return quantized, gamma
    
    def dequantize(self, quantized_tensor, gamma):
        """Restore quantized tensor"""
        return quantized_tensor * (gamma + self.epsilon)

class ClientGaLoreOptimizer:
    """
    IMPROVED GaLore optimizer with better error handling and fallback
    """
    
    def __init__(self, rank=64, update_proj_gap=200, scale=0.25, min_param_size=1000, enabled=True):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.min_param_size = min_param_size
        self.step_count = 0
        self.projectors = {}
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        
    def should_apply_galore(self, gradient):
        """Check if GaLore should be applied to this gradient"""
        if not self.enabled or gradient is None:
            return False
        return gradient.numel() >= self.min_param_size and gradient.dim() >= 2
        
    def update_projection_matrix(self, gradient, param_name):
        """Update projection matrix using SVD - SAFE VERSION"""
        if not self.should_apply_galore(gradient):
            return
            
        try:
            # Ensure gradient is contiguous and finite
            gradient = gradient.contiguous()
            if not torch.isfinite(gradient).all():
                self.logger.warning(f"Non-finite values in gradient for {param_name}")
                return
            
            # Reshape gradient to 2D for SVD
            original_shape = gradient.shape
            if gradient.dim() > 2:
                gradient_2d = gradient.view(gradient.shape[0], -1)
            else:
                gradient_2d = gradient.clone()
                
            # Skip if matrix is too small
            if min(gradient_2d.shape) < self.rank:
                return
                
            # Perform SVD with proper error handling
            try:
                U, S, Vt = torch.linalg.svd(gradient_2d, full_matrices=False)
            except Exception as e:
                self.logger.warning(f"SVD failed for {param_name}: {e}")
                return
                
            # Keep top-k components
            k = min(self.rank, min(U.shape[1], Vt.shape[0]))
            
            # Store projectors with proper dimensions
            self.projectors[param_name] = {
                'U': U[:, :k].detach().clone(),
                'Vt': Vt[:k, :].detach().clone(),
                'original_shape': original_shape,
                'k': k
            }
            
        except Exception as e:
            self.logger.warning(f"Projection matrix update failed for {param_name}: {e}")
            self.projectors[param_name] = None
    
    def project_gradient(self, gradient, param_name):
        """Project gradient to low-rank space - SAFE VERSION"""
        if (not self.enabled or 
            param_name not in self.projectors or 
            self.projectors[param_name] is None or
            not self.should_apply_galore(gradient)):
            return gradient
            
        projector = self.projectors[param_name]
        
        try:
            # Ensure gradient is contiguous and finite
            gradient = gradient.contiguous()
            if not torch.isfinite(gradient).all():
                return gradient
            
            # Reshape gradient to match projector dimensions
            original_shape = gradient.shape
            if gradient.dim() > 2:
                gradient_2d = gradient.view(gradient.shape[0], -1)
            else:
                gradient_2d = gradient.clone()
                
            # Get projector matrices
            U = projector['U']  # Shape: [m, k]
            Vt = projector['Vt']  # Shape: [k, n]
            
            # Check dimensions before multiplication
            if gradient_2d.shape[0] != U.shape[0] or gradient_2d.shape[1] != Vt.shape[1]:
                self.logger.warning(f"Dimension mismatch for {param_name}, skipping projection")
                return gradient
            
            # Project: U^T @ gradient_2d @ Vt^T
            projected = torch.mm(U.T, torch.mm(gradient_2d, Vt.T))
            
            # Apply scaling
            projected = projected * self.scale
            
            return projected
                
        except Exception as e:
            self.logger.warning(f"Projection failed for {param_name}: {e}")
            return gradient
    
    def compress_gradients(self, gradients):
        """Compress gradients using GaLore - SAFE VERSION"""
        if not gradients or not self.enabled:
            return gradients
            
        compressed_gradients = {}
        
        # Update projection matrices periodically
        if self.step_count % self.update_proj_gap == 0:
            for param_name, grad in gradients.items():
                if grad is not None and grad.requires_grad:
                    self.update_projection_matrix(grad, param_name)
        
        # Project gradients
        for param_name, grad in gradients.items():
            if grad is not None:
                try:
                    compressed_gradients[param_name] = self.project_gradient(grad, param_name)
                except Exception as e:
                    self.logger.warning(f"Gradient compression failed for {param_name}: {e}")
                    compressed_gradients[param_name] = grad
            else:
                compressed_gradients[param_name] = None
        
        self.step_count += 1
        return compressed_gradients
    
    def disable(self):
        """Disable GaLore compression"""
        self.enabled = False
        self.logger.info("GaLore compression disabled")
    
    def enable(self):
        """Enable GaLore compression"""
        self.enabled = True
        self.logger.info("GaLore compression enabled")

# Import transformer components (same as before)
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids, seq_len=None):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CSVMedicalDataLoader:
    """
    CSV-only data loader for medical QA datasets
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.data_dir = Path(config.data.data_dir)
        self.batch_size = config.training.batch_size
        self.max_length = config.data.max_sequence_length
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=config.model.vocab_size)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load CSV dataset
        self.load_csv_dataset()
    
    def load_csv_dataset(self):
        """Load the CSV dataset"""
        dataset_path = self.data_dir / self.config.data.dataset_file
        
        if not dataset_path.exists():
            self.logger.info("JSON dataset not found. Processing CSV files...")
            downloader = MedicalQADownloader(str(self.data_dir))
            dataset_path, stats = downloader.download_and_process()
            
            if not dataset_path or stats['total_questions'] == 0:
                raise ValueError("No valid data found in CSV files. Please check your CSV dataset.")
            
            self.logger.info(f"Processed CSV dataset with {stats['total_questions']} questions")
        
        # Load processed dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
        if not self.dataset:
            raise ValueError("Dataset is empty. Please check your CSV files.")
        
        self.logger.info(f"Loaded {len(self.dataset)} medical QA pairs from CSV")
        
        # Split dataset
        self.split_dataset()
    
    def split_dataset(self):
        """Split dataset into train/val/test"""
        random.shuffle(self.dataset)
        
        total_size = len(self.dataset)
        train_size = int(total_size * self.config.data.train_split)
        val_size = int(total_size * self.config.data.val_split)
        
        self.train_data = self.dataset[:train_size]
        self.val_data = self.dataset[train_size:train_size + val_size]
        self.test_data = self.dataset[train_size + val_size:]
        
        # Ensure we have at least some data in each split
        if len(self.train_data) == 0:
            self.train_data = self.dataset[:max(1, total_size // 2)]
        if len(self.val_data) == 0:
            self.val_data = self.dataset[:max(1, total_size // 4)]
        if len(self.test_data) == 0:
            self.test_data = self.dataset[:max(1, total_size // 4)]
        
        self.logger.info(f"Dataset split - Train: {len(self.train_data)}, "
                        f"Val: {len(self.val_data)}, Test: {len(self.test_data)}")
    
    def get_batch(self, split="train"):
        """Get a batch of data"""
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            data = self.test_data
        
        if not data:
            data = self.train_data
        
        # Sample batch
        batch_size = min(self.batch_size, len(data))
        batch_data = random.sample(data, batch_size)
        
        questions = [item["question"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]
        
        # Tokenize
        tokenized = self.tokenize_batch(questions, answers)
        
        return questions, answers, tokenized
    
    def tokenize_batch(self, questions: List[str], answers: List[str]):
        """Tokenize a batch of questions and answers"""
        # Combine questions and answers
        conversations = []
        for q, a in zip(questions, answers):
            conversation = f"Question: {q}\nAnswer: {a}"
            conversations.append(conversation)
        
        # Tokenize using simple tokenizer
        tokenized = self.tokenizer(
            conversations,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokenized

class FederatedMedicalClient(nn.Module):
    """
    Complete federated learning client for medical QA - CSV-ONLY VERSION
    """
    
    def __init__(self, config: FederatedConfig):
        super().__init__()
        self.config = config
        
        # Setup logging
        self.setup_logging()
        
        # Initialize quantizer and GaLore
        self.quantizer = OneBitQuantizer(epsilon=config.quantization.epsilon)
        self.galore_optimizer = ClientGaLoreOptimizer(
            rank=config.galore.rank,
            update_proj_gap=config.galore.update_proj_gap,
            scale=config.galore.scale,
            min_param_size=config.galore.min_param_size,
            enabled=True  # Can be disabled for debugging
        )
        
        # Server communication settings
        self.server_url = config.client.server_url
        self.server_failure_count = 0
        self.max_server_failures = 5
        self.use_local_fallback = False
        
        self.initialize_server()
        
        # Setup model configuration
        self.model_config = LlamaConfig(
            hidden_size=config.model.hidden_size,
            intermediate_size=config.model.intermediate_size,
            num_attention_heads=config.model.num_attention_heads,
            num_key_value_heads=config.model.num_key_value_heads,
            num_hidden_layers=config.model.num_hidden_layers,
            rms_norm_eps=config.model.rms_norm_eps,
            vocab_size=config.model.vocab_size,
            max_position_embeddings=config.model.max_position_embeddings,
            rope_theta=config.model.rope_theta
        )
        
        # Build model components
        self.build_model()
        
        # Initialize CSV data loader and metrics
        self.data_loader = CSVMedicalDataLoader(config)
        self.metrics = MedicalQAMetrics()
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.logger.info("Federated medical client initialized successfully with CSV dataset")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_dir / "client.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build the federated model components"""
        # Embedding layer
        self.embed_tokens = nn.Embedding(
            self.config.model.vocab_size, 
            self.config.model.hidden_size
        )
        
        # Initial layers (client-side)
        self.initial_layers = nn.ModuleList([
            LlamaDecoderLayer(self.model_config, layer_idx=i) 
            for i in self.config.model.client_initial_layers
        ])
        
        # Final layers (client-side)
        self.final_layers = nn.ModuleList([
            LlamaDecoderLayer(self.model_config, layer_idx=i) 
            for i in self.config.model.client_final_layers
        ])
        
        # Output components
        self.norm = LlamaRMSNorm(
            self.config.model.hidden_size, 
            eps=self.config.model.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            self.config.model.hidden_size, 
            self.config.model.vocab_size, 
            bias=False
        )
        
        self.logger.info(f"Model built with {sum(p.numel() for p in self.parameters())} parameters")
    
    def initialize_server(self):
        """Initialize the server"""
        try:
            config_dict = self.config.model.__dict__.copy()
            
            response = requests.post(
                f"{self.server_url}/initialize",
                json=config_dict,
                timeout=self.config.client.request_timeout
            )
            
            if response.status_code == 200:
                self.logger.info("✓ Server initialized successfully")
            else:
                self.logger.error(f"✗ Server initialization failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"✗ Server communication error: {e}")
    
    def create_causal_mask(self, seq_length, device):
        """Create causal attention mask"""
        mask = torch.full((seq_length, seq_length), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward_initial_layers(self, input_ids):
        """Process through initial layers"""
        input_ids = input_ids.contiguous()
        
        # Token embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position IDs
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Create causal mask
        attention_mask = self.create_causal_mask(seq_length, input_ids.device)
        
        # Process through initial layers
        for layer in self.initial_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        return hidden_states, attention_mask, position_ids
    
    def send_to_server(self, hidden_states, attention_mask, position_ids):
        """Send quantized hidden states to server with fallback"""
        if self.use_local_fallback:
            # Skip server communication and return input as-is
            return {
                'processed_hidden_states': hidden_states,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        
        try:
            # Ensure tensors are contiguous
            hidden_states = hidden_states.contiguous()
            attention_mask = attention_mask.contiguous()
            position_ids = position_ids.contiguous()
            
            # Quantize hidden states
            quantized_hidden, gamma_hidden = self.quantizer.quantize(hidden_states)
            
            data_package = {
                'quantized_hidden_states': quantized_hidden,
                'gamma_hidden': gamma_hidden,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'original_shape': hidden_states.shape
            }
            
            # Serialize and send
            buffer = io.BytesIO()
            torch.save(data_package, buffer)
            buffer.seek(0)
            
            response = requests.post(
                f"{self.server_url}/process_hidden_states",
                data=buffer.read(),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self.config.client.request_timeout
            )
            
            if response.status_code == 200:
                response_buffer = io.BytesIO(response.content)
                server_response = torch.load(response_buffer)
                
                # Dequantize received data
                processed_hidden = self.quantizer.dequantize(
                    server_response['processed_hidden_states'],
                    server_response['gamma_hidden']
                )
                
                return {
                    'processed_hidden_states': processed_hidden,
                    'attention_mask': server_response['attention_mask'],
                    'position_ids': server_response['position_ids']
                }
            else:
                self.logger.error(f"Server error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Communication error: {e}")
            return None
    
    def forward_final_layers(self, hidden_states, attention_mask, position_ids):
        """Process through final layers"""
        hidden_states = hidden_states.contiguous()
        
        # Process through final layers
        for layer in self.final_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        # Final normalization and prediction
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """Compute loss with multiple components"""
        # Ensure all tensors are contiguous
        logits = logits.contiguous()
        targets = targets.contiguous()
        
        # Validate tensor shapes
        if logits.size(0) != targets.size(0):
            raise ValueError(f"Batch size mismatch: logits {logits.shape}, targets {targets.shape}")
        
        if logits.size(1) != targets.size(1):
            raise ValueError(f"Sequence length mismatch: logits {logits.shape}, targets {targets.shape}")
        
        # Flatten tensors for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.config.data.pad_token_id,
            reduction='mean'
        )
        
        # Length penalty (optional)
        if self.config.training.length_penalty_weight > 0:
            seq_lengths = (targets != self.config.data.pad_token_id).sum(dim=1).float()
            if seq_lengths.numel() > 0:
                target_length = 20.0
                length_penalty = torch.mean(torch.abs(seq_lengths - target_length) / target_length)
                
                total_loss = (self.config.training.language_modeling_weight * lm_loss + 
                             self.config.training.length_penalty_weight * length_penalty)
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss
        
        return total_loss, lm_loss
    
    def send_gradients_to_server(self, gradients):
        """Send compressed gradients to server with fallback and error handling"""
        if self.use_local_fallback:
            # Skip server communication for gradients
            return None
        
        try:
            # Filter out None gradients and ensure contiguity
            filtered_gradients = {}
            for param_name, grad in gradients.items():
                if grad is not None and torch.isfinite(grad).all():
                    filtered_gradients[param_name] = grad.contiguous()
            
            if not filtered_gradients:
                self.logger.warning("No valid gradients to send")
                return None
            
            # OPTION: Skip GaLore compression for debugging
            # compressed_gradients = filtered_gradients  # Uncomment to disable GaLore
            compressed_gradients = self.galore_optimizer.compress_gradients(filtered_gradients)
            
            # Limit gradient payload size (send only smaller gradients to server)
            small_gradients = {}
            total_size = 0
            max_size = 50 * 1024 * 1024  # 50MB limit
            
            for param_name, grad in compressed_gradients.items():
                if grad is not None:
                    grad_size = grad.numel() * 4  # Approximate size in bytes
                    if total_size + grad_size < max_size:
                        small_gradients[param_name] = grad
                        total_size += grad_size
                    else:
                        self.logger.debug(f"Skipping large gradient {param_name} (size: {grad_size})")
            
            if not small_gradients:
                self.logger.warning("All gradients too large, skipping server update")
                return None
            
            # Quantize compressed gradients
            quantized_gradients = {}
            for param_name, grad in small_gradients.items():
                try:
                    quantized_grad, gamma_grad = self.quantizer.quantize(grad)
                    quantized_gradients[param_name] = {
                        'quantized_grad': quantized_grad,
                        'gamma_grad': gamma_grad,
                        'shape': grad.shape
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to quantize gradient for {param_name}: {e}")
            
            if not quantized_gradients:
                self.logger.warning("No gradients successfully quantized")
                return None
            
            # Send to server with shorter timeout
            buffer = io.BytesIO()
            torch.save(quantized_gradients, buffer)
            buffer.seek(0)
            
            self.logger.debug(f"Sending {len(quantized_gradients)} gradients to server (size: {total_size/1024:.1f}KB)")
            
            response = requests.post(
                f"{self.server_url}/process_gradients",
                data=buffer.read(),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=min(30, self.config.client.request_timeout)  # Shorter timeout
            )
            
            if response.status_code == 200:
                response_buffer = io.BytesIO(response.content)
                updated_gradients = torch.load(response_buffer)
                self.server_failure_count = 0  # Reset failure count on success
                return updated_gradients
            else:
                self.logger.error(f"Gradient processing error: {response.status_code}")
                self.server_failure_count += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Gradient communication error: {e}")
            self.server_failure_count += 1
            return None
    
    def apply_server_gradients(self, server_gradients):
        """Apply updated gradients from server"""
        if not server_gradients:
            return
            
        param_dict = dict(self.named_parameters())
        
        for param_name, grad_info in server_gradients.items():
            if param_name in param_dict and 'quantized_grad' in grad_info:
                try:
                    # Dequantize gradient
                    updated_grad = self.quantizer.dequantize(
                        grad_info['quantized_grad'],
                        grad_info['gamma_grad']
                    )
                    
                    # Apply to parameter
                    param = param_dict[param_name]
                    if param.grad is None:
                        param.grad = updated_grad.clone()
                    else:
                        param.grad += updated_grad * 0.1  # Scale factor
                        
                except Exception as e:
                    self.logger.warning(f"Failed to apply gradient for {param_name}: {e}")
    
    def check_server_health(self):
        """Check if we should switch to local fallback"""
        if self.server_failure_count >= self.max_server_failures and not self.use_local_fallback:
            self.logger.warning(f"Server failed {self.server_failure_count} times, switching to local fallback")
            self.use_local_fallback = True
            self.galore_optimizer.disable()  # Disable GaLore when using local fallback
    
    def training_step(self, batch_data):
        """Single training step with improved error handling"""
        questions, answers, tokenized = batch_data
        
        input_ids = tokenized['input_ids']
        
        # Validate input
        if input_ids.size(0) == 0:
            self.logger.warning("Empty batch received, skipping training step")
            return None, None
        
        if input_ids.size(1) < 2:
            self.logger.warning("Sequence too short for next-token prediction, skipping")
            return None, None
        
        try:
            # Forward pass through initial layers
            hidden_states, attention_mask, position_ids = self.forward_initial_layers(input_ids)
            
            # Send to server for middle layer processing (with fallback)
            server_response = self.send_to_server(hidden_states, attention_mask, position_ids)
            
            if server_response is None:
                # Use local processing as fallback
                server_response = {
                    'processed_hidden_states': hidden_states,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
            
            # Process through final layers
            logits = self.forward_final_layers(
                server_response['processed_hidden_states'],
                server_response['attention_mask'],
                server_response['position_ids']
            )
            
            # Compute loss
            targets = input_ids[:, 1:].contiguous()
            logits_for_loss = logits[:, :-1, :].contiguous()
            
            total_loss, lm_loss = self.compute_loss(logits_for_loss, targets)
            
            # Backward pass
            total_loss.backward()
            
            # Collect gradients
            gradients = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Send gradients to server (with fallback)
            server_gradients = self.send_gradients_to_server(gradients)
            
            if server_gradients is not None:
                self.apply_server_gradients(server_gradients)
            
            # Check server health
            self.check_server_health()
            
            # Update metrics
            self.metrics.update(total_loss.item(), questions, answers)
            
            return total_loss.item(), lm_loss.item()
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            return None, None
    
    def train_epoch(self, optimizer):
        """Train for one epoch"""
        self.train()
        epoch_losses = []
        
        # Estimate number of batches
        num_batches = max(1, len(self.data_loader.train_data) // self.config.training.batch_size)
        
        for batch_idx in range(num_batches):
            try:
                optimizer.zero_grad()
                
                # Get batch
                batch_data = self.data_loader.get_batch("train")
                
                # Training step
                total_loss, lm_loss = self.training_step(batch_data)
                
                if total_loss is not None:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    
                    epoch_losses.append(total_loss)
                    self.step_count += 1
                    
                    # Log progress
                    if batch_idx % 5 == 0:
                        status = "LOCAL" if self.use_local_fallback else "FEDERATED"
                        self.logger.info(
                            f"Epoch {self.epoch_count}, Batch {batch_idx}/{num_batches}, "
                            f"Loss: {total_loss:.4f}, LM Loss: {lm_loss:.4f} [{status}]"
                        )
                else:
                    self.logger.warning(f"Batch {batch_idx} failed")
                    
            except Exception as e:
                self.logger.error(f"Training batch {batch_idx} failed: {e}")
                continue
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate(self):
        """Validation step"""
        self.eval()
        val_losses = []
        
        with torch.no_grad():
            try:
                # Get validation batch
                batch_data = self.data_loader.get_batch("val")
                
                if batch_data[2]['input_ids'].size(0) > 0:
                    questions, answers, tokenized = batch_data
                    input_ids = tokenized['input_ids']
                    
                    # Forward pass
                    hidden_states, attention_mask, position_ids = self.forward_initial_layers(input_ids)
                    server_response = self.send_to_server(hidden_states, attention_mask, position_ids)
                    
                    if server_response is None:
                        server_response = {
                            'processed_hidden_states': hidden_states,
                            'attention_mask': attention_mask,
                            'position_ids': position_ids
                        }
                    
                    logits = self.forward_final_layers(
                        server_response['processed_hidden_states'],
                        server_response['attention_mask'],
                        server_response['position_ids']
                    )
                    
                    # Compute loss
                    targets = input_ids[:, 1:].contiguous()
                    logits_for_loss = logits[:, :-1, :].contiguous()
                    
                    total_loss, _ = self.compute_loss(logits_for_loss, targets)
                    val_losses.append(total_loss.item())
                    
            except Exception as e:
                self.logger.warning(f"Validation failed: {e}")
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def check_convergence(self, val_loss):
        """Check if training has converged"""
        if val_loss < self.best_loss - self.config.training.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.training.patience
    
    def train_until_convergence(self):
        """Train until convergence"""
        self.logger.info("🚀 Starting federated training with CSV dataset...")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Training loop
        for epoch in range(self.config.training.max_epochs):
            self.epoch_count = epoch
            
            # Train epoch
            train_loss = self.train_epoch(optimizer)
            
            # Validate
            if epoch % self.config.evaluation.eval_frequency == 0:
                val_loss = self.validate()
                
                status = "LOCAL FALLBACK" if self.use_local_fallback else "FEDERATED"
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.config.training.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Best Loss: {self.best_loss:.4f} [{status}]"
                )
                
                # Check convergence
                if self.check_convergence(val_loss):
                    self.logger.info(f"🏁 Converged! No improvement for {self.config.training.patience} epochs")
                    break
                
                # Early stopping check
                if train_loss < self.config.training.convergence_threshold:
                    self.logger.info(f"🎉 Loss below threshold! Training complete")
                    break
            else:
                status = "LOCAL FALLBACK" if self.use_local_fallback else "FEDERATED"
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.config.training.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} [{status}]"
                )
        
        final_status = "with local fallback" if self.use_local_fallback else "in federated mode"
        self.logger.info(f"🏆 Training completed {final_status} using CSV dataset. Final loss: {self.best_loss:.6f}")
        return self.best_loss

def main():
    """Main function"""
    # Load configuration
    config = get_config()
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print("🏥 FEDERATED MEDICAL QA TRAINING SYSTEM (CSV-ONLY)")
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: {config.model.hidden_size}d, {config.model.num_hidden_layers} layers")
    print(f"  Training: {config.training.max_epochs} epochs, lr={config.training.learning_rate}")
    print(f"  Data: CSV files in {config.data.data_dir}/")
    print(f"  Server: {config.client.server_url}")
    print("=" * 60)
    
    # Initialize client
    try:
        client = FederatedMedicalClient(config)
        
        # Train until convergence
        final_loss = client.train_until_convergence()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final loss: {final_loss:.6f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
