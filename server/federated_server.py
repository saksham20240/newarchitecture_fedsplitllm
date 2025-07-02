# server/federated_server.py
"""
Complete Federated Learning Server for Medical QA
Handles transformer layers 3-30 with GaLore gradient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import io
import numpy as np
import logging
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import LlamaConfig
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FederatedConfig, get_config

class OneBitQuantizer:
    """
    Server-side 1-bit quantization implementation
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

class ServerGaLoreOptimizer:
    """
    Server-side GaLore optimizer for gradient processing
    """
    
    def __init__(self, rank=64, update_proj_gap=200, scale=0.25, min_param_size=1000):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.min_param_size = min_param_size
        self.step_count = 0
        self.projectors = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def should_apply_galore(self, gradient):
        """Check if GaLore should be applied to this gradient"""
        return gradient.numel() >= self.min_param_size and gradient.dim() >= 2
        
    def update_projection_matrix(self, gradient, param_name):
        """Update projection matrix using SVD"""
        if not self.should_apply_galore(gradient):
            return
            
        try:
            # Reshape gradient to 2D for SVD
            original_shape = gradient.shape
            if gradient.dim() > 2:
                gradient_2d = gradient.view(gradient.shape[0], -1)
            else:
                gradient_2d = gradient
                
            # Perform SVD
            U, S, V = torch.svd(gradient_2d)
            
            # Keep top-k components
            k = min(self.rank, min(U.shape[1], V.shape[0]))
            
            self.projectors[param_name] = {
                'U': U[:, :k].detach(),
                'V': V[:k, :].detach(),
                'original_shape': original_shape
            }
            
            self.logger.debug(f"Updated projection matrix for {param_name} with rank {k}")
            
        except Exception as e:
            self.logger.warning(f"SVD failed for {param_name}: {e}")
            self.projectors[param_name] = None
    
    def project_gradient(self, gradient, param_name):
        """Project gradient to low-rank space"""
        if (param_name not in self.projectors or 
            self.projectors[param_name] is None or
            not self.should_apply_galore(gradient)):
            return gradient
            
        projector = self.projectors[param_name]
        
        try:
            # Reshape gradient
            if gradient.dim() > 2:
                gradient_2d = gradient.view(gradient.shape[0], -1)
            else:
                gradient_2d = gradient
                
            # Project using stored U and V
            projected = torch.mm(projector['U'].T, torch.mm(gradient_2d, projector['V'].T))
            return projected * self.scale
        except Exception as e:
            self.logger.warning(f"Projection failed for {param_name}: {e}")
            return gradient
    
    def reconstruct_gradient(self, projected_grad, param_name):
        """Reconstruct gradient from low-rank projection"""
        if (param_name not in self.projectors or 
            self.projectors[param_name] is None):
            return projected_grad
            
        projector = self.projectors[param_name]
        
        try:
            # Reconstruct
            reconstructed = torch.mm(projector['U'], torch.mm(projected_grad, projector['V']))
            
            # Reshape back to original shape
            reconstructed = reconstructed.view(projector['original_shape'])
            return reconstructed
        except Exception as e:
            self.logger.warning(f"Reconstruction failed for {param_name}: {e}")
            return projected_grad
    
    def process_gradients(self, gradients):
        """Process gradients with GaLore"""
        processed_gradients = {}
        
        # Update projection matrices periodically
        if self.step_count % self.update_proj_gap == 0:
            self.logger.info("Updating projection matrices...")
            for param_name, grad in gradients.items():
                if grad is not None:
                    self.update_projection_matrix(grad, param_name)
        
        # Project gradients
        for param_name, grad in gradients.items():
            if grad is not None:
                processed_gradients[param_name] = self.project_gradient(grad, param_name)
            else:
                processed_gradients[param_name] = None
        
        self.step_count += 1
        return processed_gradients

# Import transformer components (same as client)
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

class FederatedServer(nn.Module):
    """
    Federated Learning Server containing transformer layers 3-30
    """
    
    def __init__(self, config: FederatedConfig):
        super().__init__()
        self.config = config
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.quantizer = OneBitQuantizer(epsilon=config.quantization.epsilon)
        self.galore_optimizer = ServerGaLoreOptimizer(
            rank=config.galore.rank,
            update_proj_gap=config.galore.update_proj_gap,
            scale=config.galore.scale,
            min_param_size=config.galore.min_param_size
        )
        
        # Model configuration
        self.model_config = None
        self.initialized = False
        
        # Training state
        self.training_stats = {
            'processed_batches': 0,
            'gradient_updates': 0,
            'total_communication_bytes': 0
        }
        
        self.logger.info("Federated server initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_dir / "server.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self, model_config_dict):
        """Initialize the server model with configuration"""
        try:
            # Create model configuration
            self.model_config = LlamaConfig(**model_config_dict)
            
            # Build middle layers (3-30)
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(self.model_config, layer_idx=i) 
                for i in self.config.model.server_middle_layers
            ])
            
            # Server optimizer
            self.optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            
            self.initialized = True
            
            param_count = sum(p.numel() for p in self.parameters())
            self.logger.info(f"Server model initialized with {param_count:,} parameters")
            self.logger.info(f"Middle layers: {len(self.layers)} (indices {self.config.model.server_middle_layers[0]}-{self.config.model.server_middle_layers[-1]})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False
    
    def dequantize_hidden_states(self, quantized_data):
        """Dequantize received hidden states from client"""
        try:
            quantized_hidden = quantized_data['quantized_hidden_states']
            gamma_hidden = quantized_data['gamma_hidden']
            
            # Dequantize
            hidden_states = self.quantizer.dequantize(quantized_hidden, gamma_hidden)
            
            return {
                'hidden_states': hidden_states,
                'attention_mask': quantized_data['attention_mask'],
                'position_ids': quantized_data['position_ids'],
                'original_shape': quantized_data['original_shape']
            }
        except Exception as e:
            self.logger.error(f"Dequantization failed: {e}")
            raise
    
    def forward_middle_layers(self, hidden_states, attention_mask, position_ids):
        """Process through middle layers 3-30"""
        try:
            # Process through all middle layers
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
            
            return hidden_states
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise
    
    def quantize_and_prepare_response(self, hidden_states, attention_mask, position_ids):
        """Quantize processed hidden states for sending back to client"""
        try:
            # Apply quantization
            quantized_hidden, gamma_hidden = self.quantizer.quantize(hidden_states)
            
            return {
                'quantized_hidden_states': quantized_hidden,
                'gamma_hidden': gamma_hidden,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'original_shape': hidden_states.shape
            }
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
    
    def process_client_gradients(self, gradient_data):
        """Process gradients received from client using GaLore"""
        try:
            # Dequantize gradients
            client_gradients = {}
            for param_name, grad_info in gradient_data.items():
                if isinstance(grad_info, dict) and 'quantized_grad' in grad_info:
                    # Dequantize gradient
                    dequantized_grad = self.quantizer.dequantize(
                        grad_info['quantized_grad'], 
                        grad_info['gamma_grad']
                    )
                    client_gradients[param_name] = dequantized_grad
            
            self.logger.debug(f"Dequantized {len(client_gradients)} gradients")
            
            # Process with GaLore
            processed_gradients = self.galore_optimizer.process_gradients(client_gradients)
            
            # Apply gradients to server parameters
            self.optimizer.zero_grad()
            
            # Set gradients for server parameters
            param_dict = dict(self.named_parameters())
            applied_gradients = 0
            
            for param_name, grad in processed_gradients.items():
                if param_name in param_dict and grad is not None:
                    if param_dict[param_name].grad is None:
                        param_dict[param_name].grad = grad.clone()
                    else:
                        param_dict[param_name].grad += grad
                    applied_gradients += 1
            
            # Update server parameters
            if applied_gradients > 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 
                    self.config.training.gradient_clip_norm
                )
                
                self.optimizer.step()
                self.training_stats['gradient_updates'] += 1
                
                self.logger.debug(f"Applied {applied_gradients} gradients and updated parameters")
            
            # Prepare updated gradients for client
            updated_gradients = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Quantize gradient
                    quantized_grad, gamma_grad = self.quantizer.quantize(param.grad)
                    updated_gradients[name] = {
                        'quantized_grad': quantized_grad,
                        'gamma_grad': gamma_grad,
                        'shape': param.grad.shape
                    }
            
            self.logger.debug(f"Prepared {len(updated_gradients)} updated gradients")
            return updated_gradients
            
        except Exception as e:
            self.logger.error(f"Gradient processing failed: {e}")
            raise
    
    def get_stats(self):
        """Get server statistics"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "layers": f"{self.config.model.server_middle_layers[0]}-{self.config.model.server_middle_layers[-1]}",
            "parameters": sum(p.numel() for p in self.parameters()),
            "training_stats": self.training_stats,
            "model_config": {
                "hidden_size": self.model_config.hidden_size,
                "num_layers": len(self.layers),
                "num_attention_heads": self.model_config.num_attention_heads
            } if self.model_config else None
        }

# Flask application setup
def create_app(config: FederatedConfig = None):
    """Create Flask application"""
    if config is None:
        config = get_config()
    
    app = Flask(__name__)
    
    # Configure Flask
    app.config['MAX_CONTENT_LENGTH'] = config.server.max_content_length
    
    # Global server instance
    server_model = None
    
    @app.route('/initialize', methods=['POST'])
    def initialize_server():
        """Initialize server with configuration"""
        nonlocal server_model
        
        try:
            config_data = request.json
            if not config_data:
                return jsonify({"error": "No configuration provided"}), 400
            
            # Initialize server model
            server_model = FederatedServer(config)
            success = server_model.initialize_model(config_data)
            
            if success:
                app.logger.info("Server initialized successfully")
                return jsonify({
                    "status": "initialized", 
                    "layers": f"{config.model.server_middle_layers[0]}-{config.model.server_middle_layers[-1]}",
                    "parameters": sum(p.numel() for p in server_model.parameters())
                })
            else:
                return jsonify({"error": "Model initialization failed"}), 500
        
        except Exception as e:
            app.logger.error(f"Server initialization failed: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/process_hidden_states', methods=['POST'])
    def process_hidden_states():
        """Process hidden states from client through middle layers"""
        nonlocal server_model
        
        if server_model is None or not server_model.initialized:
            return jsonify({"error": "Server not initialized"}), 500
        
        try:
            # Receive data from client
            data_buffer = io.BytesIO(request.data)
            received_data = torch.load(data_buffer, map_location='cpu')
            
            app.logger.info(f"Received hidden states with shape: {received_data['original_shape']}")
            
            # Update stats
            server_model.training_stats['processed_batches'] += 1
            server_model.training_stats['total_communication_bytes'] += len(request.data)
            
            # Dequantize hidden states
            dequantized_data = server_model.dequantize_hidden_states(received_data)
            
            # Process through middle layers
            processed_hidden_states = server_model.forward_middle_layers(
                dequantized_data['hidden_states'],
                dequantized_data['attention_mask'],
                dequantized_data['position_ids']
            )
            
            app.logger.info(f"Processed through middle layers. Output shape: {processed_hidden_states.shape}")
            
            # Quantize and prepare response
            response_data = server_model.quantize_and_prepare_response(
                processed_hidden_states,
                dequantized_data['attention_mask'],
                dequantized_data['position_ids']
            )
            
            # Send back quantized processed hidden states
            response_buffer = io.BytesIO()
            torch.save({
                'processed_hidden_states': response_data['quantized_hidden_states'],
                'gamma_hidden': response_data['gamma_hidden'],
                'attention_mask': response_data['attention_mask'],
                'position_ids': response_data['position_ids']
            }, response_buffer)
            response_buffer.seek(0)
            
            return response_buffer.read(), 200, {'Content-Type': 'application/octet-stream'}
        
        except Exception as e:
            app.logger.error(f"Error processing hidden states: {e}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    @app.route('/process_gradients', methods=['POST'])
    def process_gradients():
        """Process gradients from client using GaLore"""
        nonlocal server_model
        
        if server_model is None or not server_model.initialized:
            return jsonify({"error": "Server not initialized"}), 500
        
        try:
            # Receive gradient data from client
            data_buffer = io.BytesIO(request.data)
            gradient_data = torch.load(data_buffer, map_location='cpu')
            
            app.logger.info(f"Received gradients for {len(gradient_data)} parameters")
            
            # Process gradients with GaLore
            updated_gradients = server_model.process_client_gradients(gradient_data)
            
            app.logger.info(f"Processed gradients with GaLore, sending {len(updated_gradients)} updated gradients back")
            
            # Send back updated gradients
            response_buffer = io.BytesIO()
            torch.save(updated_gradients, response_buffer)
            response_buffer.seek(0)
            
            return response_buffer.read(), 200, {'Content-Type': 'application/octet-stream'}
        
        except Exception as e:
            app.logger.error(f"Error processing gradients: {e}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    @app.route('/status', methods=['GET'])
    def get_status():
        """Get server status"""
        nonlocal server_model
        
        if server_model is None:
            return jsonify({"status": "not_initialized"})
        
        try:
            stats = server_model.get_stats()
            return jsonify(stats)
        except Exception as e:
            app.logger.error(f"Error getting status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "service": "federated_server"})

    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "Payload too large"}), 413

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app

def main():
    """Start the federated learning server"""
    # Load configuration
    config = get_config()
    
    # Create Flask app
    app = create_app(config)
    
    print("🖥️  FEDERATED LEARNING SERVER")
    print("=" * 50)
    print(f"Host: {config.server.host}")
    print(f"Port: {config.server.port}")
    print(f"Debug: {config.server.debug}")
    print("Endpoints:")
    print("  POST /initialize - Initialize server with config")
    print("  POST /process_hidden_states - Process hidden states from client")
    print("  POST /process_gradients - Process gradients with GaLore")
    print("  GET /status - Get server status")
    print("  GET /health - Health check")
    print("=" * 50)
    print("Server handles transformer layers 3-30")
    print("Features:")
    print("  ✓ 1-bit quantization for communication")
    print("  ✓ GaLore gradient compression")
    print("  ✓ Robust error handling")
    print("  ✓ Training statistics tracking")
    print("=" * 50)
    
    try:
        app.run(
            host=config.server.host,
            port=config.server.port,
            debug=config.server.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
