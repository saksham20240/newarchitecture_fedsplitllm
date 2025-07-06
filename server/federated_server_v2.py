# server/federated_server_v2.py
"""
Complete Federated Learning Server for Medical QA - VERSION 2
Improved performance and better resource management
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
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_v2 import get_runtime_config

class OneBitQuantizer:
    """Server-side 1-bit quantization implementation"""
    
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

class ServerOptimizer:
    """Lightweight server-side gradient optimizer"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.processing_times = []
        self.gradient_counts = []
    
    def process_gradients(self, gradients):
        """Process gradients with simple validation"""
        if not self.enabled or not gradients:
            return gradients
        
        start_time = time.time()
        processed_gradients = {}
        gradient_count = 0
        
        for param_name, grad in gradients.items():
            if grad is not None:
                # Simple validation and clipping
                if torch.isfinite(grad).all():
                    grad_norm = torch.norm(grad)
                    if grad_norm > 10.0:  # Clip very large gradients
                        grad = grad / grad_norm * 10.0
                    processed_gradients[param_name] = grad
                    gradient_count += 1
            else:
                processed_gradients[param_name] = None
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.gradient_counts.append(gradient_count)
        
        # Keep only last 100 measurements
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
            self.gradient_counts = self.gradient_counts[-100:]
        
        self.step_count += 1
        return processed_gradients
    
    def get_stats(self):
        """Get optimizer statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'avg_gradient_count': np.mean(self.gradient_counts),
            'total_steps': self.step_count
        }

# Import transformer components
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
    """Federated Learning Server - VERSION 2"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Device setup with memory management
        self.device = self.get_device()
        self.setup_gpu_memory()
        
        print(f"🖥️  Server using device: {self.device}")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.quantizer = OneBitQuantizer(epsilon=config.quantization.epsilon)
        self.optimizer_component = ServerOptimizer(enabled=True)
        
        # Model configuration
        self.model_config = None
        self.initialized = False
        
        # Training state
        self.training_stats = {
            'processed_batches': 0,
            'gradient_updates': 0,
            'total_communication_bytes': 0,
            'processing_times': [],
            'memory_usage': []
        }
        
        # Performance monitoring
        self.max_processing_time = config.performance.max_server_processing_time
        
        self.logger.info("Federated server v2 initialized")
    
    def get_device(self):
        """Get device respecting CUDA_VISIBLE_DEVICES"""
        if torch.cuda.is_available():
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible and cuda_visible != '':
                return torch.device("cuda:0")
            else:
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def setup_gpu_memory(self):
        """Setup GPU memory management"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**3
            print(f"📊 GPU {current_gpu} ({gpu_name}): {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_dir / "server_v2.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self, model_config_dict):
        """Initialize the server model with configuration"""
        try:
            # Create model configuration
            self.model_config = LlamaConfig(**model_config_dict)
            
            # Build middle layers - REDUCED SET
            middle_layers = self.config.model.server_middle_layers
            
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(self.model_config, layer_idx=i) 
                for i in middle_layers
            ])
            
            # Move model to device
            self.to(self.device)
            print(f"✅ Server model moved to {self.device}")
            
            # Server optimizer
            self.optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
            
            self.initialized = True
            
            param_count = sum(p.numel() for p in self.parameters())
            self.logger.info(f"Server model initialized with {param_count:,} parameters")
            self.logger.info(f"Middle layers: {len(self.layers)} layers")
            self.logger.info(f"Server model moved to device: {self.device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False
    
    def dequantize_hidden_states(self, quantized_data):
        """Dequantize received hidden states from client"""
        try:
            # Load and move to device
            quantized_hidden = quantized_data['quantized_hidden_states'].to(self.device)
            gamma_hidden = quantized_data['gamma_hidden'].to(self.device)
            
            # Dequantize
            hidden_states = self.quantizer.dequantize(quantized_hidden, gamma_hidden)
            
            return {
                'hidden_states': hidden_states,
                'attention_mask': quantized_data['attention_mask'].to(self.device),
                'position_ids': quantized_data['position_ids'].to(self.device),
                'original_shape': quantized_data['original_shape']
            }
        except Exception as e:
            self.logger.error(f"Dequantization failed: {e}")
            raise
    
    def forward_middle_layers(self, hidden_states, attention_mask, position_ids):
        """Process through middle layers with timeout protection"""
        try:
            start_time = time.time()
            
            # Process through layers with time monitoring
            for i, layer in enumerate(self.layers):
                if time.time() - start_time > self.max_processing_time:
                    self.logger.warning(f"Processing timeout at layer {i}, returning current state")
                    break
                    
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
            
            processing_time = time.time() - start_time
            self.training_stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.training_stats['processing_times']) > 100:
                self.training_stats['processing_times'] = self.training_stats['processing_times'][-100:]
            
            # Track memory usage
            if self.device.type == "cuda":
                memory_used = torch.cuda.memory_allocated(self.device.index) / 1024**3
                self.training_stats['memory_usage'].append(memory_used)
                if len(self.training_stats['memory_usage']) > 100:
                    self.training_stats['memory_usage'] = self.training_stats['memory_usage'][-100:]
            
            return hidden_states
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise
    
    def quantize_and_prepare_response(self, hidden_states, attention_mask, position_ids):
        """Quantize processed hidden states for sending back to client"""
        try:
            # Move to CPU for network transfer
            hidden_states_cpu = hidden_states.cpu()
            attention_mask_cpu = attention_mask.cpu()
            position_ids_cpu = position_ids.cpu()
            
            # Apply quantization
            quantized_hidden, gamma_hidden = self.quantizer.quantize(hidden_states_cpu)
            
            return {
                'quantized_hidden_states': quantized_hidden,
                'gamma_hidden': gamma_hidden,
                'attention_mask': attention_mask_cpu,
                'position_ids': position_ids_cpu,
                'original_shape': hidden_states.shape
            }
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
    
    def process_client_gradients(self, gradient_data):
        """Process gradients received from client - OPTIMIZED"""
        try:
            start_time = time.time()
            
            # Quick processing with timeout
            if time.time() - start_time > self.max_processing_time:
                self.logger.warning("Gradient processing timeout, returning empty gradients")
                return {}
            
            # Process gradients
            processed_gradients = {}
            param_dict = dict(self.named_parameters())
            
            for param_name, grad_info in gradient_data.items():
                if isinstance(grad_info, dict) and 'quantized_grad' in grad_info:
                    try:
                        # Dequantize
                        dequantized_grad = self.quantizer.dequantize(
                            grad_info['quantized_grad'].to(self.device), 
                            grad_info['gamma_grad'].to(self.device)
                        )
                        
                        # Apply to server parameter if exists
                        if param_name in param_dict:
                            param = param_dict[param_name]
                            if param.grad is None:
                                param.grad = dequantized_grad * 0.05  # Small update
                            else:
                                param.grad += dequantized_grad * 0.05
                        
                        # Prepare return gradient
                        quantized_grad, gamma_grad = self.quantizer.quantize(dequantized_grad * 0.05)
                        processed_gradients[param_name] = {
                            'quantized_grad': quantized_grad,
                            'gamma_grad': gamma_grad,
                            'shape': dequantized_grad.shape
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process gradient {param_name}: {e}")
                        continue
            
            # Quick optimizer step
            if len(processed_gradients) > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.training_stats['gradient_updates'] += 1
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Processed {len(processed_gradients)} gradients in {processing_time:.2f}s")
            
            return processed_gradients
            
        except Exception as e:
            self.logger.error(f"Gradient processing failed: {e}")
            return {}
    
    def get_stats(self):
        """Get server statistics"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        avg_processing_time = (np.mean(self.training_stats['processing_times']) 
                              if self.training_stats['processing_times'] else 0)
        
        avg_memory_usage = (np.mean(self.training_stats['memory_usage']) 
                           if self.training_stats['memory_usage'] else 0)
        
        stats = {
            "status": "initialized",
            "layers": len(self.layers),
            "parameters": sum(p.numel() for p in self.parameters()),
            "training_stats": self.training_stats,
            "avg_processing_time": avg_processing_time,
            "avg_memory_usage": avg_memory_usage,
            "model_config": {
                "hidden_size": self.model_config.hidden_size,
                "num_layers": len(self.layers),
                "num_attention_heads": self.model_config.num_attention_heads
            } if self.model_config else None
        }
        
        # Add optimizer stats
        optimizer_stats = self.optimizer_component.get_stats()
        stats.update(optimizer_stats)
        
        return stats

# Flask application setup
def create_app(config=None):
    """Create Flask application with improved performance"""
    if config is None:
        config = get_runtime_config()
    
    app = Flask(__name__)
    
    # Configure Flask for better performance
    app.config['MAX_CONTENT_LENGTH'] = config.server.max_content_length
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
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
                    "layers": len(server_model.layers),
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
            start_time = time.time()
            
            # Receive data from client
            data_buffer = io.BytesIO(request.data)
            received_data = torch.load(data_buffer, map_location='cpu')
            
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
            
            processing_time = time.time() - start_time
            app.logger.info(f"Processed hidden states in {processing_time:.2f}s")
            
            return response_buffer.read(), 200, {'Content-Type': 'application/octet-stream'}
        
        except Exception as e:
            app.logger.error(f"Error processing hidden states: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/process_gradients', methods=['POST'])
    def process_gradients():
        """Process gradients from client - OPTIMIZED VERSION"""
        nonlocal server_model
        
        if server_model is None or not server_model.initialized:
            return jsonify({"error": "Server not initialized"}), 500
        
        try:
            start_time = time.time()
            
            # Quick timeout check
            if time.time() - start_time > config.server.timeout:
                app.logger.warning("Gradient processing timeout")
                return jsonify({"error": "Processing timeout"}), 408
            
            # Receive gradient data from client
            data_buffer = io.BytesIO(request.data)
            gradient_data = torch.load(data_buffer, map_location='cpu')
            
            # Process with timeout protection
            updated_gradients = server_model.process_client_gradients(gradient_data)
            
            # Send back updated gradients
            response_buffer = io.BytesIO()
            torch.save(updated_gradients, response_buffer)
            response_buffer.seek(0)
            
            processing_time = time.time() - start_time
            app.logger.info(f"Processed gradients in {processing_time:.2f}s")
            
            return response_buffer.read(), 200, {'Content-Type': 'application/octet-stream'}
        
        except Exception as e:
            app.logger.error(f"Error processing gradients: {e}")
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
        return jsonify({"status": "healthy", "service": "federated_server_v2"})

    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "Payload too large"}), 413

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(408)
    def timeout_error(e):
        return jsonify({"error": "Request timeout"}), 408

    return app

def main():
    """Start the federated learning server"""
    config = get_runtime_config()
    
    # Create Flask app
    app = create_app(config)
    
    print("🖥️  FEDERATED LEARNING SERVER - VERSION 2")
    print("=" * 60)
    print(f"Host: {config.server.host}")
    print(f"Port: {config.server.port}")
    print(f"Debug: {config.server.debug}")
    print("Improvements:")
    print("  ✓ Reduced server layers for faster processing")
    print("  ✓ Better memory management")
    print("  ✓ Improved timeout handling")
    print("  ✓ Performance monitoring")
    print("  ✓ Runtime configuration support")
    print("=" * 60)
    
    try:
        # Use production server
        from werkzeug.serving import make_server
        
        server = make_server(
            config.server.host,
            config.server.port,
            app,
            threaded=True
        )
        
        print("🚀 Server started successfully!")
        server.serve_forever()
        
    except ImportError:
        # Fallback to development server
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
