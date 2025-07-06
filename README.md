# Federated Medical QA System

A privacy-preserving federated learning system for medical question-answering using split learning architecture.

## 🏗️ Architecture

- **Client**: Handles most computation (20M+ parameters)
- **Server**: Processes middle layers (1M parameters)  
- **Split Learning**: Computation divided between client/server
- **Privacy**: Raw medical data stays on client

## 📁 Core Files

- `run_federated_v2.py` - Main training orchestrator
- `federated_server_v2.py` - Server implementation
- `federated_client_v2.py` - Client implementation  
- `requirements.txt` - Dependencies

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup wandb (optional)
wandb login

# Run training
python run_federated_v2.py --epochs 20 --gpu 0 --wandb
```

## 📊 Features

- **Wandb Integration**: Real-time metrics and visualization
- **GPU Support**: CUDA acceleration
- **Medical Focus**: Specialized for healthcare Q&A
- **Privacy-First**: Federated learning approach

## 🔧 Configuration

- Epochs: 20 (default)
- Batch size: 1
- Learning rate: 0.0001
- Memory fraction: 50%
- Server port: 5000

## 📈 Metrics Tracked

- Training/validation loss
- Communication times
- GPU memory usage
- Gradient norms
- Server processing times

## 🏥 Dataset

Medical Q&A covering:
- Diabetes, Hypertension, Cardiology
- Respiratory, Infectious diseases
- Mental health, Oncology
- Endocrinology, Nephrology

## 🛠️ System Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Network access for federated communication
