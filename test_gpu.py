#!/usr/bin/env python3
"""
Test script for GPU allocation and system components
"""

import os
import sys
import torch
import subprocess
import time
from pathlib import Path

def test_gpu_availability():
    """Test GPU availability and allocation"""
    print("🔍 GPU AVAILABILITY TEST")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # List all GPUs
        for i in range(torch.cuda.device_count()):
            try:
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
            except:
                print(f"GPU {i}: Information not available")
        
        # Test current device
        try:
            current_device = torch.cuda.current_device()
            print(f"Current Device: {current_device}")
            
            # Test memory allocation
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"Test allocation successful on GPU {current_device}")
            
            # Clear memory
            del test_tensor
            torch.cuda.empty_cache()
            print("Memory cleared successfully")
            
        except Exception as e:
            print(f"GPU test failed: {e}")
    
    else:
        print("CUDA not available - will use CPU")
    
    print("=" * 50)

def test_imports():
    """Test if all required packages are available"""
    print("📦 PACKAGE IMPORT TEST")
    print("=" * 50)
    
    required_packages = [
        "torch",
        "transformers", 
        "flask",
        "requests",
        "numpy",
        "pandas"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: OK")
        except ImportError as e:
            print(f"❌ {package}: FAILED - {e}")
    
    print("=" * 50)

def test_data_availability():
    """Test if data directory and files exist"""
    print("📁 DATA AVAILABILITY TEST")
    print("=" * 50)
    
    data_dir = Path("./data")
    
    if data_dir.exists():
        print(f"✅ Data directory exists: {data_dir}")
        
        # Check for dataset files
        dataset_files = [
            "dataset.csv",
            "medical_qa_dataset.json"
        ]
        
        for file in dataset_files:
            file_path = data_dir / file
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # KB
                print(f"✅ {file}: {size:.1f} KB")
            else:
                print(f"⚠️  {file}: Not found")
        
        # List all files in data directory
        all_files = list(data_dir.glob("*"))
        print(f"📊 Total files in data directory: {len(all_files)}")
        
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("💡 Create data directory and add your dataset.csv file")
    
    print("=" * 50)

def test_server_connection():
    """Test if server can be reached"""
    print("🌐 SERVER CONNECTION TEST")
    print("=" * 50)
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not reachable (not running or connection refused)")
    except requests.exceptions.Timeout:
        print("❌ Server timeout")
    except Exception as e:
        print(f"❌ Server test failed: {e}")
    
    print("=" * 50)

def test_config_loading():
    """Test configuration loading"""
    print("⚙️  CONFIGURATION TEST")
    print("=" * 50)
    
    try:
        sys.path.append(".")
        from config.config import get_config, get_optimized_config
        
        # Test default config
        config = get_config()
        print("✅ Default config loaded successfully")
        print(f"   Model size: {config.model.hidden_size}d")
        print(f"   Training epochs: {config.training.max_epochs}")
        print(f"   Batch size: {config.training.batch_size}")
        
        # Test optimized config
        opt_config = get_optimized_config()
        print("✅ Optimized config loaded successfully")
        print(f"   Model size: {opt_config.model.hidden_size}d")
        print(f"   Training epochs: {opt_config.training.max_epochs}")
        print(f"   Batch size: {opt_config.training.batch_size}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("=" * 50)

def test_model_components():
    """Test basic model components"""
    print("🧠 MODEL COMPONENTS TEST")
    print("=" * 50)
    
    try:
        sys.path.append(".")
        from config.config import get_optimized_config
        from transformers import LlamaConfig
        
        config = get_optimized_config()
        model_config = LlamaConfig(
            hidden_size=config.model.hidden_size,
            num_attention_heads=config.model.num_attention_heads,
            num_hidden_layers=config.model.num_hidden_layers,
            vocab_size=config.model.vocab_size
        )
        
        print("✅ Model configuration created successfully")
        print(f"   Hidden size: {model_config.hidden_size}")
        print(f"   Attention heads: {model_config.num_attention_heads}")
        print(f"   Layers: {model_config.num_hidden_layers}")
        print(f"   Vocab size: {model_config.vocab_size}")
        
        # Test basic tensor operations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_tensor = torch.randn(1, 10, model_config.hidden_size, device=device)
        print(f"✅ Test tensor created on {device}")
        print(f"   Shape: {test_tensor.shape}")
        
        # Clear memory
        del test_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
    
    print("=" * 50)

def main():
    """Run all tests"""
    print("🔧 FEDERATED MEDICAL QA SYSTEM - DIAGNOSTIC TESTS")
    print("=" * 80)
    
    # Run all tests
    test_gpu_availability()
    test_imports()
    test_data_availability()
    test_config_loading()
    test_model_components()
    test_server_connection()
    
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("💡 If any tests failed, please address those issues before running the main system.")
    print("💡 For GPU allocation issues, try: CUDA_VISIBLE_DEVICES=0 python test_gpu.py")
    print("💡 For server connection issues, start the server first: python main.py --mode server")

if __name__ == "__main__":
    main()
