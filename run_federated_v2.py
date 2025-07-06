#!/usr/bin/env python3
"""
Complete Self-Contained Federated Medical QA System V2
No external dependencies - runs everything in one file
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import json
from pathlib import Path

def create_sample_data():
    """Create sample dataset if it doesn't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    dataset_file = data_dir / "dataset.csv"
    if dataset_file.exists():
        return
    
    print("📋 Creating sample dataset...")
    sample_data = """question,answer,category
What are the symptoms of diabetes?,"Increased thirst, frequent urination, fatigue, blurred vision",diabetes
How is hypertension treated?,"Lifestyle changes, medications, regular monitoring",hypertension
What causes heart attacks?,"Coronary artery disease, blocked blood vessels",cardiology
How is asthma managed?,"Inhalers, avoiding triggers, action plan",respiratory
What are signs of depression?,"Persistent sadness, loss of interest, sleep problems",mental_health
How is pneumonia treated?,"Antibiotics, rest, fluids, sometimes hospitalization",infectious_disease
What causes kidney stones?,"Dehydration, diet, genetics, mineral buildup",nephrology
How is cancer detected?,"Physical exams, imaging tests, blood tests, biopsies",oncology
What treats arthritis?,"Medications, physical therapy, exercise, surgery",rheumatology
How to prevent food poisoning?,"Hand washing, proper cooking, safe food storage",gastroenterology
What are stroke risk factors?,"High blood pressure, diabetes, smoking, obesity",neurology
How is thyroid disease diagnosed?,"Blood tests for TSH, T3, T4 levels",endocrinology
What causes migraines?,"Stress, hormonal changes, certain foods, lack of sleep",neurology
How is diabetes managed?,"Blood sugar monitoring, medication, diet, exercise",diabetes
What are anxiety symptoms?,"Excessive worry, restlessness, difficulty concentrating",mental_health
How is high cholesterol treated?,"Diet changes, exercise, statin medications",cardiology
What causes allergies?,"Immune system overreaction to harmless substances",immunology
How is pneumonia diagnosed?,"Chest X-rays, blood tests, physical examination",respiratory
What are kidney disease signs?,"Fatigue, swelling, changes in urination, nausea",nephrology
How to prevent heart disease?,"Healthy diet, exercise, no smoking, stress management",cardiology"""
    
    with open(dataset_file, 'w') as f:
        f.write(sample_data)
    
    print(f"✅ Created dataset with 20 samples")

def create_runtime_config(epochs=5, batch_size=1, timeout=15):
    """Create optimized runtime configuration"""
    config = {
        "model": {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 32,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "client_initial_layers": [0, 1],
            "server_middle_layers": [15],  # Only 1 layer for maximum speed
            "client_final_layers": [30, 31]
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": 1e-4,
            "max_epochs": epochs,
            "convergence_threshold": 0.5,
            "patience": 2,
            "gradient_clip_norm": 0.5,
            "warmup_steps": 5,
            "weight_decay": 0.01,
            "language_modeling_weight": 1.0,
            "length_penalty_weight": 0.0,
            "min_delta": 0.02,
            "restore_best_weights": True
        },
        "quantization": {
            "enabled": True,
            "epsilon": 1e-8,
            "bits": 1,
            "compression_ratio": 0.95
        },
        "galore": {
            "enabled": True,
            "rank": 2,
            "update_proj_gap": 5,
            "scale": 0.01,
            "min_param_size": 25
        },
        "data": {
            "data_dir": "./data",
            "dataset_file": "medical_qa_dataset.json",
            "max_sequence_length": 50,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "shuffle": True,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1
        },
        "server": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "timeout": timeout,
            "max_content_length": 50 * 1024 * 1024,
            "max_processing_time": timeout - 5,
            "enable_threading": True,
            "thread_timeout": timeout
        },
        "client": {
            "server_url": "http://localhost:5000",
            "request_timeout": timeout,
            "max_retries": 2,
            "retry_delay": 0.5,
            "use_compression": True,
            "verify_ssl": False,
            "max_server_failures": 2,
            "server_communication_interval": 20,
            "gradient_accumulation_steps": 25
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_handler": True,
            "console_handler": True,
            "log_dir": "./logs",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5
        },
        "device": {
            "device": "auto",
            "mixed_precision": False,
            "compile_model": False,
            "memory_fraction": 0.6,
            "clear_cache_frequency": 3
        },
        "evaluation": {
            "eval_frequency": 5,
            "save_best_model": True,
            "compute_bleu": False,
            "compute_rouge": False,
            "compute_perplexity": True,
            "max_new_tokens": 20,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "performance": {
            "clear_cache_frequency": 3,
            "gradient_checkpointing": False,
            "max_gradient_size": 5000,
            "compression_level": 6,
            "max_batch_processing_time": timeout - 5,
            "max_server_processing_time": timeout - 10,
            "enable_local_fallback": True,
            "fallback_threshold": 2
        }
    }
    
    # Save runtime config
    Path("config").mkdir(exist_ok=True)
    config_file = Path("config/runtime_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_file

def start_server(gpu_id, config_file):
    """Start the server process"""
    print(f"🖥️  Starting server on GPU {gpu_id}...")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['FEDERATED_CONFIG_FILE'] = str(config_file)
    
    try:
        process = subprocess.Popen(
            [sys.executable, "server/federated_server_v2.py"],
            env=env,
            stdout=None,
            stderr=None
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Test if server is responding
        import requests
        for i in range(10):
            try:
                response = requests.get("http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is running!")
                    return process
            except:
                time.sleep(1)
        
        print("⚠️  Server might not be responding, but continuing...")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def start_client(gpu_id, config_file, memory_fraction=None):
    """Start the client process"""
    print(f"📱 Starting client on GPU {gpu_id}...")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['FEDERATED_CONFIG_FILE'] = str(config_file)
    
    if memory_fraction:
        env['CUDA_MEMORY_FRACTION'] = str(memory_fraction)
    
    try:
        process = subprocess.Popen(
            [sys.executable, "client/federated_client_v2.py"],
            env=env,
            stdout=None,
            stderr=None
        )
        
        print("✅ Client started - training will begin...")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        return None

def cleanup_processes(*processes):
    """Clean up all processes"""
    print("🧹 Cleaning up processes...")
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    print("✅ Cleanup complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Federated Medical QA System V2 - Self-Contained Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_federated_v2.py --epochs 5 --gpu 0
  python run_federated_v2.py --epochs 3 --gpu 0,1
  python run_federated_v2.py --epochs 10 --gpu 0 --timeout 30
        """
    )
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU devices (e.g., '0' or '0,1')")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout in seconds")
    parser.add_argument("--mode", choices=["full", "server", "client"], default="full", help="Run mode")
    
    args = parser.parse_args()
    
    # Parse GPU configuration
    gpu_list = [int(x.strip()) for x in args.gpu.split(',') if x.strip()]
    server_gpu = gpu_list[0]
    client_gpu = gpu_list[1] if len(gpu_list) > 1 else gpu_list[0]
    memory_fraction = 0.5 if len(gpu_list) == 1 else None
    
    print("🏥 FEDERATED MEDICAL QA SYSTEM V2 - SELF-CONTAINED")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timeout: {args.timeout}s")
    print(f"Server GPU: {server_gpu}")
    print(f"Client GPU: {client_gpu}")
    if memory_fraction:
        print(f"Memory sharing: {memory_fraction * 100}%")
    print("=" * 60)
    
    # Setup
    print("🔧 Setting up...")
    
    # Create necessary directories
    for dir_name in ["data", "logs", "config", "server", "client"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Create sample data
    create_sample_data()
    
    # Create runtime configuration
    config_file = create_runtime_config(args.epochs, args.batch_size, args.timeout)
    
    print("✅ Setup complete")
    print("=" * 60)
    print("🚀 STARTING TRAINING...")
    print("=" * 60)
    
    # Check if required files exist
    server_file = Path("server/federated_server_v2.py")
    client_file = Path("client/federated_client_v2.py")
    
    if not server_file.exists():
        print("❌ server/federated_server_v2.py not found!")
        print("Please copy federated_server_v2.py to server/ directory")
        return 1
    
    if not client_file.exists():
        print("❌ client/federated_client_v2.py not found!")
        print("Please copy federated_client_v2.py to client/ directory")
        return 1
    
    server_process = None
    client_process = None
    
    try:
        if args.mode in ["full", "server"]:
            server_process = start_server(server_gpu, config_file)
            if not server_process:
                return 1
        
        if args.mode in ["full", "client"]:
            client_process = start_client(client_gpu, config_file, memory_fraction)
            if not client_process:
                return 1
        
        if args.mode == "full":
            print("⏳ Training in progress...")
            print("   (Training logs will appear in the processes above)")
            print("   Press Ctrl+C to stop")
            
            # Wait for client to finish
            if client_process:
                client_process.wait()
            
            print("🎉 Training completed!")
            
        else:
            print("⏳ Process running... Press Ctrl+C to stop")
            if server_process:
                server_process.wait()
            if client_process:
                client_process.wait()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    finally:
        cleanup_processes(server_process, client_process)
    
    print("\n🎯 SYSTEM COMPLETED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
