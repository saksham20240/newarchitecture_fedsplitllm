#!/usr/bin/env python3
"""
Main script for Federated Medical QA System - VERSION 2 FIXED
Fixed import issues and configuration handling
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FIXED: Direct import to avoid config.__init__.py conflicts
try:
    from config.config_v2 import get_config, get_optimized_config
except ImportError:
    # Fallback: Try to import from current directory
    sys.path.insert(0, 'config')
    from config_v2 import get_config, get_optimized_config

class FederatedSystemManager:
    """Manages the federated learning system components - VERSION 2"""
    
    def __init__(self, config_type: str = "default"):
        self.config = get_optimized_config() if config_type == "optimized" else get_config()
        self.server_process: Optional[subprocess.Popen] = None
        self.client_process: Optional[subprocess.Popen] = None
        
        # GPU device management
        self.available_gpus = self.get_available_gpus()
        self.server_device = None
        self.client_device = None
        
        # Setup logging
        self.setup_logging()
        
        # Signal handlers for cleanup
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def get_available_gpus(self):
        """Get available GPU devices"""
        try:
            import torch
            if torch.cuda.is_available():
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if cuda_visible:
                    return [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
                else:
                    return list(range(torch.cuda.device_count()))
            return []
        except:
            return []
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_dir / "main_v2.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def save_runtime_config(self, config_overrides: dict):
        """Save runtime configuration with overrides"""
        runtime_config = self.config.to_dict()
        
        # Apply overrides
        for key, value in config_overrides.items():
            if key == "epochs":
                runtime_config["training"]["max_epochs"] = value
            elif key == "batch_size":
                runtime_config["training"]["batch_size"] = value
            elif key == "timeout":
                runtime_config["server"]["timeout"] = value
                runtime_config["client"]["request_timeout"] = value
            elif key == "learning_rate":
                runtime_config["training"]["learning_rate"] = value
        
        # Save to temporary file
        config_file = Path("config/runtime_config.json")
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(runtime_config, f, indent=2)
        
        self.logger.info(f"Runtime configuration saved to {config_file}")
        return config_file
    
    def start_server(self, config_overrides: dict = None) -> bool:
        """Start the federated learning server"""
        try:
            self.logger.info("🖥️  Starting federated learning server...")
            
            # Set GPU device for server
            server_env = os.environ.copy()
            if self.available_gpus:
                self.server_device = self.available_gpus[0]
                server_env['CUDA_VISIBLE_DEVICES'] = str(self.server_device)
                self.logger.info(f"   Server will use GPU {self.server_device}")
            else:
                server_env['CUDA_VISIBLE_DEVICES'] = ''
                self.logger.info("   Server will use CPU")
            
            # Save runtime config
            if config_overrides:
                config_file = self.save_runtime_config(config_overrides)
                server_env['FEDERATED_CONFIG_FILE'] = str(config_file)
            
            # Start server process
            self.server_process = subprocess.Popen(
                [sys.executable, "server/federated_server_v2.py"],
                env=server_env,
                stdout=None,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait for server to start
            self.logger.info("   Waiting for server to start...")
            max_wait = 30
            for i in range(max_wait):
                if self.server_process.poll() is not None:
                    self.logger.error("❌ Server process ended unexpectedly")
                    return False
                
                # Check if server is responding
                try:
                    import requests
                    response = requests.get(
                        f"{self.config.client.server_url}/health",
                        timeout=2
                    )
                    if response.status_code == 200:
                        self.logger.info("✅ Server is responding!")
                        return True
                except:
                    pass
                
                time.sleep(1)
                if i % 5 == 0 and i > 0:
                    self.logger.info(f"   Still waiting... ({i}/{max_wait})")
            
            self.logger.error("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}")
            return False
    
    def start_client(self, config_overrides: dict = None) -> bool:
        """Start the federated learning client"""
        try:
            self.logger.info("📱 Starting federated learning client...")
            
            # Set GPU device for client
            client_env = os.environ.copy()
            if self.available_gpus:
                if len(self.available_gpus) > 1:
                    # Use second GPU if available
                    self.client_device = self.available_gpus[1]
                    client_env['CUDA_VISIBLE_DEVICES'] = str(self.client_device)
                    self.logger.info(f"   Client will use GPU {self.client_device}")
                else:
                    # Share GPU with server but use different memory fraction
                    self.client_device = self.available_gpus[0]
                    client_env['CUDA_VISIBLE_DEVICES'] = str(self.client_device)
                    client_env['CUDA_MEMORY_FRACTION'] = '0.5'  # Use half GPU memory
                    self.logger.info(f"   Client will share GPU {self.client_device} with server (50% memory)")
            else:
                client_env['CUDA_VISIBLE_DEVICES'] = ''
                self.logger.info("   Client will use CPU")
            
            # Save runtime config
            if config_overrides:
                config_file = self.save_runtime_config(config_overrides)
                client_env['FEDERATED_CONFIG_FILE'] = str(config_file)
            
            # Start client process
            self.client_process = subprocess.Popen(
                [sys.executable, "client/federated_client_v2.py"],
                env=client_env,
                stdout=None,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.logger.info("✅ Client started - training logs will appear below:")
            self.logger.info("=" * 80)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start client: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor server and client processes"""
        try:
            self.logger.info("🔍 Monitoring processes...")
            
            while True:
                # Check server status
                if self.server_process and self.server_process.poll() is not None:
                    self.logger.error("❌ Server process ended unexpectedly")
                    break
                
                # Check client status
                if self.client_process and self.client_process.poll() is not None:
                    if self.client_process.returncode == 0:
                        self.logger.info("=" * 80)
                        self.logger.info("✅ Client training completed successfully!")
                    else:
                        self.logger.error(f"❌ Client process ended with error (code: {self.client_process.returncode})")
                    break
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Received interrupt signal")
    
    def cleanup(self):
        """Clean up processes"""
        self.logger.info("🧹 Cleaning up...")
        
        # Stop client process
        if self.client_process and self.client_process.poll() is None:
            self.logger.info("Stopping client process...")
            self.client_process.terminate()
            try:
                self.client_process.wait(timeout=5)
                self.logger.info("✅ Client process stopped")
            except subprocess.TimeoutExpired:
                self.logger.warning("Force killing client process...")
                self.client_process.kill()
                self.client_process.wait()
        
        # Stop server process
        if self.server_process and self.server_process.poll() is None:
            self.logger.info("Stopping server process...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                self.logger.info("✅ Server process stopped")
            except subprocess.TimeoutExpired:
                self.logger.warning("Force killing server process...")
                self.server_process.kill()
                self.server_process.wait()
        
        # Clean up temporary config file
        config_file = Path("config/runtime_config.json")
        if config_file.exists():
            config_file.unlink()
    
    def run_full_system(self, config_overrides: dict = None):
        """Run the complete federated learning system"""
        try:
            self.logger.info("🚀 STARTING COMPLETE FEDERATED SYSTEM V2")
            self.logger.info("=" * 50)
            
            # Show configuration
            if config_overrides:
                self.logger.info("📊 Configuration overrides:")
                for key, value in config_overrides.items():
                    self.logger.info(f"   {key}: {value}")
            
            # Show GPU allocation
            if self.available_gpus:
                self.logger.info(f"📊 Available GPUs: {self.available_gpus}")
            else:
                self.logger.info("📊 No GPUs available, using CPU")
            
            # Start server
            if not self.start_server(config_overrides):
                self.logger.error("Failed to start server")
                return False
            
            # Small delay for server initialization
            time.sleep(2)
            
            # Start client
            if not self.start_client(config_overrides):
                self.logger.error("Failed to start client")
                return False
            
            # Monitor processes
            self.monitor_processes()
            
            return True
            
        except Exception as e:
            self.logger.error(f"System error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()
    
    def run_server_only(self, config_overrides: dict = None):
        """Run only the server component"""
        try:
            self.logger.info("🖥️  STARTING SERVER ONLY V2")
            self.logger.info("=" * 30)
            
            if not self.start_server(config_overrides):
                return False
            
            self.logger.info("Server is running. Press Ctrl+C to stop.")
            
            try:
                while True:
                    if self.server_process.poll() is not None:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("🛑 Stopping server...")
            
            return True
            
        finally:
            self.cleanup()
    
    def run_client_only(self, config_overrides: dict = None):
        """Run only the client component"""
        try:
            self.logger.info("📱 STARTING CLIENT ONLY V2")
            self.logger.info("=" * 30)
            
            if not self.start_client(config_overrides):
                return False
            
            while True:
                if self.client_process.poll() is not None:
                    if self.client_process.returncode == 0:
                        self.logger.info("✅ Training completed successfully!")
                    else:
                        self.logger.error(f"❌ Training failed (code: {self.client_process.returncode})")
                    break
                time.sleep(1)
            
            return True
            
        finally:
            self.cleanup()

def main():
    """Main function with improved argument parsing"""
    parser = argparse.ArgumentParser(
        description="Federated Medical QA System - VERSION 2 FIXED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_v2_fixed.py --mode full --epochs 5 --batch-size 1
  python main_v2_fixed.py --mode server --timeout 60
  python main_v2_fixed.py --mode client --learning-rate 0.0001
        """
    )
    
    parser.add_argument("--mode", choices=["full", "server", "client"], default="full")
    parser.add_argument("--config", choices=["default", "optimized"], default="optimized")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--gpu-server", type=int, help="GPU device for server")
    parser.add_argument("--gpu-client", type=int, help="GPU device for client")
    
    args = parser.parse_args()
    
    # Print system information
    print("=" * 80)
    print("🏥 FEDERATED MEDICAL QA SYSTEM - VERSION 2 FIXED")
    print("Split LLM Architecture with Privacy-Preserving Learning")
    print("=" * 80)
    
    print("\n📋 SYSTEM ARCHITECTURE:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    CLIENT (Medical)                        │")
    print("│  🔸 Layers 0-1 (Initial) + 30-31 (Final)                   │")
    print("│  🔸 Medical data processing & privacy preservation          │")
    print("│  🔸 1-bit quantization & GaLore compression                │")
    print("└─────────────────────┬───────────────────────────────────────┘")
    print("                      │ Encrypted Communication")
    print("                      ▼")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    SERVER                                   │")
    print("│  🔸 Layers 3-30 (Middle processing)                        │")
    print("│  🔸 Heavy computation & gradient optimization               │")
    print("│  🔸 No access to raw medical data                          │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Mode: {args.mode}")
    print(f"   Config: {args.config}")
    
    # Build configuration overrides
    config_overrides = {}
    if args.epochs:
        config_overrides["epochs"] = args.epochs
        print(f"   Epochs: {args.epochs}")
    if args.batch_size:
        config_overrides["batch_size"] = args.batch_size
        print(f"   Batch size: {args.batch_size}")
    if args.timeout:
        config_overrides["timeout"] = args.timeout
        print(f"   Timeout: {args.timeout}s")
    if args.learning_rate:
        config_overrides["learning_rate"] = args.learning_rate
        print(f"   Learning rate: {args.learning_rate}")
    
    # Show GPU information
    try:
        import torch
        if torch.cuda.is_available():
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
            print(f"   CUDA Devices: {cuda_visible}")
            print(f"   Available GPUs: {torch.cuda.device_count()}")
            
            # Show GPU memory
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                except:
                    pass
        else:
            print("   CUDA: Not available")
    except:
        print("   CUDA: Unknown")
    
    print("=" * 80)
    print("✅ All dependencies are installed")
    print("🚀 STARTING SYSTEM V2 - Training logs will appear below")
    print("=" * 80)
    
    # Create system manager
    try:
        manager = FederatedSystemManager(config_type=args.config)
        
        # Override GPU allocation if specified
        if args.gpu_server is not None:
            manager.available_gpus = [args.gpu_server]
            if args.gpu_client is not None:
                manager.available_gpus = [args.gpu_server, args.gpu_client]
        elif args.gpu_client is not None:
            manager.available_gpus = [args.gpu_client]
        
        # Run based on mode
        if args.mode == "full":
            success = manager.run_full_system(config_overrides)
        elif args.mode == "server":
            success = manager.run_server_only(config_overrides)
        elif args.mode == "client":
            success = manager.run_client_only(config_overrides)
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 SYSTEM COMPLETED SUCCESSFULLY!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ SYSTEM ENCOUNTERED ERRORS")
            print("=" * 80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
