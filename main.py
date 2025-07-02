# main.py
"""
Main launcher for Federated Medical QA System
Comprehensive launcher with multiple run modes and configuration options
"""

import argparse
import subprocess
import sys
import time
import requests
import threading
import os
import signal
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import FederatedConfig, get_config
from dataset.medical_qa_downloader import MedicalQADownloader

class FederatedSystemLauncher:
    """
    Main launcher for the federated medical QA system
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.server_process = None
        self.client_process = None
        
        # Setup logging
        self.setup_logging()
        
        # System state
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_dir / "launcher.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.cleanup()
        sys.exit(0)
    
    def print_banner(self):
        """Print system banner"""
        print("\n" + "="*80)
        print("🏥 FEDERATED MEDICAL QA SYSTEM")
        print("Split LLM Architecture with Privacy-Preserving Learning")
        print("="*80)
        
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
        print(f"   Model: {self.config.model.hidden_size}d, {self.config.model.num_hidden_layers} layers")
        print(f"   Server: {self.config.client.server_url}")
        print(f"   Data: {self.config.data.data_dir}")
        print(f"   Training: {self.config.training.max_epochs} epochs, lr={self.config.training.learning_rate}")
        print("="*80)
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        required_packages = [
            'torch', 'flask', 'requests', 'numpy', 'pandas', 'transformers'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
            print("Install with: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies are installed")
        return True
    
    def setup_dataset(self) -> bool:
        """Setup the medical QA dataset"""
        try:
            print("📊 Setting up medical QA dataset...")
            
            dataset_path = Path(self.config.data.data_dir) / self.config.data.dataset_file
            
            if not dataset_path.exists():
                print("📥 Dataset not found. Downloading...")
                downloader = MedicalQADownloader(self.config.data.data_dir)
                dataset_path, stats = downloader.download_and_process()
                
                print(f"✅ Dataset ready: {stats['total_questions']} questions")
                print(f"   Categories: {list(stats['categories'].keys())}")
            else:
                print(f"✅ Dataset found: {dataset_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset setup failed: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start the federated learning server"""
        try:
            print("🖥️  Starting federated learning server...")
            
            # Start server process
            cmd = [sys.executable, "server/federated_server.py"]
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # Wait for server to start
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(
                        f"{self.config.client.server_url}/health",
                        timeout=2
                    )
                    if response.status_code == 200:
                        print("✅ Server started successfully!")
                        return True
                except:
                    pass
                
                time.sleep(1)
                if i % 5 == 0:
                    print(f"   Waiting for server... ({i+1}/{max_retries})")
            
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def run_client(self) -> bool:
        """Run the federated learning client"""
        try:
            print("📱 Starting federated learning client...")
            
            # Run client
            cmd = [sys.executable, "client/federated_client.py"]
            self.client_process = subprocess.run(
                cmd,
                env=os.environ.copy(),
                timeout=self.config.training.max_epochs * 120  # 2 minutes per epoch
            )
            
            if self.client_process.returncode == 0:
                print("✅ Client training completed successfully!")
                return True
            else:
                print(f"⚠️  Client exited with code {self.client_process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Client training timeout")
            return False
        except Exception as e:
            print(f"❌ Client failed: {e}")
            return False
    
    def run_server_only(self) -> bool:
        """Run only the server"""
        print("🖥️  FEDERATED LEARNING SERVER")
        print("Running in standalone mode...")
        print("-" * 50)
        
        try:
            # Import and run server directly
            from server.federated_server import main as server_main
            server_main()
            return True
        except KeyboardInterrupt:
            print("\n⏹️  Server stopped by user")
            return True
        except Exception as e:
            print(f"❌ Server error: {e}")
            return False
    
    def run_client_only(self) -> bool:
        """Run only the client (assumes server is running)"""
        print("📱 FEDERATED LEARNING CLIENT")
        print("Connecting to existing server...")
        print("-" * 50)
        
        # Check if server is available
        try:
            response = requests.get(
                f"{self.config.client.server_url}/health",
                timeout=5
            )
            if response.status_code != 200:
                print(f"❌ Server not available at {self.config.client.server_url}")
                return False
        except:
            print(f"❌ Cannot connect to server at {self.config.client.server_url}")
            return False
        
        try:
            # Import and run client directly
            from client.federated_client import main as client_main
            client_main()
            return True
        except KeyboardInterrupt:
            print("\n⏹️  Client stopped by user")
            return True
        except Exception as e:
            print(f"❌ Client error: {e}")
            return False
    
    def run_full_system(self) -> bool:
        """Run the complete federated learning system"""
        print("🚀 STARTING COMPLETE FEDERATED SYSTEM")
        print("-" * 50)
        
        # Setup dataset
        if not self.setup_dataset():
            return False
        
        # Start server
        if not self.start_server():
            return False
        
        try:
            # Run client training
            success = self.run_client()
            
            if success:
                print("\n🎉 FEDERATED TRAINING COMPLETED SUCCESSFULLY!")
                print("📊 Check logs for detailed metrics and results")
            else:
                print("\n⚠️  Training completed with issues")
            
            return success
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup processes and resources"""
        print("\n🧹 Cleaning up...")
        
        # Terminate client process
        if self.client_process and hasattr(self.client_process, 'terminate'):
            try:
                self.client_process.terminate()
                self.client_process.wait(timeout=5)
            except:
                pass
        
        # Terminate server process
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("✅ Server process terminated")
            except Exception as e:
                print(f"⚠️  Server cleanup warning: {e}")
                # Force kill if needed
                try:
                    self.server_process.kill()
                except:
                    pass
    
    def run_tests(self) -> bool:
        """Run system tests"""
        print("🧪 RUNNING SYSTEM TESTS")
        print("-" * 50)
        
        success = True
        
        try:
            # Test dataset downloader
            print("📊 Testing dataset downloader...")
            downloader = MedicalQADownloader("./test_data")
            dataset_path, stats = downloader.download_and_process()
            print(f"✅ Dataset test passed: {stats['total_questions']} questions")
            
            # Test tokenizer
            print("🔤 Testing tokenizer...")
            from utils.tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer(vocab_size=1000)
            test_text = "What are the symptoms of diabetes?"
            result = tokenizer([test_text], max_length=50, return_tensors="pt")
            print(f"✅ Tokenizer test passed: {result['input_ids'].shape}")
            
            # Test metrics
            print("📈 Testing metrics...")
            from utils.metrics import MedicalQAMetrics
            metrics = MedicalQAMetrics()
            metrics.update(2.5, ["Test question?"], ["Test answer."], ["Test prediction."])
            print("✅ Metrics test passed")
            
            # Test configuration
            print("⚙️  Testing configuration...")
            config = get_config()
            assert config.model.hidden_size > 0
            print("✅ Configuration test passed")
            
            print("\n🎉 ALL TESTS PASSED!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            success = False
        
        # Cleanup test data
        import shutil
        if Path("./test_data").exists():
            shutil.rmtree("./test_data")
        
        return success
    
    def show_status(self) -> bool:
        """Show system status"""
        print("📊 SYSTEM STATUS")
        print("-" * 50)
        
        # Check server availability
        try:
            response = requests.get(
                f"{self.config.client.server_url}/status",
                timeout=5
            )
            if response.status_code == 200:
                status = response.json()
                print("🖥️  Server Status: ✅ Online")
                print(f"   Initialized: {status.get('status', 'unknown')}")
                if 'layers' in status:
                    print(f"   Layers: {status['layers']}")
                if 'parameters' in status:
                    print(f"   Parameters: {status['parameters']:,}")
            else:
                print("🖥️  Server Status: ❌ Error")
        except:
            print("🖥️  Server Status: ❌ Offline")
        
        # Check dataset
        dataset_path = Path(self.config.data.data_dir) / self.config.data.dataset_file
        if dataset_path.exists():
            print("📊 Dataset: ✅ Available")
            print(f"   Path: {dataset_path}")
        else:
            print("📊 Dataset: ❌ Not found")
        
        # Check logs
        log_dir = Path(self.config.logging.log_dir)
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"📝 Logs: ✅ {len(log_files)} files in {log_dir}")
        else:
            print("📝 Logs: ❌ No log directory")
        
        # Check dependencies
        print("📦 Dependencies:", end=" ")
        if self.check_dependencies():
            print("✅ All installed")
        else:
            print("❌ Missing packages")
        
        return True

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Federated Medical QA System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                    # Run complete system
  python main.py --mode server                  # Run server only
  python main.py --mode client                  # Run client only (server must be running)
  python main.py --mode test                    # Run system tests
  python main.py --mode status                  # Show system status
  
  python main.py --config custom_config.json    # Use custom configuration
  python main.py --mode full --epochs 20        # Override training epochs
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["full", "server", "client", "test", "status"],
        default="full",
        help="Run mode (default: full)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override maximum training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        help="Override server URL"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override data directory"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner and reduce output"
    )
    
    return parser

def main():
    """Main function"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = FederatedConfig.load(args.config)
    else:
        config = get_config()
    
    # Override configuration with command line arguments
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.server_url:
        config.client.server_url = args.server_url
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.verbose:
        config.logging.level = "DEBUG"
    
    # Create launcher
    launcher = FederatedSystemLauncher(config)
    
    # Show banner unless quiet mode
    if not args.quiet:
        launcher.print_banner()
    
    # Check dependencies
    if not launcher.check_dependencies():
        sys.exit(1)
    
    # Run based on mode
    success = False
    
    try:
        if args.mode == "full":
            success = launcher.run_full_system()
        elif args.mode == "server":
            success = launcher.run_server_only()
        elif args.mode == "client":
            success = launcher.run_client_only()
        elif args.mode == "test":
            success = launcher.run_tests()
        elif args.mode == "status":
            success = launcher.show_status()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        launcher.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        launcher.cleanup()
        sys.exit(1)
    
    # Cleanup and exit
    launcher.cleanup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
