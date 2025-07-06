#!/usr/bin/env python3
"""
Setup script for Federated Medical QA System - VERSION 2
Automated setup and testing
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "logs", 
        "checkpoints",
        "config",
        "server",
        "client",
        "utils",
        "dataset"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def create_sample_data():
    """Create sample medical QA data"""
    data_dir = Path("data")
    dataset_file = data_dir / "dataset.csv"
    
    if dataset_file.exists():
        print(f"✅ Dataset already exists: {dataset_file}")
        return
    
    print("📋 Creating sample medical QA dataset...")
    
    sample_data = """question,answer,category
What are the symptoms of diabetes?,"Common symptoms include increased thirst, frequent urination, extreme fatigue, blurred vision, and slow-healing cuts or wounds.",diabetes
How is hypertension diagnosed?,"Hypertension is diagnosed through blood pressure measurements. A reading of 130/80 mmHg or higher is considered high blood pressure.",hypertension
What causes heart attacks?,"Heart attacks are usually caused by coronary artery disease, where plaque builds up in the arteries that supply blood to the heart.",cardiology
How can asthma be managed?,"Asthma can be managed through avoiding triggers, using prescribed medications like inhalers, and following an asthma action plan.",respiratory
What are the signs of depression?,"Signs include persistent sadness, loss of interest in activities, changes in appetite, sleep problems, and difficulty concentrating.",mental_health
How is pneumonia treated?,"Pneumonia is typically treated with antibiotics for bacterial infections, rest, fluids, and sometimes hospitalization for severe cases.",infectious_disease
What causes kidney stones?,"Kidney stones form when minerals and salts crystallize in the kidneys, often due to dehydration, diet, or genetic factors.",nephrology
How is cancer detected?,"Cancer can be detected through various methods including physical exams, imaging tests, blood tests, and tissue biopsies.",oncology
What is the treatment for arthritis?,"Arthritis treatment includes medications, physical therapy, exercise, weight management, and sometimes surgery.",rheumatology
How can food poisoning be prevented?,"Food poisoning can be prevented by proper hand washing, cooking food thoroughly, storing food at correct temperatures, and avoiding cross-contamination.",gastroenterology
What are the risk factors for stroke?,"Risk factors include high blood pressure, diabetes, smoking, obesity, high cholesterol, and atrial fibrillation.",neurology
How is thyroid disease diagnosed?,"Thyroid disease is diagnosed through blood tests measuring TSH, T3, and T4 levels, along with physical examination.",endocrinology
What causes migraines?,"Migraines can be triggered by stress, hormonal changes, certain foods, lack of sleep, and environmental factors.",neurology
How is diabetes managed?,"Diabetes is managed through blood sugar monitoring, medication or insulin, healthy diet, regular exercise, and medical check-ups.",diabetes
What are the symptoms of anxiety?,"Symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, muscle tension, and sleep problems.",mental_health
How is high cholesterol treated?,"High cholesterol is treated with lifestyle changes including diet and exercise, and medications like statins when necessary.",cardiology
What causes allergic reactions?,"Allergic reactions occur when the immune system overreacts to harmless substances like pollen, foods, or medications.",immunology
How is pneumonia diagnosed?,"Pneumonia is diagnosed through chest X-rays, blood tests, sputum cultures, and physical examination of the lungs.",respiratory
What are the signs of kidney disease?,"Signs include fatigue, swelling, changes in urination, nausea, and high blood pressure.",nephrology
How can heart disease be prevented?,"Heart disease can be prevented through healthy diet, regular exercise, not smoking, managing stress, and controlling blood pressure.",cardiology
What is the treatment for migraines?,"Migraine treatment includes pain relievers, preventive medications, lifestyle changes, and avoiding known triggers.",neurology
How is arthritis diagnosed?,"Arthritis is diagnosed through physical examination, blood tests, imaging studies, and sometimes joint fluid analysis.",rheumatology
What causes high blood pressure?,"High blood pressure can be caused by genetics, age, diet, lack of exercise, stress, and underlying medical conditions.",cardiology
How is asthma diagnosed?,"Asthma is diagnosed through medical history, physical examination, lung function tests, and sometimes allergy testing.",respiratory
What are the symptoms of kidney stones?,"Symptoms include severe pain in the back or side, nausea, vomiting, blood in urine, and frequent urination.",nephrology"""
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"✅ Sample dataset created: {dataset_file}")
    print(f"📊 Contains 25 medical QA pairs")

def create_required_modules():
    """Create required utility modules"""
    print("🔧 Creating required modules...")
    
    # Create utils/__init__.py
    utils_init = Path("utils/__init__.py")
    utils_init.write_text("")
    
    # Create simple tokenizer
    tokenizer_file = Path("utils/tokenizer.py")
    tokenizer_code = '''"""
Simple tokenizer for medical QA system
"""
import torch
import re
from typing import List, Dict, Union

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization by splitting on whitespace and punctuation"""
        # Basic tokenization
        tokens = re.findall(r'\\w+|[^\\w\\s]', text.lower())
        
        # Convert to IDs (simple hash-based approach)
        token_ids = []
        for token in tokens:
            token_id = hash(token) % (self.vocab_size - 10) + 10  # Reserve first 10 IDs
            token_ids.append(token_id)
        
        return token_ids
    
    def __call__(self, texts: Union[str, List[str]], max_length: int = 512, 
                 padding: bool = True, truncation: bool = True, 
                 return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize and prepare tensors"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.tokenize(text)
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length - 1:
                token_ids = token_ids[:max_length - 1]
            
            # Add EOS token
            token_ids.append(self.eos_token_id)
            
            all_token_ids.append(token_ids)
        
        # Pad sequences
        if padding:
            max_len = max(len(ids) for ids in all_token_ids)
            max_len = min(max_len, max_length)
            
            for i, token_ids in enumerate(all_token_ids):
                if len(token_ids) < max_len:
                    token_ids.extend([self.pad_token_id] * (max_len - len(token_ids)))
                elif len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                all_token_ids[i] = token_ids
        
        # Convert to tensors
        if return_tensors == "pt":
            input_ids = torch.tensor(all_token_ids, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
        return {"input_ids": all_token_ids}
'''
    tokenizer_file.write_text(tokenizer_code)
    
    # Create metrics module
    metrics_file = Path("utils/metrics.py")
    metrics_code = '''"""
Metrics for medical QA system
"""
import torch
import numpy as np
from typing import List

class MedicalQAMetrics:
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.losses = []
        
    def update(self, loss: float, questions: List[str], answers: List[str]):
        """Update metrics with new batch"""
        self.total_loss += loss
        self.total_samples += len(questions)
        self.losses.append(loss)
        
        # Keep only last 100 losses
        if len(self.losses) > 100:
            self.losses = self.losses[-100:]
    
    def get_average_loss(self) -> float:
        """Get average loss"""
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples
    
    def get_recent_loss(self) -> float:
        """Get recent average loss"""
        if not self.losses:
            return 0.0
        return np.mean(self.losses[-10:])  # Last 10 losses
    
    def reset(self):
        """Reset metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.losses = []
'''
    metrics_file.write_text(metrics_code)
    
    # Create dataset module
    dataset_init = Path("dataset/__init__.py")
    dataset_init.write_text("")
    
    dataset_file = Path("dataset/medical_qa_downloader.py")
    dataset_code = '''"""
Medical QA dataset downloader and processor
"""
import json
import pandas as pd
from pathlib import Path
import logging

class MedicalQADownloader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
    def download_and_process(self):
        """Download and process medical QA dataset"""
        try:
            # Look for CSV file
            csv_file = self.data_dir / "dataset.csv"
            
            if not csv_file.exists():
                self.logger.error(f"Dataset file not found: {csv_file}")
                return None, {"total_questions": 0}
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Convert to JSON format
            dataset = []
            for _, row in df.iterrows():
                dataset.append({
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                    "category": str(row.get("category", "general"))
                })
            
            # Save as JSON
            json_file = self.data_dir / "medical_qa_dataset.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            stats = {
                "total_questions": len(dataset),
                "categories": len(set(item["category"] for item in dataset))
            }
            
            self.logger.info(f"Processed {len(dataset)} QA pairs")
            return json_file, stats
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            return None, {"total_questions": 0}
'''
    dataset_file.write_text(dataset_code)
    
    print("✅ Required modules created")

def copy_version_files():
    """Copy the v2 files to their correct locations"""
    print("📄 Setting up version 2 files...")
    
    # Note: In a real setup, these files would be copied from the artifacts
    # For now, we just create the directory structure
    
    # Create server directory
    server_dir = Path("server")
    server_dir.mkdir(exist_ok=True)
    
    # Create client directory  
    client_dir = Path("client")
    client_dir.mkdir(exist_ok=True)
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    print("✅ Directory structure ready")
    print("📝 Please copy the v2 files to their respective directories:")
    print("   - main_v2.py -> ./")
    print("   - config_v2.py -> config/")
    print("   - federated_server_v2.py -> server/")
    print("   - federated_client_v2.py -> client/")

def test_setup():
    """Test the setup"""
    print("🔍 Testing setup...")
    
    try:
        # Test imports
        import torch
        print("✅ PyTorch available")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
        else:
            print("⚠️  CUDA not available, will use CPU")
        
        # Test data
        data_file = Path("data/dataset.csv")
        if data_file.exists():
            print("✅ Dataset file exists")
        else:
            print("❌ Dataset file missing")
        
        # Test modules
        sys.path.append(".")
        from utils.tokenizer import SimpleTokenizer
        from utils.metrics import MedicalQAMetrics
        from dataset.medical_qa_downloader import MedicalQADownloader
        
        print("✅ All modules importable")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 FEDERATED MEDICAL QA SYSTEM - VERSION 2 SETUP")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data() 
    
    # Create required modules
    create_required_modules()
    
    # Setup version files
    copy_version_files()
    
    # Test setup
    if test_setup():
        print("\n✅ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 NEXT STEPS:")
        print("1. Copy the v2 files to their directories")
        print("2. Run: python main_v2.py --mode full --epochs 5")
        print("=" * 60)
    else:
        print("\n❌ SETUP FAILED")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    main()
