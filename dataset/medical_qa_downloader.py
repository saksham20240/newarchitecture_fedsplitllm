"""
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
