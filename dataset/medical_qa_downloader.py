# dataset/medical_qa_downloader.py
"""
Medical QA Dataset Processor
Processes only the provided CSV dataset
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple
import logging
import time
from pathlib import Path
import numpy as np

class MedicalQADownloader:
    """
    Processes medical QA datasets from CSV files only
    """
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Expected CSV column mappings
        self.csv_column_mappings = {
            'question': ['question', 'Question', 'QUESTION', 'query', 'Query', 'input', 'Input', 'text', 'Text'],
            'answer': ['answer', 'Answer', 'ANSWER', 'response', 'Response', 'output', 'Output', 'target', 'Target'],
            'category': ['category', 'Category', 'CATEGORY', 'type', 'Type', 'subject', 'Subject', 'topic', 'Topic'],
            'difficulty': ['difficulty', 'Difficulty', 'DIFFICULTY', 'level', 'Level', 'complexity', 'Complexity'],
            'source': ['source', 'Source', 'SOURCE', 'dataset', 'Dataset', 'origin', 'Origin']
        }
    
    def find_csv_column(self, df: pd.DataFrame, target_field: str) -> str:
        """Find the appropriate column name for a target field"""
        possible_names = self.csv_column_mappings.get(target_field, [target_field])
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # If not found, return None
        return None
    
    def detect_csv_structure(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect the structure of the CSV file"""
        self.logger.info(f"Detecting CSV structure. Available columns: {list(df.columns)}")
        
        structure = {}
        
        # Find question column
        question_col = self.find_csv_column(df, 'question')
        if not question_col:
            # If no standard question column, use the first column
            question_col = df.columns[0]
            self.logger.warning(f"No standard question column found. Using first column: {question_col}")
        structure['question'] = question_col
        
        # Find answer column
        answer_col = self.find_csv_column(df, 'answer')
        if not answer_col:
            # If no standard answer column, use the second column if available
            if len(df.columns) > 1:
                answer_col = df.columns[1]
                self.logger.warning(f"No standard answer column found. Using second column: {answer_col}")
            else:
                # If only one column, duplicate it as answer
                answer_col = question_col
                self.logger.warning(f"Only one column found. Using same column for both question and answer")
        structure['answer'] = answer_col
        
        # Find optional columns
        structure['category'] = self.find_csv_column(df, 'category')
        structure['difficulty'] = self.find_csv_column(df, 'difficulty')
        structure['source'] = self.find_csv_column(df, 'source')
        
        self.logger.info(f"Detected structure: {structure}")
        return structure
    
    def load_csv_dataset(self, csv_path: str) -> List[Dict]:
        """Load dataset from CSV file"""
        try:
            self.logger.info(f"Loading CSV dataset from {csv_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to load with {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"Could not read CSV file with any of the tried encodings: {encodings}")
            
            self.logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
            
            # Detect CSV structure
            structure = self.detect_csv_structure(df)
            
            # Process data
            processed_data = []
            
            for idx, row in df.iterrows():
                try:
                    # Get question and answer
                    question = str(row[structure['question']]).strip()
                    answer = str(row[structure['answer']]).strip()
                    
                    # Skip rows with missing essential data
                    if pd.isna(row[structure['question']]) or pd.isna(row[structure['answer']]):
                        continue
                    
                    # Skip empty strings
                    if not question or not answer or question == 'nan' or answer == 'nan':
                        continue
                    
                    # Create data entry
                    data_entry = {
                        "question": question,
                        "answer": answer,
                        "category": "medical",  # Default category
                        "difficulty": "intermediate",  # Default difficulty
                        "source": "csv_dataset"  # Default source
                    }
                    
                    # Add optional fields if available
                    if structure['category'] and structure['category'] in row and not pd.isna(row[structure['category']]):
                        data_entry['category'] = str(row[structure['category']]).strip()
                    
                    if structure['difficulty'] and structure['difficulty'] in row and not pd.isna(row[structure['difficulty']]):
                        data_entry['difficulty'] = str(row[structure['difficulty']]).strip()
                    
                    if structure['source'] and structure['source'] in row and not pd.isna(row[structure['source']]):
                        data_entry['source'] = str(row[structure['source']]).strip()
                    
                    processed_data.append(data_entry)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {len(processed_data)} valid entries from CSV")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading CSV dataset: {e}")
            raise
    
    def find_csv_files(self) -> List[str]:
        """Find all CSV files in the data directory"""
        csv_files = []
        
        # Look for CSV files in the data directory
        for file_path in self.data_dir.glob("*.csv"):
            csv_files.append(str(file_path))
        
        # Also check for common CSV filenames
        common_names = ['dataset.csv', 'data.csv', 'medical_qa.csv', 'questions.csv', 'qa_pairs.csv']
        for name in common_names:
            file_path = self.data_dir / name
            if file_path.exists() and str(file_path) not in csv_files:
                csv_files.append(str(file_path))
        
        return csv_files
    
    def process_all_csv_files(self) -> List[Dict]:
        """Process all CSV files found in the data directory"""
        csv_files = self.find_csv_files()
        
        if not csv_files:
            self.logger.warning("No CSV files found in data directory")
            return []
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                self.logger.info(f"Processing CSV file: {csv_file}")
                data = self.load_csv_dataset(csv_file)
                all_data.extend(data)
                self.logger.info(f"Added {len(data)} entries from {csv_file}")
            except Exception as e:
                self.logger.error(f"Error processing {csv_file}: {e}")
                continue
        
        return all_data
    
    def save_dataset(self, data: List[Dict], filename: str = "medical_qa_dataset.json"):
        """Save processed dataset"""
        if not data:
            self.logger.warning("No data to save")
            return None
        
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(data)} questions to {filepath}")
            
            # Also save as CSV for easy viewing
            df = pd.DataFrame(data)
            csv_path = filepath.with_suffix('.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"Also saved as CSV: {csv_path}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
            raise
    
    def get_dataset_stats(self, data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        if not data:
            return {
                "total_questions": 0,
                "categories": {},
                "difficulties": {},
                "sources": {},
                "avg_question_length": 0,
                "avg_answer_length": 0
            }
        
        df = pd.DataFrame(data)
        
        stats = {
            "total_questions": len(data),
            "categories": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            "difficulties": df['difficulty'].value_counts().to_dict() if 'difficulty' in df.columns else {},
            "sources": df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            "avg_question_length": df['question'].str.len().mean() if 'question' in df.columns else 0,
            "avg_answer_length": df['answer'].str.len().mean() if 'answer' in df.columns else 0
        }
        
        return stats
    
    def download_and_process(self) -> Tuple[str, Dict]:
        """Main method to process CSV data"""
        self.logger.info("Starting CSV dataset processing...")
        
        # Process all CSV files
        all_data = self.process_all_csv_files()
        
        if not all_data:
            self.logger.error("No valid data found in CSV files")
            return None, {"total_questions": 0}
        
        # Save dataset
        dataset_path = self.save_dataset(all_data)
        
        if not dataset_path:
            return None, {"total_questions": 0}
        
        # Get statistics
        stats = self.get_dataset_stats(all_data)
        
        self.logger.info("Dataset processing complete!")
        self.logger.info(f"Total questions: {stats.get('total_questions', 0)}")
        self.logger.info(f"Categories: {list(stats.get('categories', {}).keys())}")
        
        return str(dataset_path), stats

def main():
    """Main function for testing"""
    downloader = MedicalQADownloader()
    dataset_path, stats = downloader.download_and_process()
    
    if dataset_path:
        print("\n" + "="*50)
        print("MEDICAL QA DATASET PROCESSING COMPLETE")
        print("="*50)
        print(f"Dataset saved to: {dataset_path}")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Categories: {list(stats['categories'].keys())}")
        print(f"Average question length: {stats['avg_question_length']:.1f} characters")
        print(f"Average answer length: {stats['avg_answer_length']:.1f} characters")
    else:
        print("\n" + "="*50)
        print("DATASET PROCESSING FAILED")
        print("="*50)
        print("Please check if CSV files are present in the data directory")

if __name__ == "__main__":
    main()
