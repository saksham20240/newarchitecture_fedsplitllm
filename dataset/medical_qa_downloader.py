# dataset/medical_qa_downloader.py
"""
Medical QA Dataset Downloader
Downloads and processes medical question-answering datasets from public sources
"""

import os
import json
import requests
import pandas as pd
from typing import List, Dict, Tuple
import logging
import time
from pathlib import Path

class MedicalQADownloader:
    """
    Downloads medical QA datasets from various public sources
    """
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dataset sources
        self.dataset_sources = {
            "medqa_sample": "https://raw.githubusercontent.com/jind11/MedQA/master/data_clean/questions/US/4_options/questions_clean.json",
            "pubmedqa_sample": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
            "medical_flashcards": "https://raw.githubusercontent.com/kbressem/medAlpaca/main/medical_meadow_medical_flashcards.json"
        }
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL"""
        try:
            self.logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def create_comprehensive_medical_dataset(self) -> List[Dict]:
        """Create a comprehensive medical QA dataset"""
        
        medical_qa_data = [
            # Diabetes
            {
                "question": "What are the early symptoms of type 2 diabetes?",
                "answer": "Early symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing cuts, frequent infections, and unexplained weight loss.",
                "category": "diabetes",
                "difficulty": "basic"
            },
            {
                "question": "How is diabetes mellitus diagnosed using laboratory tests?",
                "answer": "Diabetes is diagnosed through fasting glucose test (≥126 mg/dL), HbA1c test (≥6.5%), oral glucose tolerance test (≥200 mg/dL at 2 hours), or random glucose test (≥200 mg/dL) with symptoms.",
                "category": "diabetes",
                "difficulty": "intermediate"
            },
            {
                "question": "What are the long-term complications of uncontrolled diabetes?",
                "answer": "Long-term complications include diabetic retinopathy, nephropathy, neuropathy, cardiovascular disease, stroke, poor wound healing, and increased infection risk.",
                "category": "diabetes",
                "difficulty": "advanced"
            },
            {
                "question": "What lifestyle modifications help manage type 2 diabetes?",
                "answer": "Key modifications include regular exercise, balanced diet with controlled carbohydrates, weight management, blood sugar monitoring, medication adherence, stress management, and regular medical check-ups.",
                "category": "diabetes",
                "difficulty": "intermediate"
            },
            
            # Hypertension
            {
                "question": "What blood pressure values indicate different stages of hypertension?",
                "answer": "Normal: <120/80 mmHg. Elevated: 120-129/<80. Stage 1: 130-139/80-89. Stage 2: ≥140/90. Hypertensive crisis: >180/120 requiring immediate medical attention.",
                "category": "hypertension",
                "difficulty": "basic"
            },
            {
                "question": "What are the complications of untreated hypertension?",
                "answer": "Complications include heart attack, stroke, heart failure, kidney disease, vision problems, aneurysms, cognitive decline, and peripheral artery disease.",
                "category": "hypertension",
                "difficulty": "intermediate"
            },
            {
                "question": "What are the first-line antihypertensive medications?",
                "answer": "First-line medications include ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. Choice depends on patient factors, comorbidities, and contraindications.",
                "category": "hypertension",
                "difficulty": "advanced"
            },
            
            # Cardiovascular
            {
                "question": "What are the warning signs of a myocardial infarction?",
                "answer": "Warning signs include chest pain or pressure, shortness of breath, pain radiating to arms/neck/jaw/back, cold sweats, nausea, vomiting, lightheadedness, and fatigue.",
                "category": "cardiology",
                "difficulty": "basic"
            },
            {
                "question": "How is acute coronary syndrome managed in the emergency department?",
                "answer": "Management includes MONA (Morphine, Oxygen, Nitroglycerin, Aspirin), anticoagulation, beta-blockers, statins, and emergent reperfusion therapy (PCI or thrombolytics).",
                "category": "cardiology",
                "difficulty": "advanced"
            },
            {
                "question": "What are the risk factors for coronary artery disease?",
                "answer": "Risk factors include age, male gender, family history, smoking, hypertension, diabetes, dyslipidemia, obesity, sedentary lifestyle, and chronic kidney disease.",
                "category": "cardiology",
                "difficulty": "intermediate"
            },
            
            # Respiratory
            {
                "question": "What are the different types of asthma medications and their mechanisms?",
                "answer": "Quick-relief: Beta-2 agonists (bronchodilation). Long-term control: Inhaled corticosteroids (anti-inflammatory), LABAs, leukotriene modifiers, and biologics for severe asthma.",
                "category": "respiratory",
                "difficulty": "intermediate"
            },
            {
                "question": "How is community-acquired pneumonia diagnosed and treated?",
                "answer": "Diagnosed through clinical presentation, chest X-ray, blood tests, and sputum culture. Treated with antibiotics (amoxicillin, azithromycin, or fluoroquinolones) based on severity and risk factors.",
                "category": "respiratory",
                "difficulty": "advanced"
            },
            {
                "question": "What are the indications for mechanical ventilation?",
                "answer": "Indications include respiratory failure (hypoxemia, hypercapnia), airway protection, reduced work of breathing, and perioperative management during surgery.",
                "category": "respiratory",
                "difficulty": "advanced"
            },
            
            # Infectious Diseases
            {
                "question": "What are the stages and symptoms of COVID-19 infection?",
                "answer": "Stages: Incubation (2-14 days), mild symptoms (fever, cough, fatigue), moderate (pneumonia), severe (respiratory failure), critical (multi-organ failure, ARDS).",
                "category": "infectious_disease",
                "difficulty": "intermediate"
            },
            {
                "question": "When should antibiotics be prescribed for bacterial infections?",
                "answer": "Antibiotics should be prescribed for confirmed or highly suspected bacterial infections, not viral infections. Consider culture results, local resistance patterns, and patient factors.",
                "category": "infectious_disease",
                "difficulty": "intermediate"
            },
            
            # Mental Health
            {
                "question": "What are the DSM-5 criteria for major depressive disorder?",
                "answer": "Five or more symptoms for ≥2 weeks: depressed mood, anhedonia, weight changes, sleep disturbances, psychomotor changes, fatigue, guilt/worthlessness, concentration problems, suicidal ideation.",
                "category": "mental_health",
                "difficulty": "advanced"
            },
            {
                "question": "What are the first-line treatments for anxiety disorders?",
                "answer": "First-line treatments include cognitive behavioral therapy (CBT), SSRIs, SNRIs, and lifestyle modifications. Benzodiazepines for short-term use only.",
                "category": "mental_health",
                "difficulty": "intermediate"
            },
            
            # Oncology
            {
                "question": "What are the recommended cancer screening guidelines for average-risk adults?",
                "answer": "Mammography (50+), colonoscopy (45+), cervical cancer screening (21-65), lung cancer CT for high-risk smokers (50-80), skin cancer checks annually.",
                "category": "oncology",
                "difficulty": "intermediate"
            },
            {
                "question": "What are the common side effects of chemotherapy?",
                "answer": "Common side effects include nausea/vomiting, fatigue, hair loss, increased infection risk, bleeding, neuropathy, and organ-specific toxicities depending on the agent.",
                "category": "oncology",
                "difficulty": "intermediate"
            },
            
            # Endocrinology
            {
                "question": "What are the causes and symptoms of hypothyroidism?",
                "answer": "Causes: Hashimoto's thyroiditis, iodine deficiency, medications. Symptoms: fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, depression.",
                "category": "endocrinology",
                "difficulty": "intermediate"
            },
            {
                "question": "How is diabetic ketoacidosis (DKA) diagnosed and managed?",
                "answer": "Diagnosis: glucose >250 mg/dL, ketones, pH <7.3, anion gap >10. Management: IV fluids, insulin infusion, electrolyte replacement, and treating precipitating factors.",
                "category": "endocrinology",
                "difficulty": "advanced"
            },
            
            # Nephrology
            {
                "question": "What are the stages of chronic kidney disease and their management?",
                "answer": "Stage 1-2: GFR >60, manage risk factors. Stage 3: GFR 30-59, monitor complications. Stage 4: GFR 15-29, prepare for RRT. Stage 5: GFR <15, RRT needed.",
                "category": "nephrology",
                "difficulty": "advanced"
            },
            
            # Gastroenterology
            {
                "question": "What are the alarm symptoms that warrant upper endoscopy?",
                "answer": "Alarm symptoms include dysphagia, odynophagia, weight loss, GI bleeding, iron deficiency anemia, persistent vomiting, and family history of gastric cancer.",
                "category": "gastroenterology",
                "difficulty": "intermediate"
            },
            
            # Pharmacology
            {
                "question": "What are the contraindications for ACE inhibitors?",
                "answer": "Contraindications include pregnancy, bilateral renal artery stenosis, hyperkalemia, previous angioedema, and severe aortic stenosis.",
                "category": "pharmacology",
                "difficulty": "advanced"
            },
            {
                "question": "What are the drug interactions with warfarin?",
                "answer": "Interactions include antibiotics (increase INR), NSAIDs (bleeding risk), amiodarone (increase INR), rifampin (decrease INR), and vitamin K-rich foods (decrease INR).",
                "category": "pharmacology",
                "difficulty": "advanced"
            }
        ]
        
        return medical_qa_data
    
    def download_public_datasets(self) -> Dict[str, bool]:
        """Attempt to download public medical datasets"""
        results = {}
        
        for dataset_name, url in self.dataset_sources.items():
            filename = f"{dataset_name}.json"
            success = self.download_file(url, filename)
            results[dataset_name] = success
            
            # Add delay between downloads
            if success:
                time.sleep(1)
        
        return results
    
    def process_downloaded_data(self) -> List[Dict]:
        """Process downloaded data and combine with synthetic data"""
        all_data = []
        
        # Add comprehensive synthetic data
        synthetic_data = self.create_comprehensive_medical_dataset()
        all_data.extend(synthetic_data)
        
        # Process downloaded files if they exist
        for filename in self.data_dir.glob("*.json"):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process different data formats
                if filename.name.startswith("medqa"):
                    processed = self.process_medqa_format(data)
                elif filename.name.startswith("pubmedqa"):
                    processed = self.process_pubmedqa_format(data)
                elif filename.name.startswith("medical_flashcards"):
                    processed = self.process_flashcards_format(data)
                else:
                    continue
                
                all_data.extend(processed)
                self.logger.info(f"Processed {len(processed)} questions from {filename.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
        
        return all_data
    
    def process_medqa_format(self, data) -> List[Dict]:
        """Process MedQA format data"""
        processed = []
        
        # Handle different MedQA formats
        if isinstance(data, list):
            items = data[:50]  # Limit to first 50 for demo
        elif isinstance(data, dict):
            items = list(data.values())[:50]
        else:
            return processed
        
        for item in items:
            try:
                if isinstance(item, dict) and "question" in item:
                    processed.append({
                        "question": item["question"],
                        "answer": item.get("answer", item.get("answerKey", "Unknown")),
                        "category": "medical_exam",
                        "difficulty": "professional",
                        "source": "medqa"
                    })
            except Exception as e:
                continue
        
        return processed
    
    def process_pubmedqa_format(self, data) -> List[Dict]:
        """Process PubMedQA format data"""
        processed = []
        
        try:
            if isinstance(data, dict):
                items = list(data.items())[:30]  # Limit for demo
                
                for pmid, item in items:
                    if isinstance(item, dict) and "QUESTION" in item:
                        processed.append({
                            "question": item["QUESTION"],
                            "answer": item.get("final_decision", "Unknown"),
                            "category": "biomedical_research",
                            "difficulty": "research",
                            "source": "pubmedqa",
                            "context": " ".join(item.get("CONTEXTS", [])[:2])  # First 2 contexts
                        })
        except Exception as e:
            self.logger.error(f"Error processing PubMedQA data: {e}")
        
        return processed
    
    def process_flashcards_format(self, data) -> List[Dict]:
        """Process medical flashcards format"""
        processed = []
        
        try:
            if isinstance(data, list):
                items = data[:40]  # Limit for demo
                
                for item in items:
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        processed.append({
                            "question": item["input"],
                            "answer": item["output"],
                            "category": "medical_education",
                            "difficulty": "educational",
                            "source": "flashcards"
                        })
        except Exception as e:
            self.logger.error(f"Error processing flashcards data: {e}")
        
        return processed
    
    def save_dataset(self, data: List[Dict], filename: str = "medical_qa_dataset.json"):
        """Save processed dataset"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(data)} questions to {filepath}")
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(data)
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Also saved as CSV: {csv_path}")
        
        return filepath
    
    def get_dataset_stats(self, data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        if not data:
            return {}
        
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
        """Main method to download and process all data"""
        self.logger.info("Starting medical QA dataset download and processing...")
        
        # Try to download public datasets
        download_results = self.download_public_datasets()
        self.logger.info(f"Download results: {download_results}")
        
        # Process all data
        all_data = self.process_downloaded_data()
        
        # Save dataset
        dataset_path = self.save_dataset(all_data)
        
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
    
    print("\n" + "="*50)
    print("MEDICAL QA DATASET DOWNLOAD COMPLETE")
    print("="*50)
    print(f"Dataset saved to: {dataset_path}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Categories: {list(stats['categories'].keys())}")
    print(f"Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"Average answer length: {stats['avg_answer_length']:.1f} characters")

if __name__ == "__main__":
    main()
