# utils/metrics.py
"""
Comprehensive metrics and evaluation utilities for medical QA
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

class MedicalQAMetrics:
    """
    Comprehensive metrics for medical QA evaluation
    """
    
    def __init__(self):
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Initialize BLEU smoothing if available
        if BLEU_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method1
        
        # Metrics storage
        self.reset_metrics()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'loss': [],
            'perplexity': [],
            'bleu_scores': [],
            'rouge1_f1': [],
            'rouge2_f1': [],
            'rougeL_f1': [],
            'answer_length': [],
            'question_length': [],
            'exact_match': [],
            'semantic_similarity': [],
            'medical_term_coverage': []
        }
        
        self.step_count = 0
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from cross-entropy loss"""
        try:
            return float(torch.exp(torch.tensor(loss)).item())
        except:
            return float('inf')
    
    def calculate_bleu(self, predicted_text: str, reference_text: str) -> float:
        """Calculate BLEU score"""
        if not BLEU_AVAILABLE:
            return 0.0
        
        try:
            # Tokenize texts
            reference_tokens = reference_text.lower().split()
            predicted_tokens = predicted_text.lower().split()
            
            if not predicted_tokens:
                return 0.0
            
            # Calculate BLEU score
            score = sentence_bleu(
                [reference_tokens], 
                predicted_tokens,
                smoothing_function=self.smoothing_function
            )
            return float(score)
        except Exception as e:
            self.logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_rouge(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference_text, predicted_text)
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure
            }
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
    
    def calculate_exact_match(self, predicted_text: str, reference_text: str) -> float:
        """Calculate exact match score"""
        # Normalize texts for comparison
        pred_normalized = self.normalize_text(predicted_text)
        ref_normalized = self.normalize_text(reference_text)
        
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def calculate_medical_term_coverage(self, predicted_text: str, reference_text: str) -> float:
        """Calculate coverage of medical terms"""
        # Simple medical terms list (can be expanded)
        medical_terms = {
            'diabetes', 'hypertension', 'cardiovascular', 'myocardial', 'infarction',
            'pneumonia', 'bronchitis', 'asthma', 'copd', 'emphysema',
            'oncology', 'cancer', 'tumor', 'chemotherapy', 'radiation',
            'nephrology', 'kidney', 'dialysis', 'renal', 'creatinine',
            'cardiology', 'heart', 'arrhythmia', 'tachycardia', 'bradycardia',
            'neurology', 'stroke', 'seizure', 'epilepsy', 'dementia',
            'endocrinology', 'thyroid', 'insulin', 'glucose', 'hormone',
            'gastroenterology', 'liver', 'hepatitis', 'cirrhosis', 'ulcer',
            'dermatology', 'skin', 'rash', 'eczema', 'psoriasis',
            'orthopedic', 'bone', 'fracture', 'arthritis', 'joint',
            'psychiatric', 'depression', 'anxiety', 'bipolar', 'schizophrenia'
        }
        
        # Extract medical terms from reference
        ref_words = set(self.normalize_text(reference_text).split())
        ref_medical = ref_words.intersection(medical_terms)
        
        if not ref_medical:
            return 1.0  # No medical terms to cover
        
        # Extract medical terms from prediction
        pred_words = set(self.normalize_text(predicted_text).split())
        pred_medical = pred_words.intersection(medical_terms)
        
        # Calculate coverage
        coverage = len(pred_medical.intersection(ref_medical)) / len(ref_medical)
        return coverage
    
    def calculate_semantic_similarity(self, predicted_text: str, reference_text: str) -> float:
        """Calculate semantic similarity (simplified)"""
        # Simple word overlap-based similarity
        pred_words = set(self.normalize_text(predicted_text).split())
        ref_words = set(self.normalize_text(reference_text).split())
        
        if not ref_words:
            return 0.0
        
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        
        if not union:
            return 0.0
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        
        # Weighted by reference length (favor recall)
        recall = len(intersection) / len(ref_words) if ref_words else 0.0
        
        return 0.7 * jaccard + 0.3 * recall
    
    def update(self, loss: float, questions: List[str], answers: List[str], 
               predictions: Optional[List[str]] = None):
        """Update metrics with batch data"""
        self.step_count += 1
        
        # Basic metrics
        self.metrics['loss'].append(loss)
        self.metrics['perplexity'].append(self.calculate_perplexity(loss))
        
        # Text quality metrics (if predictions available)
        if predictions is not None:
            batch_metrics = defaultdict(list)
            
            for pred, ref_q, ref_a in zip(predictions, questions, answers):
                # Use answer as reference for evaluation
                reference = ref_a
                
                # Calculate all metrics
                bleu = self.calculate_bleu(pred, reference)
                rouge_scores = self.calculate_rouge(pred, reference)
                exact_match = self.calculate_exact_match(pred, reference)
                semantic_sim = self.calculate_semantic_similarity(pred, reference)
                medical_coverage = self.calculate_medical_term_coverage(pred, reference)
                
                # Store metrics
                batch_metrics['bleu_scores'].append(bleu)
                batch_metrics['rouge1_f1'].append(rouge_scores['rouge1_f1'])
                batch_metrics['rouge2_f1'].append(rouge_scores['rouge2_f1'])
                batch_metrics['rougeL_f1'].append(rouge_scores['rougeL_f1'])
                batch_metrics['exact_match'].append(exact_match)
                batch_metrics['semantic_similarity'].append(semantic_sim)
                batch_metrics['medical_term_coverage'].append(medical_coverage)
                
                # Length metrics
                batch_metrics['answer_length'].append(len(pred.split()))
                batch_metrics['question_length'].append(len(ref_q.split()))
            
            # Update metrics
            for metric_name, values in batch_metrics.items():
                self.metrics[metric_name].extend(values)
        else:
            # Just update length metrics
            for q, a in zip(questions, answers):
                self.metrics['question_length'].append(len(q.split()))
                self.metrics['answer_length'].append(len(a.split()))
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values (latest or recent average)"""
        current = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                if len(values) >= 5:
                    # Average of last 5 values for stability
                    current[metric_name] = np.mean(values[-5:])
                else:
                    # Latest value
                    current[metric_name] = values[-1]
            else:
                current[metric_name] = 0.0
        
        return current
    
    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary statistics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values),
                    'latest': float(values[-1]) if values else 0.0
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'count': 0, 'latest': 0.0
                }
        
        return summary
    
    def print_current_metrics(self, step: Optional[int] = None):
        """Print current metrics in a formatted way"""
        if step is None:
            step = self.step_count
        
        current = self.get_current_metrics()
        
        print(f"\n📊 Metrics at Step {step}")
        print("-" * 40)
        
        # Core metrics
        if current['loss'] > 0:
            print(f"Loss:       {current['loss']:.4f}")
        if current['perplexity'] > 0:
            print(f"Perplexity: {current['perplexity']:.2f}")
        
        # Quality metrics
        if current['bleu_scores'] > 0:
            print(f"BLEU:       {current['bleu_scores']:.3f}")
        if current['rouge1_f1'] > 0:
            print(f"ROUGE-1:    {current['rouge1_f1']:.3f}")
        if current['exact_match'] > 0:
            print(f"Exact Match: {current['exact_match']:.3f}")
        if current['semantic_similarity'] > 0:
            print(f"Semantic Sim: {current['semantic_similarity']:.3f}")
        if current['medical_term_coverage'] > 0:
            print(f"Medical Terms: {current['medical_term_coverage']:.3f}")
        
        # Length metrics
        if current['answer_length'] > 0:
            print(f"Avg Answer Length: {current['answer_length']:.1f} words")
        
        print("-" * 40)
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        summary = self.get_summary_metrics()
        
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE METRICS REPORT")
        print("=" * 60)
        
        # Performance metrics
        print("\n🎯 PERFORMANCE METRICS:")
        for metric in ['loss', 'perplexity', 'bleu_scores', 'rouge1_f1', 'exact_match']:
            if summary[metric]['count'] > 0:
                stats = summary[metric]
                print(f"  {metric.upper():<15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        # Quality metrics
        print("\n🏥 MEDICAL QA QUALITY:")
        for metric in ['semantic_similarity', 'medical_term_coverage']:
            if summary[metric]['count'] > 0:
                stats = summary[metric]
                print(f"  {metric.replace('_', ' ').title():<20}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Length metrics
        print("\n📏 LENGTH STATISTICS:")
        for metric in ['question_length', 'answer_length']:
            if summary[metric]['count'] > 0:
                stats = summary[metric]
                print(f"  {metric.replace('_', ' ').title():<15}: {stats['mean']:.1f} ± {stats['std']:.1f} words")
        
        # Training statistics
        print(f"\n📊 TRAINING STATISTICS:")
        print(f"  Total Steps: {self.step_count}")
        print(f"  Total Samples: {summary['loss']['count']}")
        
        print("=" * 60)
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        import json
        from pathlib import Path
        
        # Prepare data for saving
        data = {
            'summary': self.get_summary_metrics(),
            'raw_metrics': {k: v for k, v in self.metrics.items()},
            'step_count': self.step_count,
            'config': {
                'rouge_available': ROUGE_AVAILABLE,
                'bleu_available': BLEU_AVAILABLE
            }
        }
        
        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['raw_metrics']
        self.step_count = data['step_count']
        
        self.logger.info(f"Metrics loaded from {filepath}")

class TrainingMonitor:
    """
    Monitor training progress and handle early stopping
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
        
        self.history = []
        
    def __call__(self, score: float, step: int) -> bool:
        """Check if training should stop"""
        self.history.append((step, score))
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check if should stop
        self.should_stop = self.patience_counter >= self.patience
        
        return self.should_stop
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved"""
        return self.best_score
    
    def reset(self):
        """Reset the monitor"""
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
        self.history = []

def test_metrics():
    """Test the metrics implementation"""
    print("🧪 Testing Medical QA Metrics...")
    
    # Initialize metrics
    metrics = MedicalQAMetrics()
    
    # Test data
    questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?"
    ]
    
    answers = [
        "Symptoms include increased thirst, frequent urination, and fatigue.",
        "Treatment includes lifestyle changes and medications like ACE inhibitors."
    ]
    
    predictions = [
        "Common symptoms are thirst, urination, and tiredness.",
        "Hypertension is treated with diet, exercise, and blood pressure medications."
    ]
    
    # Update metrics
    metrics.update(loss=2.5, questions=questions, answers=answers, predictions=predictions)
    metrics.update(loss=2.3, questions=questions, answers=answers, predictions=predictions)
    metrics.update(loss=2.1, questions=questions, answers=answers, predictions=predictions)
    
    # Print results
    metrics.print_current_metrics()
    metrics.print_summary_report()
    
    print("✅ Metrics test completed!")

if __name__ == "__main__":
    test_metrics()
