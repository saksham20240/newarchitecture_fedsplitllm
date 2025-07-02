"""
Utilities Package for Federated Medical QA

This package contains utility functions for metrics, tokenization, and other
helper functionality for the federated learning system.
"""

from .metrics import (
    MedicalQAMetrics,
    TrainingMonitor
)

from .tokenizer import SimpleTokenizer

__version__ = "1.0.0"
__all__ = [
    # Metrics and evaluation
    "MedicalQAMetrics",
    "TrainingMonitor",
    
    # Tokenization
    "SimpleTokenizer"
]

def create_medical_tokenizer(vocab_size=32000):
    """Create a tokenizer optimized for medical text"""
    return SimpleTokenizer(vocab_size=vocab_size)

def create_metrics_tracker():
    """Create a metrics tracker for medical QA evaluation"""
    return MedicalQAMetrics()

def create_training_monitor(patience=5, min_delta=0.001):
    """Create a training monitor for early stopping"""
    return TrainingMonitor(patience=patience, min_delta=min_delta)

def evaluate_medical_qa(questions, predictions, references, metrics=None):
    """
    Evaluate medical QA performance
    
    Args:
        questions: List of question strings
        predictions: List of predicted answer strings  
        references: List of reference answer strings
        metrics: MedicalQAMetrics instance (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    if metrics is None:
        metrics = MedicalQAMetrics()
    
    # Update metrics with predictions
    metrics.update(
        loss=0.0,  # No loss for evaluation-only
        questions=questions,
        answers=references,
        predictions=predictions
    )
    
    return metrics.get_current_metrics()

__all__.extend([
    "create_medical_tokenizer",
    "create_metrics_tracker", 
    "create_training_monitor",
    "evaluate_medical_qa"
])

# Package-level utility functions
def check_package_health():
    """Check health of all utility components"""
    health = {
        "metrics_available": False,
        "tokenizer_available": False,
        "dependencies_ok": False
    }
    
    try:
        # Test metrics
        metrics = MedicalQAMetrics()
        health["metrics_available"] = True
    except Exception as e:
        health["metrics_error"] = str(e)
    
    try:
        # Test tokenizer
        tokenizer = SimpleTokenizer(vocab_size=1000)
        test_result = tokenizer(["test"], return_tensors="pt")
        health["tokenizer_available"] = True
    except Exception as e:
        health["tokenizer_error"] = str(e)
    
    # Check dependencies
    try:
        import torch, numpy
        health["dependencies_ok"] = True
    except ImportError as e:
        health["dependency_error"] = str(e)
    
    return health

__all__.append("check_package_health")
