"""
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
