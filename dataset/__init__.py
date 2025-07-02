"""
Dataset Package for Medical QA

This package handles downloading, processing, and managing medical question-answering
datasets without requiring external dependencies like Hugging Face.
"""

from .medical_qa_downloader import MedicalQADownloader

__version__ = "1.0.0"
__all__ = ["MedicalQADownloader"]

def download_medical_dataset(data_dir="./data", force_download=False):
    """
    Convenience function to download medical QA dataset
    
    Args:
        data_dir: Directory to store dataset
        force_download: Force re-download even if dataset exists
    
    Returns:
        Tuple of (dataset_path, statistics)
    """
    import os
    from pathlib import Path
    
    dataset_path = Path(data_dir) / "medical_qa_dataset.json"
    
    if not dataset_path.exists() or force_download:
        downloader = MedicalQADownloader(data_dir)
        return downloader.download_and_process()
    else:
        # Load existing dataset stats
        import json
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        stats = {
            'total_questions': len(data),
            'dataset_path': str(dataset_path)
        }
        
        return str(dataset_path), stats

def get_dataset_info(data_dir="./data"):
    """Get information about the current dataset"""
    import json
    from pathlib import Path
    
    dataset_path = Path(data_dir) / "medical_qa_dataset.json"
    
    if not dataset_path.exists():
        return {"status": "not_found", "path": str(dataset_path)}
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Analyze dataset
        categories = {}
        difficulties = {}
        sources = {}
        
        for item in data:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            diff = item.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            src = item.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        return {
            "status": "found",
            "path": str(dataset_path),
            "total_questions": len(data),
            "categories": categories,
            "difficulties": difficulties,
            "sources": sources
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

__all__.extend(["download_medical_dataset", "get_dataset_info"])
