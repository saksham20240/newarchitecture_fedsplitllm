"""
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
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        
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
