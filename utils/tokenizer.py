# utils/tokenizer.py
"""
Simple tokenizer for medical QA without external dependencies
"""

import torch
import re
import json
from pathlib import Path
from typing import List, Dict, Union, Optional

class SimpleTokenizer:
    """
    Simple tokenizer for medical text without external dependencies
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,      # Start of sequence
            '</s>': 3,     # End of sequence
            '<mask>': 4,   # Masked token
        }
        
        # Build vocabulary
        self.build_vocabulary()
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Token properties
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.mask_token = '<mask>'
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.mask_token_id = self.special_tokens['<mask>']
    
    def build_vocabulary(self):
        """Build vocabulary with medical terms and common words"""
        # Start with special tokens
        self.token_to_id = self.special_tokens.copy()
        
        # Medical terminology
        medical_terms = [
            # Basic medical terms
            'patient', 'doctor', 'nurse', 'hospital', 'clinic', 'medicine', 'treatment',
            'diagnosis', 'symptom', 'disease', 'condition', 'therapy', 'medication',
            'prescription', 'dose', 'dosage', 'side', 'effect', 'effects', 'reaction',
            
            # Body systems and anatomy
            'heart', 'lung', 'brain', 'liver', 'kidney', 'stomach', 'blood', 'bone',
            'muscle', 'nerve', 'skin', 'eye', 'ear', 'nose', 'throat', 'chest',
            'abdomen', 'back', 'head', 'neck', 'arm', 'leg', 'hand', 'foot',
            
            # Common conditions
            'diabetes', 'hypertension', 'cancer', 'infection', 'inflammation', 'pain',
            'fever', 'cough', 'headache', 'nausea', 'vomiting', 'diarrhea', 'fatigue',
            'weakness', 'dizziness', 'shortness', 'breath', 'breathing', 'swelling',
            
            # Cardiovascular
            'cardiac', 'cardiology', 'heart', 'artery', 'vein', 'circulation', 'pulse',
            'blood', 'pressure', 'hypertension', 'hypotension', 'myocardial', 'infarction',
            'stroke', 'clot', 'thrombosis', 'embolism', 'angina', 'arrhythmia',
            
            # Respiratory
            'respiratory', 'pulmonary', 'asthma', 'copd', 'pneumonia', 'bronchitis',
            'emphysema', 'tuberculosis', 'oxygen', 'ventilation', 'intubation',
            
            # Endocrine
            'endocrine', 'hormone', 'insulin', 'glucose', 'thyroid', 'adrenal',
            'pancreas', 'metabolic', 'diabetes', 'hypoglycemia', 'hyperglycemia',
            
            # Gastrointestinal
            'gastro', 'intestinal', 'stomach', 'liver', 'gallbladder', 'pancreas',
            'bowel', 'colon', 'rectum', 'ulcer', 'gastritis', 'hepatitis', 'cirrhosis',
            
            # Neurological
            'neurological', 'neurology', 'brain', 'spinal', 'cord', 'seizure', 'epilepsy',
            'migraine', 'dementia', 'alzheimer', 'parkinson', 'multiple', 'sclerosis',
            
            # Oncology
            'oncology', 'cancer', 'tumor', 'malignant', 'benign', 'metastasis',
            'chemotherapy', 'radiation', 'biopsy', 'remission', 'recurrence',
            
            # Procedures and tests
            'surgery', 'operation', 'procedure', 'examination', 'test', 'scan',
            'x-ray', 'ct', 'mri', 'ultrasound', 'biopsy', 'blood', 'urine', 'stool',
            'ecg', 'ekg', 'eeg', 'colonoscopy', 'endoscopy', 'mammography',
            
            # Medications
            'antibiotic', 'antiviral', 'antifungal', 'painkiller', 'analgesic',
            'anti-inflammatory', 'steroid', 'aspirin', 'ibuprofen', 'acetaminophen',
            'insulin', 'metformin', 'statin', 'beta', 'blocker', 'ace', 'inhibitor',
            
            # Units and measurements
            'mg', 'ml', 'kg', 'lb', 'celsius', 'fahrenheit', 'mmhg', 'bpm',
            'percent', 'level', 'count', 'rate', 'normal', 'abnormal', 'high', 'low',
            
            # Time and frequency
            'daily', 'weekly', 'monthly', 'morning', 'evening', 'bedtime', 'meals',
            'before', 'after', 'during', 'acute', 'chronic', 'sudden', 'gradual',
        ]
        
        # Common English words
        common_words = [
            # Articles and pronouns
            'a', 'an', 'the', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself',
            
            # Verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'get', 'got', 'give', 'take', 'make', 'go',
            'come', 'see', 'know', 'think', 'feel', 'want', 'need', 'use', 'help',
            
            # Prepositions and conjunctions
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'and', 'or', 'but', 'so', 'if', 'when', 'where',
            'how', 'why', 'what', 'who', 'which', 'that', 'because', 'since',
            
            # Adjectives and adverbs
            'good', 'bad', 'great', 'small', 'large', 'long', 'short', 'high', 'low',
            'hot', 'cold', 'warm', 'cool', 'fast', 'slow', 'easy', 'hard', 'light',
            'dark', 'clean', 'dirty', 'old', 'new', 'young', 'early', 'late',
            'first', 'last', 'next', 'previous', 'important', 'serious', 'mild',
            'severe', 'common', 'rare', 'typical', 'unusual', 'normal', 'abnormal',
            
            # Numbers
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'twenty', 'thirty', 'forty', 'fifty',
            'hundred', 'thousand', 'million', 'first', 'second', 'third',
            
            # Question words
            'question', 'answer', 'what', 'when', 'where', 'who', 'why', 'how',
            'which', 'does', 'do', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must'
        ]
        
        # Punctuation and special characters
        punctuation = [
            '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}',
            '"', "'", '-', '_', '/', '\\', '&', '#', '@', '$', '%', '*', '+', '='
        ]
        
        # Combine all vocabulary
        all_tokens = list(medical_terms) + list(common_words) + list(punctuation)
        
        # Add tokens to vocabulary
        current_id = len(self.special_tokens)
        for token in all_tokens:
            if token not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[token] = current_id
                current_id += 1
        
        # Fill remaining slots with common subwords/characters
        if current_id < self.vocab_size:
            # Add common letter combinations and single characters
            additional_tokens = []
            
            # Single characters
            for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
                additional_tokens.append(c)
            
            # Common prefixes and suffixes
            affixes = [
                'un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'up',
                'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment',
                'able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ic', 'ical'
            ]
            additional_tokens.extend(affixes)
            
            # Add to vocabulary
            for token in additional_tokens:
                if token not in self.token_to_id and current_id < self.vocab_size:
                    self.token_to_id[token] = current_id
                    current_id += 1
                if current_id >= self.vocab_size:
                    break
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        
        # Split on whitespace and punctuation
        # Keep punctuation as separate tokens
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = self.convert_ids_to_tokens(ids)
        
        if skip_special_tokens:
            # Remove special tokens
            special_token_names = set(self.special_tokens.keys())
            tokens = [token for token in tokens if token not in special_token_names]
        
        # Join tokens with spaces
        text = ' '.join(tokens)
        
        # Clean up spacing around punctuation
        text = re.sub(r' ([.!?,:;])', r'\1', text)
        text = re.sub(r'(\w) ([\'"])', r'\1\2', text)
        
        return text
    
    def __call__(self, texts: Union[str, List[str]], 
                 max_length: int = 512,
                 padding: bool = True,
                 truncation: bool = True,
                 return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts with padding and truncation
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode all texts
        all_ids = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=True)
            
            # Truncate if necessary
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.eos_token_id]
            
            all_ids.append(ids)
        
        # Pad sequences if necessary
        if padding:
            max_len = min(max_length, max(len(ids) for ids in all_ids))
            padded_ids = []
            attention_masks = []
            
            for ids in all_ids:
                # Pad sequence
                if len(ids) < max_len:
                    padding_length = max_len - len(ids)
                    ids = ids + [self.pad_token_id] * padding_length
                
                # Create attention mask
                attention_mask = [1 if id != self.pad_token_id else 0 for id in ids]
                
                padded_ids.append(ids)
                attention_masks.append(attention_mask)
            
            all_ids = padded_ids
        else:
            # Create attention masks without padding
            attention_masks = [[1] * len(ids) for ids in all_ids]
        
        # Convert to tensors if requested
        result = {
            'input_ids': all_ids,
            'attention_mask': attention_masks
        }
        
        if return_tensors == "pt":
            result['input_ids'] = torch.tensor(all_ids, dtype=torch.long)
            result['attention_mask'] = torch.tensor(attention_masks, dtype=torch.long)
        
        return result
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)
    
    def save_vocabulary(self, vocab_path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocabulary(self, vocab_path: str):
        """Load vocabulary from file"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.special_tokens = vocab_data['special_tokens']
        self.vocab_size = vocab_data['vocab_size']
        
        # Recreate reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

def test_tokenizer():
    """Test the tokenizer implementation"""
    print("🧪 Testing Simple Tokenizer...")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Test medical text
    test_texts = [
        "What are the symptoms of diabetes?",
        "The patient has high blood pressure and needs medication.",
        "Common side effects include nausea, headache, and dizziness."
    ]
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.special_tokens}")
    
    # Test encoding and decoding
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Encode
        ids = tokenizer.encode(text)
        print(f"IDs: {ids}")
        
        # Decode
        decoded = tokenizer.decode(ids)
        print(f"Decoded: {decoded}")
    
    # Test batch processing
    print("\n" + "="*50)
    print("Testing batch processing...")
    
    batch_result = tokenizer(
        test_texts,
        max_length=20,
        padding=True,
        return_tensors="pt"
    )
    
    print(f"Input IDs shape: {batch_result['input_ids'].shape}")
    print(f"Attention mask shape: {batch_result['attention_mask'].shape}")
    print(f"Sample input IDs: {batch_result['input_ids'][0]}")
    print(f"Sample attention mask: {batch_result['attention_mask'][0]}")
    
    print("✅ Tokenizer test completed!")

if __name__ == "__main__":
    test_tokenizer()
