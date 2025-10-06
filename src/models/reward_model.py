import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
from typing import List

logger = logging.getLogger(__name__)


import ast
import re
from typing import List

class ImprovedCodeRewardModel(nn.Module):
    """Improved reward model trained on human evaluations."""
    
    def __init__(self, model_name="microsoft/codebert-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Multi-head evaluation
        self.consistency_head = nn.Linear(768, 1)
        self.correctness_head = nn.Linear(768, 1)
        self.usefulness_head = nn.Linear(768, 1)
        self.overall_head = nn.Linear(768, 1)
        
        # Code quality indicators (for compatibility)
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    # ДОБАВЛЯЕМ МЕТОДЫ ДЛЯ СОВМЕСТИМОСТИ С СТАРЫМ КОДОМ
    def _check_syntax(self, code: str) -> float:
        """Check Python syntax validity."""
        try:
            ast.parse(code)
            lines = code.strip().split('\n')
            if len(lines) >= 2:
                return 0.9
            else:
                return 0.7
        except SyntaxError:
            return 0.0
    
    def _check_structure(self, code: str) -> float:
        """Check code structure and best practices."""
        score = 0.0
        lines = [line for line in code.split('\n') if line.strip()]
        
        if any('import ' in line for line in lines):
            score += 0.3
        
        if any('def ' in line for line in lines):
            score += 0.4
        
        good_count = sum(1 for practice in self.good_practices if practice in code)
        score += min(good_count * 0.05, 0.2)
        
        if len(lines) >= 2 and len(lines) <= 15:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_execution(self, prompt: str, code: str) -> float:
        """Basic execution check."""
        try:
            compile(code, '<string>', 'exec')
            return 0.5
        except:
            return 0.0
    
    def _check_relevance(self, prompt: str, code: str) -> float:
        """Check relevance to prompt."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        keyword_mappings = {
            'signal': ['signal', 'kill', 'pid'],
            'decode': ['decode', 'hex', 'utf'],
            'dictionary': ['dict', 'kwargs', 'items'],
            'subprocess': ['subprocess', 'call', 'check_output'],
            'pandas': ['pandas', 'series', 'dataframe'],
            'http': ['http', 'header', 'client'],
            'datetime': ['datetime', 'strptime', 'date'],
            'split': ['split', 'string', 'lines'],
            'concatenate': ['join', 'concatenate'],
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    return 0.7
        
        return 0.3
    
    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete."""
        score = 0.0
        
        if code.strip().endswith((')', ']', '}', '"', "'")):
            score += 0.3
        
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.3
        
        if any(mod in code for mod in ['subprocess', 'datetime', 'pandas']):
            if 'import' in code:
                score += 0.2
        else:
            score += 0.2
        
        return score
    
    def _check_bad_practices(self, code: str) -> float:
        """Check for bad coding practices."""
        penalty = 0.0
        danger_count = sum(1 for practice in self.bad_practices if practice in code)
        penalty += danger_count * 0.1
        
        try:
            ast.parse(code)
        except:
            penalty += 0.2
        
        return min(penalty, 0.3)
        
    def forward(self, questions: List[str], answers: List[str]):
        # Encode questions and answers
        inputs = self.tokenizer(
            questions, answers,
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.bert(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get scores for different aspects
        consistency = torch.sigmoid(self.consistency_head(pooled_output))
        correctness = torch.sigmoid(self.correctness_head(pooled_output))
        usefulness = torch.sigmoid(self.usefulness_head(pooled_output))
        overall = torch.sigmoid(self.overall_head(pooled_output))
        
        return {
            'consistency': consistency,
            'correctness': correctness, 
            'usefulness': usefulness,
            'overall': overall
        }
    
    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute reward for PPO training."""
        with torch.no_grad():
            results = self.forward(prompts, responses)
            # Use overall score as reward
            return results['overall'].squeeze()
    
    def predict_quality(self, prompt: str, response: str) -> dict:
        """Predict quality scores for a single prompt-response pair."""
        with torch.no_grad():
            results = self.forward([prompt], [response])
            return {
                'consistency': results['consistency'].item(),
                'correctness': results['correctness'].item(),
                'usefulness': results['usefulness'].item(),
                'overall': results['overall'].item()
            }