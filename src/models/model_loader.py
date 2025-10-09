import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import torch
from typing import List

logger = logging.getLogger(__name__)

class ModelLoader:
    """Improved model loader with better error handling."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_models(self):
        """Load tokenizer and policy model."""
        logger.info(f"Loading models with device: {self.device}")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer()
        
        # Load policy model
        policy_model = self._load_policy_model()
        
        # Reference model (can be None for PPOTrainer)
        ref_model = None
        
        logger.info("Models loaded successfully!!")
        return tokenizer, policy_model, ref_model
    
    def _load_tokenizer(self):
        """Load and configure tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"First attempt failed: {e}, trying fallback...")
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info("Using GPT-2 tokenizer as fallback")
            except Exception as e2:
                logger.error(f"Failed to load tokenizer: {e2}")
                raise
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    def _load_policy_model(self):
        """Load policy model with value head."""
        try:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            model = model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load policy model: {e}")
            raise

class CodeRewardModel:
    """Enhanced reward model for code generation with better metrics."""
    
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimized metric weights for code quality
        self.metric_weights = {
            'syntax': 0.4,
            'structure': 0.3,
            'relevance': 0.2,
            'completeness': 0.1
        }
        
        # Enhanced code quality indicators
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]

    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute enhanced rewards for code generation."""
        rewards = []
        
        for prompt, code in zip(prompts, responses):
            if not code or len(code.strip()) < self.config.min_code_length:
                rewards.append(-1.0)
                continue
            
            reward = 0.0
            
            # 1. Syntax validity (most important)
            syntax_score = self._check_syntax(code)
            reward += syntax_score * self.metric_weights['syntax']
            
            # 2. Code structure and best practices
            structure_score = self._check_structure(code)
            reward += structure_score * self.metric_weights['structure']
            
            # 3. Relevance to prompt
            relevance_score = self._check_relevance(prompt, code)
            reward += relevance_score * self.metric_weights['relevance']
            
            # 4. Code completeness
            completeness_score = self._check_completeness(code)
            reward += completeness_score * self.metric_weights['completeness']
            
            # Penalties for bad practices
            penalties = self._check_bad_practices(code)
            reward -= penalties
            
            # Normalize and ensure reasonable range
            reward = max(min(reward, 1.0), -0.5)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _check_syntax(self, code: str) -> float:
        """Enhanced syntax checking."""
        try:
            ast.parse(code)
            
            lines = code.strip().split('\n')
            if len(lines) >= 2:
                return 0.9
            else:
                return 0.7
                
        except SyntaxError as e:
            error_msg = str(e)
            if 'unexpected EOF' in error_msg or 'parenthesis' in error_msg:
                return 0.3
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

    def _check_relevance(self, prompt: str, code: str) -> float:
        """Enhanced relevance checking."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        relevance = 0.0
        
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
            'django': ['django', 'model', 'queryset'],
            'numpy': ['numpy', 'array', 'sum'],
            'file': ['file', 'open', 'write']
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    relevance += 0.3
                    break
        
        prompt_words = set(prompt_lower.split())
        code_words = set(code_lower.split())
        if prompt_words and code_words:
            overlap = len(prompt_words.intersection(code_words))
            relevance += min(overlap / len(prompt_words) * 0.3, 0.3)
        
        return min(relevance, 1.0)

    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete and executable."""
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

    def _check_execution(self, prompt: str, code: str) -> float:
        """Basic execution check - simplified version."""
        try:
            # Safe compilation check
            compile(code, '<string>', 'exec')
            return 0.5
        except:
            return 0.0

class ImprovedCodeRewardModel:
    """Enhanced reward model for code generation with better metrics."""
    
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimized metric weights for code quality
        self.metric_weights = {
            'syntax': 0.4,        # Increased importance of syntax
            'structure': 0.3,     # Code structure and practices
            'relevance': 0.2,     # Relevance to prompt
            'completeness': 0.1   # Code completeness
        }
        
        # Enhanced code quality indicators
        self.good_practices = [
            'def ', 'return ', 'import ', 'from ', 'class ', 'try:', 'except ',
            'if __name__', 'with open', 'isinstance', 'len(', 'range(', 'subprocess.',
            'datetime.', 'pandas.', 'numpy.', 'os.', 'sys.'
        ]
        
        self.bad_practices = [
            'eval(', 'exec(', 'input()', 'while True:', 'import *',
            'except:', 'except Exception:', 'print(', 'exit()'
        ]

    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute enhanced rewards for code generation."""
        rewards = []
        
        for prompt, code in zip(prompts, responses):
            if not code or len(code.strip()) < self.config.min_code_length:
                rewards.append(-1.0)
                continue
            
            reward = 0.0
            
            # 1. Syntax validity (most important)
            syntax_score = self._check_syntax(code)
            reward += syntax_score * self.metric_weights['syntax']
            
            # 2. Code structure and best practices
            structure_score = self._check_structure(code)
            reward += structure_score * self.metric_weights['structure']
            
            # 3. Relevance to prompt
            relevance_score = self._check_relevance(prompt, code)
            reward += relevance_score * self.metric_weights['relevance']
            
            # 4. Code completeness
            completeness_score = self._check_completeness(code)
            reward += completeness_score * self.metric_weights['completeness']
            
            # Penalties for bad practices
            penalties = self._check_bad_practices(code)
            reward -= penalties
            
            # Normalize and ensure reasonable range
            reward = max(min(reward, 1.0), -0.5)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _check_syntax(self, code: str) -> float:
        """Enhanced syntax checking."""
        try:
            # Try to parse the code
            ast.parse(code)
            
            # Additional quality checks
            lines = code.strip().split('\n')
            if len(lines) >= 2:  # Multi-line code gets higher score
                return 0.9
            else:
                return 0.7
                
        except SyntaxError as e:
            error_msg = str(e)
            # Partial credit for common syntax errors that are close to correct
            if 'unexpected EOF' in error_msg or 'parenthesis' in error_msg:
                return 0.3
            return 0.0

    def _check_structure(self, code: str) -> float:
        """Check code structure and best practices."""
        score = 0.0
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Check for imports and function definitions
        if any('import ' in line for line in lines):
            score += 0.3
        
        if any('def ' in line for line in lines):
            score += 0.4
        
        # Check for good practices
        good_count = sum(1 for practice in self.good_practices if practice in code)
        score += min(good_count * 0.05, 0.2)  # Reduced weight per practice
        
        # Check code organization
        if len(lines) >= 2 and len(lines) <= 15:  # Reasonable length
            score += 0.1
        
        return min(score, 1.0)

    def _check_relevance(self, prompt: str, code: str) -> float:
        """Enhanced relevance checking."""
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        relevance = 0.0
        
        # Keyword matching with context
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
            'django': ['django', 'model', 'queryset'],
            'numpy': ['numpy', 'array', 'sum'],
            'file': ['file', 'open', 'write']
        }
        
        for prompt_key, code_keys in keyword_mappings.items():
            if prompt_key in prompt_lower:
                if any(key in code_lower for key in code_keys):
                    relevance += 0.3
                    break
        
        # Basic word overlap
        prompt_words = set(prompt_lower.split())
        code_words = set(code_lower.split())
        if prompt_words and code_words:
            overlap = len(prompt_words.intersection(code_words))
            relevance += min(overlap / len(prompt_words) * 0.3, 0.3)
        
        return min(relevance, 1.0)

    def _check_completeness(self, code: str) -> float:
        """Check if code appears complete and executable."""
        score = 0.0
        
        # Check for proper endings
        if code.strip().endswith((')', ']', '}', '"', "'")):
            score += 0.3
        
        # Check for balanced brackets
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.3
        
        # Check for imports if needed
        if any(mod in code for mod in ['subprocess', 'datetime', 'pandas']):
            if 'import' in code:
                score += 0.2
        else:
            score += 0.2  # Bonus for not needing imports
        
        return score

    def _check_bad_practices(self, code: str) -> float:
        """Check for bad coding practices."""
        penalty = 0.0
        
        # Check for dangerous functions
        danger_count = sum(1 for practice in self.bad_practices if practice in code)
        penalty += danger_count * 0.1  # Reduced penalty
        
        # Check for syntax errors in the structure
        try:
            ast.parse(code)
        except:
            penalty += 0.2
        
        return min(penalty, 0.3)