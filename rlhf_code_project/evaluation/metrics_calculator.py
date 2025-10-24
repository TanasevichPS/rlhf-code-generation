"""
Metrics Calculator for Code Generation
=====================================

Comprehensive evaluation metrics for code generation tasks.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculator for all evaluation metrics."""
    
    def __init__(self):
        self.available_metrics = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize available metrics."""
        # Try to import and initialize metrics
        try:
            import evaluate
            self.bertscore = evaluate.load("bertscore")
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            self.available_metrics.update({
                'bertscore': True,
                'bleu': True,
                'rouge': True
            })
            logger.info("Loaded evaluate metrics: bertscore, bleu, rouge")
        except Exception as e:
            logger.warning(f"Failed to load evaluate metrics: {e}")
            self.available_metrics.update({
                'bertscore': False,
                'bleu': False,
                'rouge': False
            })
        
        # Try to import CodeBLEU
        try:
            from codebleu import calc_codebleu
            self.calc_codebleu = calc_codebleu
            self.available_metrics['codebleu'] = True
            logger.info("Loaded CodeBLEU")
        except Exception as e:
            logger.warning(f"Failed to load CodeBLEU: {e}")
            self.available_metrics['codebleu'] = False
    
    def calculate_all_metrics(self, generated_codes: List[str], reference_codes: List[str]) -> Dict[str, float]:
        """Calculate all available metrics."""
        results = {}
        
        # BERTScore
        if self.available_metrics.get('bertscore', False):
            results['bertscore'] = self._calculate_bertscore(generated_codes, reference_codes)
        
        # BLEU
        if self.available_metrics.get('bleu', False):
            results['bleu'] = self._calculate_bleu(generated_codes, reference_codes)
        
        # ROUGE
        if self.available_metrics.get('rouge', False):
            results['rouge'] = self._calculate_rouge(generated_codes, reference_codes)
        
        # CodeBLEU
        if self.available_metrics.get('codebleu', False):
            results['codebleu'] = self._calculate_codebleu(generated_codes, reference_codes)
        
        # Custom Ruby metric (always available)
        results['ruby'] = self._calculate_ruby(generated_codes, reference_codes)
        
        return results
    
    def _calculate_bertscore(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate BERTScore."""
        try:
            results = self.bertscore.compute(
                predictions=generated_codes,
                references=reference_codes,
                lang="en"
            )
            return results['f1']
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return 0.0
    
    def _calculate_bleu(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            results = self.bleu.compute(
                predictions=generated_codes,
                references=[[ref] for ref in reference_codes]
            )
            return results['bleu']
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_rouge(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate ROUGE score."""
        try:
            results = self.rouge.compute(
                predictions=generated_codes,
                references=reference_codes
            )
            return results['rougeL']
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return 0.0
    
    def _calculate_codebleu(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate CodeBLEU score."""
        try:
            # CodeBLEU expects specific format
            results = self.calc_codebleu(
                references=[[ref] for ref in reference_codes],
                predictions=generated_codes,
                lang="python",
                weights=[0.25, 0.25, 0.25, 0.25]
            )
            return results['codebleu']
        except Exception as e:
            logger.warning(f"CodeBLEU calculation failed: {e}")
            return 0.0
    
    def _calculate_ruby(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate custom Ruby metric for code quality."""
        try:
            scores = []
            
            for code in generated_codes:
                # Syntax correctness (40%)
                syntax_score = self._check_syntax(code)
                
                # Code complexity (20%)
                complexity_score = self._analyze_complexity(code)
                
                # Code style (20%)
                style_score = self._analyze_style(code)
                
                # Execution test (20%)
                execution_score = self._test_execution(code)
                
                # Combined Ruby score
                ruby_score = (
                    syntax_score * 0.4 +
                    complexity_score * 0.2 +
                    style_score * 0.2 +
                    execution_score * 0.2
                )
                
                scores.append(ruby_score)
            
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Ruby metric calculation failed: {e}")
            return 0.0
    
    def _check_syntax(self, code: str) -> float:
        """Check syntax correctness of code."""
        try:
            import ast
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.0
    
    def _analyze_complexity(self, code: str) -> float:
        """Analyze code complexity."""
        try:
            import ast
            
            tree = ast.parse(code)
            
            # Count different constructs
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            
            # Calculate complexity score (simplified)
            total_complexity = functions + classes + loops + conditionals
            complexity_score = min(1.0, max(0.0, 1.0 - total_complexity / 20.0))
            
            return complexity_score
        except Exception:
            return 0.0
    
    def _analyze_style(self, code: str) -> float:
        """Analyze code style."""
        try:
            lines = code.split('\n')
            
            if not lines:
                return 0.0
            
            # Basic style metrics
            avg_line_length = np.mean([len(line) for line in lines if line.strip()])
            long_lines = sum(1 for line in lines if len(line) > 80)
            empty_lines = sum(1 for line in lines if not line.strip())
            
            # Style score (simplified)
            style_score = 1.0
            if avg_line_length > 100:
                style_score -= 0.2
            if long_lines / len(lines) > 0.1:
                style_score -= 0.2
            if empty_lines / len(lines) > 0.3:
                style_score -= 0.1
            
            return max(0.0, style_score)
        except Exception:
            return 0.0
    
    def _test_execution(self, code: str) -> float:
        """Test if code can be executed safely."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                }
            }
            
            # Try to compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, safe_globals)
            return 1.0
            
        except Exception:
            return 0.0
    
    def evaluate_against_targets(self, metrics: Dict[str, float], targets: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate if metrics meet target thresholds."""
        results = {}
        
        for metric_name, target in targets.items():
            if metric_name in metrics:
                results[metric_name] = metrics[metric_name] >= target
            else:
                results[metric_name] = False
        
        return results
    
    def get_summary(self, metrics: Dict[str, float], targets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get a summary of metrics and target achievement."""
        summary = {
            'metrics': metrics,
            'all_targets_met': True,
            'targets_met_count': 0,
            'targets_total': 0
        }
        
        if targets:
            target_results = self.evaluate_against_targets(metrics, targets)
            summary['targets_met'] = target_results
            summary['targets_met_count'] = sum(target_results.values())
            summary['targets_total'] = len(target_results)
            summary['all_targets_met'] = all(target_results.values())
        
        return summary
