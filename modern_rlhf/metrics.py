"""
Modern Evaluation Metrics for Code Generation
============================================

Comprehensive evaluation metrics for code generation tasks including:
- BERTScore for semantic similarity
- CodeBLEU for code-specific evaluation
- BLEU for n-gram overlap
- ROUGE for summarization metrics
- Custom Ruby metric for code quality
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import re
import ast
import subprocess
import tempfile
import os

# Import evaluation libraries
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("BERTScore not available. Install with: pip install bert-score")

try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False
    logging.warning("CodeBLEU not available. Install with: pip install codebleu")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("ROUGE not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("BLEU not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CodeQualityAnalyzer:
    """Analyzer for code quality metrics."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None
    
    def analyze_syntax(self, code: str) -> Dict[str, Any]:
        """Analyze syntax correctness of code."""
        try:
            # Try to parse the code
            ast.parse(code)
            return {
                "syntax_correct": True,
                "syntax_error": None,
                "syntax_score": 1.0
            }
        except SyntaxError as e:
            return {
                "syntax_correct": False,
                "syntax_error": str(e),
                "syntax_score": 0.0
            }
        except Exception as e:
            return {
                "syntax_correct": False,
                "syntax_error": f"Parse error: {str(e)}",
                "syntax_score": 0.0
            }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            tree = ast.parse(code)
            
            # Count different constructs
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            
            # Calculate complexity score (simplified)
            complexity_score = min(1.0, max(0.0, 1.0 - (functions + classes + loops + conditionals) / 20.0))
            
            return {
                "functions": functions,
                "classes": classes,
                "loops": loops,
                "conditionals": conditionals,
                "complexity_score": complexity_score
            }
        except Exception as e:
            return {
                "functions": 0,
                "classes": 0,
                "loops": 0,
                "conditionals": 0,
                "complexity_score": 0.0,
                "error": str(e)
            }
    
    def analyze_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style metrics."""
        lines = code.split('\n')
        
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
        
        return {
            "avg_line_length": avg_line_length,
            "long_lines": long_lines,
            "empty_lines": empty_lines,
            "style_score": max(0.0, style_score)
        }


class ModernMetricsEvaluator:
    """Modern metrics evaluator for code generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.code_analyzer = CodeQualityAnalyzer()
        self.rouge_scorer = None
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BERTScore for semantic similarity."""
        # Fallback to a simple token-overlap based proxy if bert-score package is unavailable
        if not BERTSCORE_AVAILABLE:
            try:
                scores = []
                for pred, ref in zip(predictions, references):
                    if not pred or not ref:
                        scores.append(0.0)
                        continue
                    pred_tokens = set(pred.split())
                    ref_tokens = set(ref.split())
                    if not ref_tokens:
                        scores.append(0.0)
                        continue
                    overlap = len(pred_tokens & ref_tokens) / max(1, len(ref_tokens))
                    scores.append(overlap)
                score = float(np.mean(scores)) if scores else 0.0
                return MetricResult(metric_name="bertscore", score=score, details={"method": "token_overlap_proxy"})
            except Exception as e:
                return MetricResult(metric_name="bertscore", score=0.0, error=str(e))
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            
            # Return F1 score (harmonic mean of precision and recall)
            score = float(F1.mean())
            
            return MetricResult(
                metric_name="bertscore",
                score=score,
                details={
                    "precision": float(P.mean()),
                    "recall": float(R.mean()),
                    "f1": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="bertscore",
                score=0.0,
                error=str(e)
            )
    
    def compute_codebleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute CodeBLEU for code-specific evaluation."""
        # Fallback: simple token-precision proxy if codebleu not installed
        if not CODEBLEU_AVAILABLE:
            try:
                results = []
                for pred, ref in zip(predictions, references):
                    if not pred or not ref:
                        results.append(0.0)
                        continue
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                    match = sum(1 for t in pred_tokens if t in ref_tokens)
                    prec = match / max(1, len(pred_tokens))
                    results.append(prec)
                score = float(np.mean(results)) if results else 0.0
                return MetricResult(metric_name="codebleu", score=score, details={"method": "precision_proxy"})
            except Exception as e:
                return MetricResult(metric_name="codebleu", score=0.0, error=str(e))
        
        try:
            # CodeBLEU expects specific format
            results = []
            for pred, ref in zip(predictions, references):
                try:
                    # Ensure we have valid strings
                    if not pred or not ref:
                        results.append(0.0)
                        continue
                    
                    # CodeBLEU expects references as list of strings
                    score = calc_codebleu(
                        [ref], pred, lang="python", weights=[0.25, 0.25, 0.25, 0.25]
                    )
                    results.append(score)
                except Exception as e:
                    logger.warning(f"CodeBLEU computation failed for sample: {e}")
                    results.append(0.0)
            
            score = np.mean(results) if results else 0.0
            
            return MetricResult(
                metric_name="codebleu",
                score=score,
                details={"individual_scores": results}
            )
        except Exception as e:
            return MetricResult(
                metric_name="codebleu",
                score=0.0,
                error=str(e)
            )
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BLEU score for n-gram overlap."""
        # Fallback simple unigram-precision BLEU if nltk not available
        if not BLEU_AVAILABLE:
            try:
                results = []
                for pred, ref in zip(predictions, references):
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                    if len(pred_tokens) == 0:
                        results.append(0.0)
                        continue
                    match = sum(1 for t in pred_tokens if t in ref_tokens)
                    score = match / max(1, len(pred_tokens))
                    results.append(score)
                score = float(np.mean(results)) if results else 0.0
                return MetricResult(metric_name="bleu", score=score, details={"method": "unigram_precision"})
            except Exception as e:
                return MetricResult(metric_name="bleu", score=0.0, error=str(e))
        
        try:
            results = []
            for pred, ref in zip(predictions, references):
                # Tokenize (simple whitespace tokenization)
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(pred_tokens) == 0:
                    results.append(0.0)
                    continue
                
                # Compute BLEU score
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.code_analyzer.smoothing)
                results.append(score)
            
            score = np.mean(results)
            
            return MetricResult(
                metric_name="bleu",
                score=score,
                details={"individual_scores": results}
            )
        except Exception as e:
            return MetricResult(
                metric_name="bleu",
                score=0.0,
                error=str(e)
            )
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute ROUGE scores for summarization metrics."""
        # Fallback: approximate ROUGE-L by longest-common-subsequence ratio if rouge-score not installed
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            try:
                def lcs_len(a: List[str], b: List[str]) -> int:
                    # simple DP LCS
                    la, lb = len(a), len(b)
                    dp = [[0] * (lb + 1) for _ in range(la + 1)]
                    for i in range(la - 1, -1, -1):
                        for j in range(lb - 1, -1, -1):
                            if a[i] == b[j]:
                                dp[i][j] = 1 + dp[i + 1][j + 1]
                            else:
                                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    return dp[0][0]

                rougeL_scores = []
                for pred, ref in zip(predictions, references):
                    p_tokens = pred.split()
                    r_tokens = ref.split()
                    if not r_tokens:
                        rougeL_scores.append(0.0)
                        continue
                    lcs = lcs_len(p_tokens, r_tokens)
                    rougeL_scores.append(lcs / max(1, len(r_tokens)))
                score = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
                return MetricResult(metric_name="rouge", score=score, details={"method": "lcs_proxy"})
            except Exception as e:
                return MetricResult(metric_name="rouge", score=0.0, error=str(e))
        
        try:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # Return average ROUGE-L score
            score = np.mean(rougeL_scores)
            
            return MetricResult(
                metric_name="rouge",
                score=score,
                details={
                    "rouge1": np.mean(rouge1_scores),
                    "rouge2": np.mean(rouge2_scores),
                    "rougeL": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="rouge",
                score=0.0,
                error=str(e)
            )
    
    def compute_ruby(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute custom Ruby metric for code quality."""
        try:
            results = []
            
            for pred, ref in zip(predictions, references):
                # Analyze syntax
                syntax_analysis = self.code_analyzer.analyze_syntax(pred)
                syntax_score = syntax_analysis["syntax_score"]
                
                # Analyze complexity
                complexity_analysis = self.code_analyzer.analyze_complexity(pred)
                complexity_score = complexity_analysis["complexity_score"]
                
                # Analyze style
                style_analysis = self.code_analyzer.analyze_style(pred)
                style_score = style_analysis["style_score"]
                
                # Simple execution test (if possible)
                execution_score = self._test_execution(pred)
                
                # Combined Ruby score
                ruby_score = (
                    syntax_score * 0.4 +
                    complexity_score * 0.2 +
                    style_score * 0.2 +
                    execution_score * 0.2
                )
                
                results.append(ruby_score)
            
            score = np.mean(results)
            
            return MetricResult(
                metric_name="ruby",
                score=score,
                details={
                    "syntax_scores": [self.code_analyzer.analyze_syntax(p)["syntax_score"] for p in predictions],
                    "complexity_scores": [self.code_analyzer.analyze_complexity(p)["complexity_score"] for p in predictions],
                    "style_scores": [self.code_analyzer.analyze_style(p)["style_score"] for p in predictions],
                    "execution_scores": [self._test_execution(p) for p in predictions]
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="ruby",
                score=0.0,
                error=str(e)
            )
    
    def _test_execution(self, code: str) -> float:
        """Test if code can be executed (simplified version)."""
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
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, safe_globals)
            return 1.0
            
        except Exception:
            return 0.0
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, MetricResult]:
        """Compute all available metrics."""
        metrics = {}
        
        # Compute each metric
        metrics["bertscore"] = self.compute_bertscore(predictions, references)
        metrics["codebleu"] = self.compute_codebleu(predictions, references)
        metrics["bleu"] = self.compute_bleu(predictions, references)
        metrics["rouge"] = self.compute_rouge(predictions, references)
        metrics["ruby"] = self.compute_ruby(predictions, references)
        
        return metrics
    
    def evaluate_against_targets(self, metrics: Dict[str, MetricResult], targets: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate if metrics meet target thresholds."""
        results = {}
        
        for metric_name, target in targets.items():
            if metric_name in metrics:
                results[metric_name] = metrics[metric_name].score >= target
            else:
                results[metric_name] = False
        
        return results
    
    def get_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "scores": {},
            "errors": {},
            "overall_success": True
        }
        
        for metric_name, result in metrics.items():
            summary["scores"][metric_name] = result.score
            if result.error:
                summary["errors"][metric_name] = result.error
                summary["overall_success"] = False
        
        return summary


# Utility functions for batch evaluation
def evaluate_batch(
    predictions: List[str],
    references: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a batch of predictions against references."""
    evaluator = ModernMetricsEvaluator(config)
    metrics = evaluator.compute_all_metrics(predictions, references)
    summary = evaluator.get_summary(metrics)
    
    return {
        "metrics": metrics,
        "summary": summary
    }


def evaluate_single(
    prediction: str,
    reference: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a single prediction against a reference."""
    return evaluate_batch([prediction], [reference], config)
