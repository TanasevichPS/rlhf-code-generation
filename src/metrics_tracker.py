#!/usr/bin/env python3
"""Track training metrics and save to file."""

import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and save training metrics."""
    
    def __init__(self, output_dir: str = "./training_metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics file
        self.metrics_file = os.path.join(output_dir, "training_metrics.csv")
        self.detailed_metrics_file = os.path.join(output_dir, "detailed_metrics.json")
        
        # Initialize metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.metrics_file):
            pd.DataFrame(columns=[
                'timestamp', 'epoch', 'batch', 'reward', 'syntax_score', 
                'structure_score', 'bertscore', 'bleu', 'codebleu', 'rouge', 'ruby',
                'kl_divergence', 'policy_loss', 'value_loss'
            ]).to_csv(self.metrics_file, index=False)
    
    def calculate_bertscore(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate BERTScore metric."""
        try:
            from bert_score import BERTScorer
            scorer = BERTScorer(lang="en")
            P, R, F1 = scorer.score(generated_texts, reference_texts)
            return F1.mean().item()
        except ImportError:
            logger.warning("BERTScore not available, installing...")
            os.system("pip install bert-score")
            try:
                from bert_score import BERTScorer
                scorer = BERTScorer(lang="en")
                P, R, F1 = scorer.score(generated_texts, reference_texts)
                return F1.mean().item()
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def calculate_bleu(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method4
            
            scores = []
            for gen, ref in zip(generated_texts, reference_texts):
                # Tokenize
                gen_tokens = gen.split()
                ref_tokens = ref.split()
                
                if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                    score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                    scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("NLTK not available, installing...")
            os.system("pip install nltk")
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothie = SmoothingFunction().method4
                
                scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    gen_tokens = gen.split()
                    ref_tokens = ref.split()
                    
                    if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                        scores.append(score)
                
                return sum(scores) / len(scores) if scores else 0.0
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_codebleu(self, generated_codes: List[str], reference_codes: List[str]) -> float:
        """Calculate CodeBLEU score."""
        try:
            # Some codebleu implementations expect files; to be robust we write temporary files
            # Defensive: filter out pairs where reference is empty and ensure equal lengths
            paired = [(p, r) for p, r in zip(generated_codes, reference_codes) if (r and r.strip())]
            if not paired:
                logger.warning("No non-empty references available for CodeBLEU calculation")
                return 0.0
            preds_filtered, refs_filtered = zip(*paired)

            try:
                from codebleu import calc_codebleu
            except Exception as imp_e:
                logger.exception(f"Failed to import codebleu.calc_codebleu: {imp_e}")
                raise
            import tempfile
            import json

            with tempfile.TemporaryDirectory() as td:
                refs_path = os.path.join(td, 'refs.txt')
                preds_path = os.path.join(td, 'preds.txt')
                # Write one code per line
                with open(refs_path, 'w', encoding='utf-8') as fr:
                    for r in refs_filtered:
                        fr.write(r.replace('\n', '\\n') + '\n')
                with open(preds_path, 'w', encoding='utf-8') as fp:
                    for p in preds_filtered:
                        fp.write(p.replace('\n', '\\n') + '\n')

                # Call calc_codebleu - prefer list API, but fall back to file-based or weighted ngram match
                try:
                    # Prefer list API when available. Provide references as a list of
                    # single-reference lists: [[ref1], [ref2], ...] and predictions
                    # as a flat list so shapes match typical list-based APIs.
                    results = calc_codebleu(references=[[r] for r in refs_filtered], predictions=list(preds_filtered), lang="python")
                except Exception as e:
                    logger.warning(f"calc_codebleu failed, attempting fallback: {e}")
                    # Try file-based invocation if supported
                    try:
                        results = calc_codebleu(refs_path, preds_path, lang="python")
                    except Exception as e2:
                        logger.warning(f"file-based calc_codebleu also failed: {e2}")
                        # Try weighted_ngram_match as a lightweight fallback
                        try:
                            from codebleu.codebleu import weighted_ngram_match
                            # weighted_ngram_match expects a list of reference-lists
                            # and a list of tokenized predictions. We provide refs as
                            # [[tokens_of_ref1], [tokens_of_ref2], ...]
                            refs_tokenized = [[r.split()] for r in refs_filtered]
                            preds_tokenized = [p.split() for p in preds_filtered]
                            wg = weighted_ngram_match(refs_tokenized, preds_tokenized, lang='python')
                            results = {'codebleu': wg}
                        except Exception as e3:
                            logger.warning(f"weighted_ngram_match fallback failed: {e3}")
                            # Final fallback: compute simple BLEU average
                            try:
                                bleu_fallback = self.calculate_bleu(list(preds_filtered), list(refs_filtered))
                                results = {'codebleu': bleu_fallback}
                            except Exception as e4:
                                logger.warning(f"Final BLEU fallback failed: {e4}")
                                results = None

            # codebleu implementations may return dict or float
            if isinstance(results, dict):
                return results.get('codebleu', 0.0) or results.get('total', 0.0)
            elif isinstance(results, (float, int)):
                return float(results)
            else:
                return 0.0
        except Exception as e:
            # Log full traceback for debugging any failures during import or execution
            logger.exception("CodeBLEU calculation failed")
            return 0.0
    
    def calculate_rouge(self, generated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate ROUGE score."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            
            for gen, ref in zip(generated_texts, reference_texts):
                score = scorer.score(ref, gen)
                scores.append(score['rougeL'].fmeasure)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except ImportError:
            logger.warning("ROUGE not available, installing...")
            os.system("pip install rouge-score")
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                scores = []
                
                for gen, ref in zip(generated_texts, reference_texts):
                    score = scorer.score(ref, gen)
                    scores.append(score['rougeL'].fmeasure)
                
                return sum(scores) / len(scores) if scores else 0.0
            except:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating ROUGE: {e}")
            return 0.0
    
    def calculate_metrics(self, prompts: List[str], generated_texts: List[str], reference_texts: List[str] = None) -> Dict[str, float]:
        """Calculate all metrics for generated texts."""
        if reference_texts is None:
            # Use prompts as simple references for demonstration
            reference_texts = prompts
        # Defensive: many metric implementations crash or return invalid results
        # when reference strings are empty. Compute each metric on the subset
        # of (gen, ref) pairs where ref is non-empty. If no valid pairs exist,
        # return 0.0 for that metric.

        def _filter_nonempty_refs(gens, refs):
            paired = [(g, r) for g, r in zip(gens, refs) if r and r.strip()]
            if not paired:
                return [], []
            gs, rs = zip(*paired)
            return list(gs), list(rs)

        # BERTScore
        gs_bs, rs_bs = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_bs:
            bscore = self.calculate_bertscore(gs_bs, rs_bs)
        else:
            logger.warning("Empty reference sentence detected; setting raw BERTScores to 0.")
            bscore = 0.0

        # BLEU
        gs_bleu, rs_bleu = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_bleu:
            bleu = self.calculate_bleu(gs_bleu, rs_bleu)
        else:
            bleu = 0.0

        # CodeBLEU (function already filters internally but keep count logging)
        gs_cb, rs_cb = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_cb:
            codebleu = self.calculate_codebleu(gs_cb, rs_cb)
        else:
            logger.warning("No non-empty references available for CodeBLEU calculation")
            codebleu = 0.0

        # ROUGE
        gs_rouge, rs_rouge = _filter_nonempty_refs(generated_texts, reference_texts)
        if gs_rouge:
            rouge = self.calculate_rouge(gs_rouge, rs_rouge)
        else:
            rouge = 0.0

        metrics = {
            'bertscore': bscore,
            'bleu': bleu,
            'codebleu': codebleu,
            'rouge': rouge,
        }
        # Try to compute RUBY metric if available; surface exceptions to logs
        try:
            from src.metrics.ruby_metric import RUBYMetric
            ruby = RUBYMetric()
            # compute average ruby only over non-empty reference pairs
            gs_ruby, rs_ruby = _filter_nonempty_refs(generated_texts, reference_texts)
            ruby_scores = []
            for gen, ref in zip(gs_ruby, rs_ruby):
                try:
                    score = ruby.compute_ruby(ref, gen)
                    if score is not None:
                        ruby_scores.append(score)
                except Exception as e:
                    logger.warning(f"RUBY compute failed for one pair: {e}")
                    continue
            metrics['ruby'] = sum(ruby_scores) / len(ruby_scores) if ruby_scores else 0.0
        except Exception as e:
            logger.warning(f"RUBY metric not available or failed: {e}")
            metrics['ruby'] = 0.0
        
        return metrics
    
    def record_batch_metrics(self, epoch: int, batch: int, batch_stats: Dict[str, float], 
                           prompts: List[str] = None, generated_texts: List[str] = None):
        """Record metrics for a training batch."""
        timestamp = datetime.now().isoformat()
        
        # Basic metrics from batch_stats
        metrics_record = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'reward': batch_stats.get('mean_reward', 0),
            'syntax_score': batch_stats.get('syntax_score', 0),
            'structure_score': batch_stats.get('structure_score', 0),
            'kl_divergence': batch_stats.get('kl_divergence', 0),
            'policy_loss': batch_stats.get('policy_loss', 0),
            'value_loss': batch_stats.get('value_loss', 0),
        }
        
        # Calculate additional metrics if texts are provided
        if prompts is not None and generated_texts is not None:
            try:
                additional_metrics = self.calculate_metrics(prompts, generated_texts)
                metrics_record.update(additional_metrics)
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
                # Set default values for failed metrics
                metrics_record.update({
                    'bertscore': 0.0,
                    'bleu': 0.0,
                    'codebleu': 0.0,
                    'rouge': 0.0,
                })
        
        # Add to history
        self.metrics_history.append(metrics_record)
        
        # Save to CSV
        try:
            df = pd.DataFrame([metrics_record])
            df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Error saving metrics to CSV: {e}")
        
        # Log metrics
        logger.info(f"Metrics - Epoch {epoch}, Batch {batch}: "
                   f"Reward: {metrics_record['reward']:.4f}, "
                   f"Syntax: {metrics_record['syntax_score']:.4f}, "
                   f"BERTScore: {metrics_record.get('bertscore', 0):.4f}, "
                   f"BLEU: {metrics_record.get('bleu', 0):.4f}")
        
        return metrics_record
    
    def save_detailed_metrics(self):
        """Save detailed metrics to JSON file."""
        try:
            with open(self.detailed_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed metrics saved to: {self.detailed_metrics_file}")
        except Exception as e:
            logger.error(f"Error saving detailed metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        summary = {}
        
        for column in df.columns:
            if column not in ['timestamp', 'epoch', 'batch']:
                summary[f'avg_{column}'] = df[column].mean()
                summary[f'std_{column}'] = df[column].std()
                summary[f'min_{column}'] = df[column].min()
                summary[f'max_{column}'] = df[column].max()
        
        return summary
    
    def plot_metrics(self, metrics_to_plot: List[str] = None):
        """Plot training metrics (optional)."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                logger.warning("No metrics data to plot")
                return
            
            df = pd.DataFrame(self.metrics_history)
            
            if metrics_to_plot is None:
                metrics_to_plot = ['reward', 'syntax_score', 'bertscore', 'bleu']
            
            # Create subplots
            n_metrics = len(metrics_to_plot)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    axes[i].plot(df['batch'], df[metric], label=metric, alpha=0.7)
                    axes[i].set_ylabel(metric)
                    axes[i].legend()
                    
                    # Add epoch separators
                    epoch_changes = df[df['epoch'].diff() != 0].index
                    for change_idx in epoch_changes:
                        axes[i].axvline(x=change_idx, color='red', alpha=0.3, linestyle='--')
            
            plt.xlabel('Batch')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "training_metrics_plot.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Metrics plot saved to: {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")