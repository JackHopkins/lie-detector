#!/usr/bin/env python3
"""
Log Cache Manager for Inspect AI Evaluation Results

This module provides caching functionality by searching existing Inspect evaluation
logs (.eval files) to avoid recomputing evaluations that have already been completed.
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

# Import Inspect AI components for log reading
try:
    from inspect_ai.log import read_eval_log, list_eval_logs, EvalLog
    from inspect_ai.scorer import Score
    INSPECT_AVAILABLE = True
except ImportError:
    print("Warning: Inspect AI not available. Cache functionality will be limited.")
    INSPECT_AVAILABLE = False


@dataclass
class CachedEvaluationResult:
    """Results extracted from a cached evaluation log."""
    eval_fold: str
    epoch: str
    trained_fold: str
    log_file_path: str
    metrics: Dict[str, float]
    num_samples: int
    completion_time: Optional[str]
    model_ref: str
    task_name: str
    is_baseline: bool


@dataclass
class CacheSearchCriteria:
    """Criteria for searching cached evaluations."""
    trained_fold: str
    eval_fold: str
    epoch: str  # "epoch_0", "epoch_1", "baseline", etc.
    model_signature: Optional[str] = None  # For model-specific caching
    

class LogCacheManager:
    """Manages caching of evaluation results using Inspect AI logs."""
    
    def __init__(self, log_base_dir: str = "../../logs"):
        """
        Initialize log cache manager.
        
        Args:
            log_base_dir: Base directory for log storage
        """
        self.log_base_dir = Path(log_base_dir)
        self.cache_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "logs_scanned": 0,
            "invalid_logs": 0
        }
    
    def generate_cache_key(self, criteria: CacheSearchCriteria) -> str:
        """
        Generate a unique cache key for evaluation criteria.
        
        Args:
            criteria: Search criteria for the evaluation
            
        Returns:
            Unique cache key string
        """
        # Create a deterministic key from the criteria
        key_parts = [
            criteria.trained_fold,
            criteria.eval_fold,
            criteria.epoch,
            criteria.model_signature or "default"
        ]
        key_string = ":".join(key_parts)
        
        # Create a hash for consistency
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        return f"{key_string}:{key_hash}"
    
    def find_log_directories(self, trained_fold: str) -> List[Path]:
        """
        Find all potential log directories for a given trained fold.
        
        Args:
            trained_fold: Name of the training fold
            
        Returns:
            List of directory paths that might contain relevant logs
        """
        directories = []
        
        # Search in the new format: DD-MM-v1/{trained_fold}/
        for date_dir in self.log_base_dir.iterdir():
            if date_dir.is_dir() and re.match(r'\d{2}-\d{2}-v\d+', date_dir.name):
                trained_fold_dir = date_dir / trained_fold
                if trained_fold_dir.exists():
                    # Look for evaluation fold directories
                    for eval_fold_dir in trained_fold_dir.iterdir():
                        if eval_fold_dir.is_dir():
                            directories.append(eval_fold_dir)
        
        # Search in old format directories for backward compatibility
        for old_dir in self.log_base_dir.iterdir():
            if old_dir.is_dir() and old_dir.name.startswith("eval"):
                # Look for evaluation fold directories directly
                for eval_fold_dir in old_dir.iterdir():
                    if eval_fold_dir.is_dir():
                        directories.append(eval_fold_dir)
        
        return directories
    
    def extract_epoch_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract epoch information from a log filename.
        
        Args:
            filename: Name of the log file
            
        Returns:
            Epoch string (e.g., "epoch_0", "baseline") or None if not found
        """
        # Match patterns like:
        # eval-mask-factual-epoch1_*.eval -> epoch_1
        # eval-games-baseline_*.eval -> baseline
        # baseline_eval_games_*.eval -> baseline
        
        patterns = [
            r'eval-[^-]+-epoch(\d+)_',  # eval-{fold}-epoch{N}_
            r'eval-[^-]+-baseline_',    # eval-{fold}-baseline_
            r'baseline[-_]eval',        # baseline_eval or baseline-eval
            r'epoch[-_](\d+)',          # epoch_N or epoch-N
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if 'baseline' in pattern:
                    return "baseline"
                else:
                    epoch_num = match.group(1)
                    return f"epoch_{epoch_num}"
        
        return None
    
    def extract_eval_fold_from_path(self, log_path: Path) -> Optional[str]:
        """
        Extract evaluation fold name from log file path.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Evaluation fold name or None if not determinable
        """
        # The evaluation fold is typically the parent directory name
        # e.g., /logs/21-09-v1/mask-factual/games/eval_games_epoch0.eval
        #       -> eval_fold = "games"
        
        parent_dir = log_path.parent.name
        
        # Filter out date-based directories and trained fold directories
        if re.match(r'\d{2}-\d{2}-v\d+', parent_dir):
            return None
        if parent_dir in ['logs', 'eval']:
            return None
            
        return parent_dir
    
    def read_and_parse_log(self, log_path: Path) -> Optional[CachedEvaluationResult]:
        """
        Read and parse an Inspect evaluation log file.
        
        Args:
            log_path: Path to the .eval file
            
        Returns:
            Parsed evaluation result or None if parsing failed
        """
        if not INSPECT_AVAILABLE:
            print(f"Cannot read log {log_path}: Inspect AI not available")
            return None
        
        try:
            self.cache_stats["logs_scanned"] += 1
            
            # Read the evaluation log
            eval_log = read_eval_log(str(log_path))
            
            if not eval_log or not eval_log.samples:
                self.cache_stats["invalid_logs"] += 1
                return None
            
            # Extract basic information
            eval_fold = self.extract_eval_fold_from_path(log_path)
            epoch = self.extract_epoch_from_filename(log_path.name)
            
            if not eval_fold or not epoch:
                self.cache_stats["invalid_logs"] += 1
                return None
            
            # Extract metrics from samples
            scores = []
            for sample in eval_log.samples:
                if hasattr(sample, 'scores') and sample.scores:
                    # Look for the binary classification scorer results
                    if 'binary_classification_scorer' in sample.scores:
                        score = sample.scores['binary_classification_scorer']
                        scores.append(score)
            
            if not scores:
                self.cache_stats["invalid_logs"] += 1
                return None
            
            # Compute metrics using the same logic as epoch_eval.py
            metrics = self._compute_metrics_from_scores(scores)
            
            # Extract additional metadata
            model_ref = getattr(eval_log.eval, 'model', 'unknown')
            task_name = getattr(eval_log.eval, 'task', 'unknown')
            completion_time = getattr(eval_log.eval, 'completed', None)
            
            # Determine trained fold from task name or path structure
            trained_fold = self._extract_trained_fold_from_context(log_path, task_name)
            
            return CachedEvaluationResult(
                eval_fold=eval_fold,
                epoch=epoch,
                trained_fold=trained_fold or "unknown",
                log_file_path=str(log_path),
                metrics=metrics,
                num_samples=len(scores),
                completion_time=completion_time.isoformat() if completion_time else None,
                model_ref=str(model_ref),
                task_name=str(task_name),
                is_baseline=(epoch == "baseline")
            )
            
        except Exception as e:
            print(f"Error reading log {log_path}: {e}")
            self.cache_stats["invalid_logs"] += 1
            return None
    
    def _compute_metrics_from_scores(self, scores: List[Score]) -> Dict[str, float]:
        """
        Compute evaluation metrics from Inspect scores.
        
        Args:
            scores: List of Score objects from Inspect evaluation
            
        Returns:
            Dictionary of computed metrics
        """
        # For many cached results, we can extract pre-computed metrics directly
        if scores and hasattr(scores[0], 'metrics'):
            try:
                # Try to extract aggregate metrics if available
                metrics_dict = scores[0].metrics
                if isinstance(metrics_dict, dict):
                    return {
                        'accuracy': float(metrics_dict.get('accuracy', 0)),
                        'precision': float(metrics_dict.get('precision', 0)),
                        'recall': float(metrics_dict.get('recall', 0)),
                        'f1': float(metrics_dict.get('f1', 0)),
                        'num_samples': len(scores)
                    }
            except:
                pass
        
        # Fallback: compute from individual scores
        predictions = []
        targets = []
        
        for score in scores:
            # Extract prediction and target from score metadata and answer
            pred_value = None
            target_value = None
            
            # Get predicted value (from answer field)
            if hasattr(score, 'answer'):
                pred_value = score.answer
            
            # Get target value (from metadata)
            if hasattr(score, 'metadata') and score.metadata:
                if 'target' in score.metadata:
                    target_value = score.metadata['target']
                elif 'correct' in score.metadata:
                    # If we have correctness info, use it directly
                    is_correct = score.metadata['correct']
                    if pred_value is not None:
                        predictions.append(is_correct)
                        targets.append(True)  # Target is always "correct" in this case
                        continue
            
            # Convert to boolean if we have both values
            if pred_value is not None and target_value is not None:
                pred = self._answer_to_bool(pred_value)
                target = self._answer_to_bool(target_value)
                
                if pred is not None and target is not None:
                    predictions.append(pred)
                    targets.append(target)
        
        if not predictions:
            # If we can't extract individual scores, try to get aggregate from first score
            if scores:
                try:
                    # Some scorers might store aggregate metrics directly
                    first_score = scores[0]
                    if hasattr(first_score, 'value') and isinstance(first_score.value, dict):
                        metrics = first_score.value
                        return {
                            'accuracy': float(metrics.get('accuracy', 0)),
                            'precision': float(metrics.get('precision', 0)),
                            'recall': float(metrics.get('recall', 0)),
                            'f1': float(metrics.get('f1', 0)),
                            'num_samples': len(scores)
                        }
                except:
                    pass
            return {'num_samples': len(scores)}
        
        # Compute standard binary classification metrics
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            f1 = f1_score(targets, predictions, zero_division=0)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'num_samples': len(predictions)
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {'num_samples': len(scores)}
    
    def _score_to_bool(self, score_value: Any) -> Optional[bool]:
        """Convert various score formats to boolean."""
        if isinstance(score_value, bool):
            return score_value
        elif isinstance(score_value, (int, float)):
            return score_value > 0.5
        elif isinstance(score_value, str):
            return score_value.lower() in ['true', '1', 'yes', 'correct']
        else:
            return None
    
    def _answer_to_bool(self, answer_value: Any) -> Optional[bool]:
        """Convert answer values (like A/B choices) to boolean."""
        if isinstance(answer_value, bool):
            return answer_value
        elif isinstance(answer_value, str):
            # Map answer choices to boolean
            # Assuming A=True (truthful), B=False (deceptive) based on common convention
            if answer_value.upper() in ['A', 'TRUE', 'TRUTHFUL', '1']:
                return True
            elif answer_value.upper() in ['B', 'FALSE', 'DECEPTIVE', '0']:
                return False
        elif isinstance(answer_value, (int, float)):
            return answer_value > 0.5
        return None
    
    def _extract_trained_fold_from_context(self, log_path: Path, task_name: str) -> Optional[str]:
        """
        Extract the trained fold name from log path or task context.
        
        Args:
            log_path: Path to the log file
            task_name: Name of the evaluation task
            
        Returns:
            Trained fold name or None
        """
        # Try to extract from path structure
        # e.g., /logs/21-09-v1/mask-factual/games/ -> trained_fold = "mask-factual"
        path_parts = log_path.parts
        
        for i, part in enumerate(path_parts):
            if re.match(r'\d{2}-\d{2}-v\d+', part) and i + 1 < len(path_parts):
                return path_parts[i + 1]
        
        # Try to extract from task name patterns
        # e.g., "eval_games_epoch0" with context -> look for common fold names
        common_folds = [
            'mask-factual', 'cot-unfaithfulness', 'games', 'sycophancy',
            'self-sycophancy', 'sandbagging_ascii', 'sandbagging_other',
            'offpolicy_doluschat', 'offpolicy_halueval', 'offpolicy_truthisuniversal',
            'unanswerable'
        ]
        
        for fold in common_folds:
            if fold in str(log_path) or fold in task_name:
                return fold
        
        return None
    
    def find_cached_evaluation(self, criteria: CacheSearchCriteria) -> Optional[CachedEvaluationResult]:
        """
        Search for a cached evaluation matching the given criteria.
        
        Args:
            criteria: Search criteria for the evaluation
            
        Returns:
            Cached evaluation result or None if not found
        """
        cache_key = self.generate_cache_key(criteria)
        print(f"Searching for cached evaluation: {cache_key}")
        
        # Find potential directories
        log_directories = self.find_log_directories(criteria.trained_fold)
        
        for log_dir in log_directories:
            # Check if this directory is for the target evaluation fold
            if log_dir.name != criteria.eval_fold:
                continue
            
            # Search for .eval files in this directory
            for log_file in log_dir.glob("*.eval"):
                # Check if filename matches the epoch pattern
                file_epoch = self.extract_epoch_from_filename(log_file.name)
                if file_epoch != criteria.epoch:
                    continue
                
                # Parse the log file
                cached_result = self.read_and_parse_log(log_file)
                if cached_result:
                    print(f"Found cached evaluation: {log_file}")
                    self.cache_stats["cache_hits"] += 1
                    return cached_result
        
        print(f"No cached evaluation found for: {cache_key}")
        self.cache_stats["cache_misses"] += 1
        return None
    
    def find_multiple_cached_evaluations(
        self, 
        trained_fold: str, 
        eval_folds: List[str], 
        epoch: str
    ) -> Dict[str, Optional[CachedEvaluationResult]]:
        """
        Search for multiple cached evaluations across different eval folds.
        
        Args:
            trained_fold: Name of the training fold
            eval_folds: List of evaluation fold names to search for
            epoch: Epoch identifier to search for
            
        Returns:
            Dictionary mapping eval_fold -> cached result (or None if not found)
        """
        results = {}
        
        for eval_fold in eval_folds:
            criteria = CacheSearchCriteria(
                trained_fold=trained_fold,
                eval_fold=eval_fold,
                epoch=epoch
            )
            
            cached_result = self.find_cached_evaluation(criteria)
            results[eval_fold] = cached_result
        
        return results
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["cache_hits"] + self.cache_stats["cache_misses"]
        hit_rate = (self.cache_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate
        }
    
    def clear_cache_statistics(self) -> None:
        """Reset cache statistics."""
        self.cache_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "logs_scanned": 0,
            "invalid_logs": 0
        }


def convert_cached_result_to_evaluation_metrics(cached_result: CachedEvaluationResult) -> Dict[str, Any]:
    """
    Convert a cached result to the same format expected by the evaluation system.
    
    Args:
        cached_result: Cached evaluation result
        
    Returns:
        Dictionary in the same format as epoch evaluation results
    """
    return {
        **cached_result.metrics,
        'model_ref': cached_result.model_ref,
        'is_endpoint': True,  # Assume cached results were from endpoints
        'is_baseline': cached_result.is_baseline,
        'fold_name': cached_result.eval_fold,
        'epoch': int(cached_result.epoch.split('_')[1]) if '_' in cached_result.epoch else 0,
        'cached': True,  # Mark as cached for tracking
        'cache_source': cached_result.log_file_path
    }