#!/usr/bin/env python3
"""
Script to run lie detection evaluation using the Inspect framework.
Supports baseline evaluation against non-finetuned models.

Usage:
    # Run baseline evaluation
    python run_eval.py \
        --base-path /Users/jackhopkins/PycharmProjects/lie-detector \
        --fold-name sandbagging_ascii \
        --model-name gpt_oss_120b \
        --baseline

    # Run normal epoch evaluation
    python run_eval.py \
        --base-path /Users/jackhopkins/PycharmProjects/lie-detector \
        --fold-name sandbagging_ascii \
        --model-name gpt_oss_120b
"""
import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from log_cache_manager import CacheSearchCriteria, convert_cached_result_to_evaluation_metrics

# Load environment variables from .env file
load_dotenv()

# Import Inspect AI components
from inspect_ai import Task, eval_async
from inspect_ai.dataset import MemoryDataset, Sample


#from together_training.epoch.epoch_eval import chat_binary_classification_scorer


async def evaluate_model_on_fold_single(
        model_ref: str,
        eval_samples: List[Sample],
        fold_name: str,
        epoch: int,
        trained_fold: str,
        limit: Optional[int] = None,
        is_endpoint: bool = False,
        is_baseline: bool = False,
        log_dir: Optional[str] = None,
        run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a single model on a single fold.

    Args:
        model_ref: Together AI model ID or endpoint name
        eval_samples: List of evaluation samples
        fold_name: Name of the evaluation fold
        epoch: Model epoch number (0 for baseline)
        trained_fold: Name of the training fold
        limit: Optional limit on number of samples
        is_endpoint: Whether model_ref is an endpoint name
        is_baseline: Whether this is a baseline evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    from epoch_eval import (
        lie_detection_solver,
        binary_classification_scorer,
        compute_metrics
    )

    # Create dataset
    dataset = MemoryDataset(samples=eval_samples[:limit] if limit else eval_samples)

    # Create task with appropriate name
    if is_baseline:
        task_name = f"baseline_eval_{fold_name}"
    else:
        task_name = f"eval_{fold_name}_epoch{epoch}"

    task = Task(
        dataset=dataset,
        solver=lie_detection_solver(),
        scorer=[binary_classification_scorer()], #,chat_binary_classification_scorer()],
        name=task_name
    )

    # Run evaluation
    print(
        f"    Running {'baseline' if is_baseline else f'epoch {epoch}'} evaluation on {len(dataset.samples)} samples...")

    try:
        # Use Together AI endpoint or model ID
        model_string = f"together/{model_ref}"
        ref_type = "baseline model" if is_baseline else ("endpoint" if is_endpoint else "model ID")
        print(f"    Using {ref_type}: {model_ref}")

        # Use provided log_dir or generate default format
        if log_dir:
            eval_log_dir = log_dir
        else:
            from datetime import datetime
            date_str = datetime.now().strftime("%d-%m")
            # Use run_id if provided, otherwise fall back to default version
            version = run_id if run_id else "v2"
            log_dir_suffix = "baseline" if is_baseline else f"{fold_name}"
            eval_log_dir = f"../../logs/{date_str}/{version}/{trained_fold}/{log_dir_suffix}"
        
        log = await eval_async(
            task,
            model=model_string,
            temperature=0.0,  # Deterministic for evaluation
            log_dir=eval_log_dir,
            max_tasks=5,
            max_connections=20,
            fail_on_error=0.3
        )

        # Extract scores
        # scores = []
        # for sample in log[0].samples:
        #     if hasattr(sample, 'scores') and 'binary_classification_scorer' in sample.scores:
        #         scores.append(sample.scores['binary_classification_scorer'])

        # After running evaluation, extract scores AND probabilities
        scores = []
        probabilities = []  # Store probability of positive class (A = "I lied")
        targets = []  # Store true labels

        for sample in log[0].samples:
            if hasattr(sample, 'scores') and 'binary_classification_scorer' in sample.scores:
                score = sample.scores['binary_classification_scorer']
                scores.append(score)

                # Extract probability from metadata
                if hasattr(sample, 'metadata') and sample.metadata:
                    # Get probability of class A (positive class)
                    a_prob = sample.metadata.get('a_probability', None)
                    if a_prob is not None:
                        probabilities.append(a_prob)
                        # Extract true label (1 for A, 0 for B)
                        target = sample.metadata.get('target', '')
                        targets.append(1 if target == 'A' else 0)

        # Compute metrics
        metrics = compute_metrics(scores)

        metrics['model_ref'] = model_ref
        metrics['is_endpoint'] = is_endpoint
        metrics['is_baseline'] = is_baseline
        metrics['fold_name'] = fold_name
        metrics['epoch'] = epoch

        if probabilities and targets:
            from sklearn.metrics import roc_auc_score, roc_curve
            auroc = roc_auc_score(targets, probabilities)
            fpr, tpr, thresholds = roc_curve(targets, probabilities)

            metrics['auroc'] = auroc
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
            metrics['probabilities'] = probabilities
            metrics['true_labels'] = targets

        return metrics

    except Exception as e:
        print(f"    Error during evaluation: {e}")
        return {
            'error': str(e),
            'model_ref': model_ref,
            'is_endpoint': is_endpoint,
            'is_baseline': is_baseline,
            'fold_name': fold_name,
            'epoch': epoch
        }


async def evaluate_model_on_all_folds_parallel(
        model_ref: str,
        eval_folds: List,
        epoch: int,
        trained_fold: str,
        limit: Optional[int] = None,
        is_endpoint: bool = False,
        is_baseline: bool = False,
        log_dir: Optional[str] = None,
        cache_manager: Optional = None,
        cache_only: bool = False,
        run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a single model on all folds in parallel using Inspect's parallel task execution.

    Args:
        model_ref: Together AI model ID or endpoint name
        eval_folds: List of evaluation folds
        epoch: Model epoch number (0 for baseline)
        trained_fold: Name of the training fold
        limit: Optional limit on number of samples
        is_endpoint: Whether model_ref is an endpoint name
        is_baseline: Whether this is a baseline evaluation

    Returns:
        Dictionary with evaluation results for all folds
    """
    from epoch_eval import (
        lie_detection_solver,
        binary_classification_scorer,
        compute_metrics,
        load_jsonl_samples,
        prepare_eval_sample
    )

    print(f"🚀 Running parallel evaluation for {'baseline' if is_baseline else f'epoch {epoch}'} on {len(eval_folds)} folds")

    # Check cache first if cache manager is available
    cached_results = {}
    folds_to_evaluate = []
    
    if cache_manager:
        print(f"🗂️  Checking cache for existing evaluations...")
        
        # Determine epoch string for cache lookup
        if is_baseline:
            epoch_str = "baseline"
        else:
            epoch_str = f"epoch_{epoch}"
        
        # Check cache for each fold
        for eval_fold in eval_folds:
            criteria = CacheSearchCriteria(
                trained_fold=trained_fold,
                eval_fold=eval_fold.name,
                epoch=epoch_str
            )
            
            cached_result = cache_manager.find_cached_evaluation(criteria)
            if cached_result:
                # Convert cached result to expected format
                metrics = convert_cached_result_to_evaluation_metrics(cached_result)
                cached_results[eval_fold.name] = metrics
                print(f"  💾 {eval_fold.name}: Found cached results ({str(metrics)})")
            else:
                folds_to_evaluate.append(eval_fold)
                print(f"  ❌ {eval_fold.name}: No cached results found")
        
        # If cache_only mode and some folds not cached, return error
        if cache_only and folds_to_evaluate:
            print(f"🚫 Cache-only mode: {len(folds_to_evaluate)} folds not cached, aborting evaluation")
            error_results = {}
            for fold in folds_to_evaluate:
                error_results[fold.name] = {
                    'error': 'No cached results available (cache-only mode)',
                    'model_ref': model_ref,
                    'is_endpoint': is_endpoint,
                    'is_baseline': is_baseline,
                    'fold_name': fold.name,
                    'epoch': epoch
                }
            # Combine cached and error results
            return {**cached_results, **error_results}
            
        print(f"📊 Cache summary: {len(cached_results)} cached, {len(folds_to_evaluate)} to evaluate")
    else:
        folds_to_evaluate = eval_folds
        print(f"🚫 Cache disabled - evaluating all {len(folds_to_evaluate)} folds")

    # If all results are cached, return them
    if not folds_to_evaluate:
        print(f"🎉 All results found in cache!")
        return cached_results

    # Create tasks for folds that need evaluation
    tasks = []
    fold_names = []
    
    for eval_fold in folds_to_evaluate:
        print(f"  📋 Preparing task for fold: {eval_fold.name}")
        
        # Load and prepare samples for this fold
        split = "train"
        samples_path = eval_fold.train_path
        
        try:
            raw_samples = load_jsonl_samples(samples_path)
            print(f"    Loaded {len(raw_samples)} raw samples from {eval_fold.name}")
            
            # Prepare samples
            eval_samples = []
            for raw_sample in raw_samples:
                try:
                    sample = prepare_eval_sample(raw_sample)
                    eval_samples.append(sample)
                except Exception as e:
                    print(f"    Warning: Failed to prepare sample: {e}")
            
            print(f"    Prepared {len(eval_samples)} evaluation samples for {eval_fold.name}")
            
            # Create dataset
            dataset = MemoryDataset(samples=eval_samples[:limit] if limit else eval_samples)
            
            # Create task with appropriate name
            if is_baseline:
                task_name = f"baseline_eval_{eval_fold.name}"
            else:
                task_name = f"eval_{eval_fold.name}_epoch{epoch}"
            
            task = Task(
                dataset=dataset,
                solver=lie_detection_solver(),
                scorer=[binary_classification_scorer()],#, chat_binary_classification_scorer()],
                name=task_name
            )
            
            tasks.append(task)
            fold_names.append(eval_fold.name)
            
        except Exception as e:
            print(f"    ❌ Error preparing fold {eval_fold.name}: {e}")
            # Add a placeholder for this fold
            tasks.append(None)
            fold_names.append(eval_fold.name)
    
    # Filter out None tasks
    valid_tasks = [(task, name) for task, name in zip(tasks, fold_names) if task is not None]
    
    if not valid_tasks:
        print("    ❌ No valid tasks to run")
        return {}
    
    print(f"  🏃‍♂️ Running {len(valid_tasks)} tasks in parallel...")
    
    # Run all tasks in parallel using Inspect
    try:
        model_string = f"together/{model_ref}"
        ref_type = "baseline model" if is_baseline else ("endpoint" if is_endpoint else "model ID")
        print(f"    Using {ref_type}: {model_ref}")
        
        # Use provided log_dir or generate default format
        if log_dir:
            eval_log_dir = log_dir
        else:
            from datetime import datetime
            date_str = datetime.now().strftime("%d-%m")
            # Use run_id if provided, otherwise fall back to default version
            version = run_id if run_id else "default"
            eval_log_dir = f"../../logs/{date_str}/{version}/{trained_fold}/epoch_{epoch if not is_baseline else 'baseline'}"
        
        # Run all tasks in parallel
        logs = await eval_async(
            [task for task, _ in valid_tasks],
            model=model_string,
            #model="fellows_safety/gpt-oss-120b-mask-test-all-train-c7857cd1"
            temperature=0.0,  # Deterministic for evaluation
            fail_on_error=0.3,
            log_dir=eval_log_dir,
        )
        
        # Process results
        results = {}
        
        for i, (task, fold_name) in enumerate(valid_tasks):
            try:
                log = logs[i]
                
                # Extract scores
                scores = []
                for sample in log.samples:
                    if hasattr(sample, 'scores') and 'binary_classification_scorer' in sample.scores:
                        scores.append(sample.scores['binary_classification_scorer'])
                
                # Compute metrics
                metrics = compute_metrics(scores)
                metrics['model_ref'] = model_ref
                metrics['is_endpoint'] = is_endpoint
                metrics['is_baseline'] = is_baseline
                metrics['fold_name'] = fold_name
                metrics['epoch'] = epoch
                
                results[fold_name] = metrics
                
                print(f"    ✅ {fold_name}: F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error processing results for {fold_name}: {e}")
                results[fold_name] = {
                    'error': str(e),
                    'model_ref': model_ref,
                    'is_endpoint': is_endpoint,
                    'is_baseline': is_baseline,
                    'fold_name': fold_name,
                    'epoch': epoch
                }
        
        print(f"  🎉 Parallel evaluation completed for {len(results)} folds")
        
        # Combine cached and computed results
        final_results = {**cached_results, **results}
        print(f"📊 Total results: {len(final_results)} folds ({len(cached_results)} cached + {len(results)} computed)")
        return final_results
        
    except Exception as e:
        print(f"    ❌ Error during parallel evaluation: {e}")
        # Return error results for all folds
        error_results = {}
        for _, fold_name in valid_tasks:
            error_results[fold_name] = {
                'error': str(e),
                'model_ref': model_ref,
                'is_endpoint': is_endpoint,
                'is_baseline': is_baseline,
                'fold_name': fold_name,
                'epoch': epoch
            }
        return error_results


def generate_run_id_from_models(trained_models: List, baseline_model_id: Optional[str] = None) -> str:
    """
    Generate a run ID by hashing all model IDs used in the evaluation.
    
    Args:
        trained_models: List of trained model objects with model_id attribute
        baseline_model_id: Optional baseline model ID to include in hash
    
    Returns:
        SHA-256 hash truncated to 8 characters as run ID
    """
    # Collect all model IDs
    model_ids = []
    
    # Add baseline model if provided
    if baseline_model_id:
        model_ids.append(baseline_model_id)
    
    # Add trained model IDs
    for model in trained_models:
        model_ids.append(model.model_id)
    
    # Sort for consistent ordering
    model_ids.sort()
    
    # Create hash from concatenated model IDs
    combined_string = '|'.join(model_ids)
    hash_obj = hashlib.sha256(combined_string.encode('utf-8'))
    
    # Return first 8 characters of hex digest
    return hash_obj.hexdigest()[:8]


def get_baseline_model_id(model_name: str) -> str:
    """
    Get the baseline model ID for Together AI based on model name.

    Args:
        model_name: Model name like 'gpt_oss_120b'

    Returns:
        Together AI model ID for the baseline model
    """
    # Map internal model names to Together AI model IDs
    model_mapping = {
        'openai_gpt_oss_120b': "openai/gpt-oss-120b",#'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',  # Update this with correct model ID
        'google_gemma_3_27b_it': "google/gemma-3-27b-it",
        'google/gemma_3_27b_it': "google/gemma-3-27b-it"
        # Add other model mappings as needed
    }

    if model_name in model_mapping:
        return model_mapping[model_name]

    # If no mapping found, try to use the model name directly
    # You may need to adjust this based on Together AI's model naming
    return model_name


def generate_auroc_charts(
        all_results: Dict[str, Any],
        trained_fold: str,
        output_dir: Path,
        split: str,
        model_name: str,
        include_baseline: bool = False
) -> None:
    """
    Generate AUROC charts showing ROC curves across epochs.

    Args:
        all_results: Dictionary with evaluation results by epoch
        trained_fold: Name of the fold used for training
        output_dir: Directory to save charts
        split: Train or val split being evaluated
        model_name: Name of the model
        include_baseline: Whether to include baseline in charts
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    # Create figure for ROC curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Colors for different epochs
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    # Track AUROC values across epochs for summary plot
    auroc_by_epoch = defaultdict(lambda: defaultdict(float))

    # Process each epoch
    for idx, (epoch_key, epoch_results) in enumerate(all_results.items()):
        if 'skipped' in epoch_results or 'error' in epoch_results:
            continue

        # Extract epoch number
        try:
            if epoch_key == "baseline":
                epoch_num = -1
            else:
                epoch_num = int(epoch_key.split('_')[1])
        except (ValueError, IndexError):
            continue

        color = colors[idx]

        # Plot ROC curve for trained fold
        if trained_fold in epoch_results:
            metrics = epoch_results[trained_fold]
            if 'roc_curve' in metrics and 'auroc' in metrics:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                auroc_val = metrics['auroc']

                auroc_by_epoch[epoch_num][trained_fold] = auroc_val

                # Plot on first subplot (trained fold)
                label = f"Epoch {epoch_num} (AUC={auroc_val:.3f})"
                if epoch_num == -1:
                    label = f"Baseline (AUC={auroc_val:.3f})"

                axes[0].plot(fpr, tpr, color=color, lw=2,
                             label=label, linestyle='--' if epoch_num != -1 else ':')

        # Calculate mean ROC curve for other folds
        other_fprs = []
        other_tprs = []
        other_aurocs = []

        for fold_name, metrics in epoch_results.items():
            if fold_name != trained_fold and 'roc_curve' in metrics:
                fpr = np.array(metrics['roc_curve']['fpr'])
                tpr = np.array(metrics['roc_curve']['tpr'])
                auroc_val = metrics['auroc']

                # Interpolate to common FPR points for averaging
                mean_fpr = np.linspace(0, 1, 100)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0

                other_fprs.append(mean_fpr)
                other_tprs.append(interp_tpr)
                other_aurocs.append(auroc_val)

                auroc_by_epoch[epoch_num][fold_name] = auroc_val

                # Plot individual fold on subplot 2
                axes[2].plot(fpr, tpr, alpha=0.3, color=color, lw=1)

        # Plot mean of other folds
        if other_tprs:
            mean_tpr = np.mean(other_tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auroc = np.mean(other_aurocs)

            label = f"Epoch {epoch_num} (AUC={mean_auroc:.3f})"
            if epoch_num == -1:
                label = f"Baseline (AUC={mean_auroc:.3f})"

            axes[1].plot(mean_fpr, mean_tpr, color=color, lw=2,
                         label=label, linestyle='-' if epoch_num != -1 else ':')

    # Configure subplots
    subplot_titles = [
        f'{trained_fold} (Trained Fold)',
        'Mean of Other Folds',
        'All Other Folds (Individual)',
        'AUROC Across Epochs'
    ]

    for i in range(3):
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(subplot_titles[i])
        axes[i].legend(loc='lower right', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])

    # Plot AUROC progression across epochs (subplot 4)
    epochs_sorted = sorted(auroc_by_epoch.keys())

    # Plot trained fold AUROC progression
    trained_aurocs = [auroc_by_epoch[e].get(trained_fold, np.nan)
                      for e in epochs_sorted]
    axes[3].plot(epochs_sorted, trained_aurocs, 'b--', lw=2,
                 marker='o', label=f'{trained_fold} (trained)', markersize=8)

    # Plot mean of other folds
    other_fold_names = set()
    for epoch_data in auroc_by_epoch.values():
        other_fold_names.update(epoch_data.keys())
    other_fold_names.discard(trained_fold)

    if other_fold_names:
        mean_other_aurocs = []
        for e in epochs_sorted:
            other_vals = [auroc_by_epoch[e].get(f, np.nan)
                          for f in other_fold_names]
            other_vals = [v for v in other_vals if not np.isnan(v)]
            mean_other_aurocs.append(np.mean(other_vals) if other_vals else np.nan)

        axes[3].plot(epochs_sorted, mean_other_aurocs, 'g-', lw=2,
                     marker='s', label='Mean of others', markersize=6)

    axes[3].set_xlabel('Epoch (-1 = baseline)' if include_baseline else 'Epoch')
    axes[3].set_ylabel('AUROC')
    axes[3].set_title(subplot_titles[3])
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([0.5, 1.0])

    # Overall title
    title = f'ROC Curves and AUROC Analysis\n{model_name} - {split} split'
    if include_baseline:
        title += ' (with baseline)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save chart
    suffix = "_with_baseline" if include_baseline else ""
    chart_file = output_dir / f"auroc_analysis_{model_name}_{trained_fold}_{split}{suffix}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"AUROC chart saved: {chart_file}")

    # Create a simple AUROC progression chart
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_sorted, trained_aurocs, 'b--', lw=3,
             marker='o', label=f'{trained_fold} (trained)', markersize=10, alpha=0.8)

    if other_fold_names and mean_other_aurocs:
        plt.plot(epochs_sorted, mean_other_aurocs, 'g-', lw=2.5,
                 marker='s', label='Mean of other folds', markersize=8, alpha=0.8)

        # Add shaded region for std dev
        std_other_aurocs = []
        for e in epochs_sorted:
            other_vals = [auroc_by_epoch[e].get(f, np.nan)
                          for f in other_fold_names]
            other_vals = [v for v in other_vals if not np.isnan(v)]
            std_other_aurocs.append(np.std(other_vals) if other_vals else 0)

        mean_other_aurocs = np.array(mean_other_aurocs)
        std_other_aurocs = np.array(std_other_aurocs)

        plt.fill_between(epochs_sorted,
                         mean_other_aurocs - std_other_aurocs,
                         mean_other_aurocs + std_other_aurocs,
                         color='green', alpha=0.2, label='±1 std dev')

    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')

    if include_baseline and -1 in epochs_sorted:
        plt.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Epoch (-1 = baseline)' if include_baseline else 'Epoch', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.title(f'AUROC Progression - {model_name} ({split} split)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.45, 1.05])

    plt.tight_layout()

    simple_chart_file = output_dir / f"auroc_progression_{model_name}_{trained_fold}_{split}{suffix}.png"
    plt.savefig(simple_chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"AUROC progression chart saved: {simple_chart_file}")

def generate_simplified_performance_chart(
        all_results: Dict[str, Any],
        trained_fold: str,
        output_dir: Path,
        split: str,
        model_name: str,
        include_baseline: bool = False
) -> None:
    """
    Generate simplified line charts showing only trained fold and mean of other folds.

    Args:
        all_results: Dictionary with evaluation results by epoch
        trained_fold: Name of the fold used for training (will be dotted line)
        output_dir: Directory to save charts
        split: Train or val split being evaluated
        model_name: Name of the model
        include_baseline: Whether to include baseline (epoch -1) in charts
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Extract data for plotting
    epochs = []
    trained_fold_metrics = defaultdict(list)  # metric -> [values]
    other_folds_metrics = defaultdict(lambda: defaultdict(list))  # metric -> epoch -> [values from different folds]

    # Collect data from results
    for epoch_key, epoch_results in all_results.items():
        if 'skipped' in epoch_results or not isinstance(epoch_results, dict):
            continue

        # Extract epoch number
        try:
            if epoch_key == "baseline":
                epoch_num = -1
            else:
                epoch_num = int(epoch_key.split('_')[1])
        except (ValueError, IndexError):
            continue

        if epoch_num not in epochs:
            epochs.append(epoch_num)

        # Separate trained fold from others
        for fold_name, metrics in epoch_results.items():
            if 'error' not in metrics and isinstance(metrics, dict):
                for metric in ['accuracy', 'f1', 'precision', 'recall']:
                    if metric in metrics:
                        if fold_name == trained_fold:
                            # Store trained fold metrics
                            trained_fold_metrics[metric].append((epoch_num, metrics[metric]))
                        else:
                            # Store other folds metrics
                            other_folds_metrics[metric][epoch_num].append(metrics[metric])

    if not epochs:
        print("No valid epochs found for charting")
        return

    # Sort epochs
    epochs = sorted(epochs)

    # Prepare data for plotting
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']

    # Create individual metric charts
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        # Plot trained fold line
        if metric in trained_fold_metrics:
            trained_epochs = [e for e, _ in sorted(trained_fold_metrics[metric])]
            trained_values = [v for _, v in sorted(trained_fold_metrics[metric])]

            plt.plot(
                trained_epochs,
                trained_values,
                color='blue',
                linestyle='--',
                linewidth=3,
                marker='o',
                markersize=8,
                label=f"{trained_fold} (trained)",
                alpha=0.8
            )

        # Calculate and plot mean of other folds with error bars
        if metric in other_folds_metrics:
            mean_epochs = []
            mean_values = []
            std_values = []

            for epoch in epochs:
                if epoch in other_folds_metrics[metric] and len(other_folds_metrics[metric][epoch]) > 0:
                    values = other_folds_metrics[metric][epoch]
                    mean_epochs.append(epoch)
                    mean_values.append(np.mean(values))
                    std_values.append(np.std(values))

            if mean_epochs:
                mean_values = np.array(mean_values)
                std_values = np.array(std_values)

                # Plot mean line
                plt.plot(
                    mean_epochs,
                    mean_values,
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    marker='s',
                    markersize=6,
                    label='Mean of other folds',
                    alpha=0.8
                )

                # Add error bars (shaded region)
                plt.fill_between(
                    mean_epochs,
                    mean_values - std_values,
                    mean_values + std_values,
                    color='green',
                    alpha=0.2,
                    label='±1 std dev'
                )

        # Add vertical line at epoch -1 if baseline included
        if include_baseline and -1 in epochs:
            plt.axvline(x=-1, color='gray', linestyle=':', alpha=0.5, label='Baseline')

        # Customize the chart
        title = f'{metric.title()} - Trained vs Others Mean\n{model_name} - {split} split'
        if include_baseline:
            title += ' (with baseline)'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epoch (-1 = baseline)' if include_baseline else 'Epoch', fontsize=12)
        plt.ylabel(f'{metric.title()}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        # Set y-axis limits
        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)

        plt.tight_layout()

        # Save chart
        suffix = "_simplified_with_baseline" if include_baseline else "_simplified"
        chart_file = output_dir / f"chart_{metric}_{model_name}_{trained_fold}_{split}{suffix}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Simplified chart saved: {chart_file}")

    # Create combined simplified chart
    plt.figure(figsize=(14, 10))

    for idx, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, idx)

        # Plot trained fold
        if metric in trained_fold_metrics:
            trained_epochs = [e for e, _ in sorted(trained_fold_metrics[metric])]
            trained_values = [v for _, v in sorted(trained_fold_metrics[metric])]

            plt.plot(
                trained_epochs,
                trained_values,
                color='blue',
                linestyle='--',
                linewidth=2.5,
                marker='o',
                markersize=5,
                label=f"{trained_fold} (trained)" if idx == 1 else "",
                alpha=0.8
            )

        # Plot mean of others with error bars
        if metric in other_folds_metrics:
            mean_epochs = []
            mean_values = []
            std_values = []

            for epoch in epochs:
                if epoch in other_folds_metrics[metric] and len(other_folds_metrics[metric][epoch]) > 0:
                    values = other_folds_metrics[metric][epoch]
                    mean_epochs.append(epoch)
                    mean_values.append(np.mean(values))
                    std_values.append(np.std(values))

            if mean_epochs:
                mean_values = np.array(mean_values)
                std_values = np.array(std_values)

                plt.plot(
                    mean_epochs,
                    mean_values,
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    marker='s',
                    markersize=4,
                    label='Mean of others' if idx == 1 else "",
                    alpha=0.8
                )

                plt.fill_between(
                    mean_epochs,
                    mean_values - std_values,
                    mean_values + std_values,
                    color='green',
                    alpha=0.15
                )

        # Add baseline marker if needed
        if include_baseline and -1 in epochs:
            plt.axvline(x=-1, color='gray', linestyle=':', alpha=0.4)

        plt.title(f'{metric.title()}', fontsize=11, fontweight='bold')
        plt.xlabel('Epoch (-1 = baseline)' if include_baseline else 'Epoch', fontsize=10)
        plt.ylabel(f'{metric.title()}', fontsize=10)
        plt.grid(True, alpha=0.3)

        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)

        # Add legend only to first subplot
        if idx == 1:
            plt.legend(loc='best', fontsize=9)

    title = f'Simplified Performance Metrics - Trained vs Others Mean\n{model_name} - {split} split'
    if include_baseline:
        title += ' (with baseline)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save combined chart
    suffix = "_combined_simplified_with_baseline" if include_baseline else "_combined_simplified"
    combined_chart_file = output_dir / f"chart_combined_{model_name}_{trained_fold}_{split}{suffix}.png"
    plt.savefig(combined_chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Simplified combined chart saved: {combined_chart_file}")

def generate_performance_charts_with_baseline(
        all_results: Dict[str, Any],
        trained_fold: str,
        output_dir: Path,
        split: str,
        model_name: str,
        include_baseline: bool = False
) -> None:
    """
    Generate line charts showing performance across epochs, optionally including baseline.

    Args:
        all_results: Dictionary with evaluation results by epoch
        trained_fold: Name of the fold used for training (will be dotted line)
        output_dir: Directory to save charts
        split: Train or val split being evaluated
        model_name: Name of the model
        include_baseline: Whether to include baseline (epoch 0) in charts
    """
    # Extract data for plotting
    epochs = []
    fold_data = defaultdict(lambda: defaultdict(list))  # fold_name -> metric -> [values]

    # Collect data from results
    for epoch_key, epoch_results in all_results.items():
        if 'skipped' in epoch_results or 'error' in epoch_results or not isinstance(epoch_results, dict) or not epoch_results:
            continue

        # Extract epoch number from key like "epoch_1" or "baseline"
        try:
            if epoch_key == "baseline":
                epoch_num = -1
            else:
                epoch_num = int(epoch_key.split('_')[1])
        except (ValueError, IndexError):
            continue

        epochs.append(epoch_num)

        for fold_name, metrics in epoch_results.items():
            if 'error' not in metrics and isinstance(metrics, dict):
                # Store metrics for this fold
                for metric in ['accuracy', 'f1', 'precision', 'recall']:
                    if metric in metrics:
                        fold_data[fold_name][metric].append(metrics[metric])

    if not epochs:
        print("No valid epochs found for charting")
        return

    # Sort epochs to ensure proper x-axis ordering
    sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in sorted_indices]

    # Sort fold data by epochs
    for fold_name in fold_data:
        for metric in fold_data[fold_name]:
            try:
                fold_data[fold_name][metric] = [fold_data[fold_name][metric][i] for i in sorted_indices]
            except IndexError:
                pass

    # Create charts for each metric
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))

        # Plot lines for each fold
        legend_elements = []
        colors = plt.cm.tab10(range(len(fold_data)))

        for i, (fold_name, metrics_dict) in enumerate(fold_data.items()):
            if metric in metrics_dict and len(metrics_dict[metric]) > 0:
                color = colors[i]

                # Use dotted line for the trained fold, solid for others
                if fold_name == trained_fold:
                    linestyle = '--'
                    linewidth = 3
                    alpha = 0.8
                    label = f"{fold_name} (trained)"
                else:
                    linestyle = '-'
                    linewidth = 2
                    alpha = 0.7
                    label = fold_name

                plt.plot(
                    epochs,
                    metrics_dict[metric],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    marker='o',
                    markersize=6,
                    label=label
                )

        # Add vertical line at epoch 0 if baseline included
        if include_baseline and 0 in epochs:
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Baseline (pre-training)')

        # Customize the chart
        title = f'{metric.title()} Across Training Epochs\n{model_name} - {split} split'
        if include_baseline:
            title += ' (with baseline)'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoch (0 = baseline)', fontsize=14)
        plt.ylabel(f'{metric.title()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set y-axis limits for better visualization
        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)

        # Improve layout and save
        plt.tight_layout()

        # Save chart
        suffix = "_with_baseline" if include_baseline else ""
        chart_file = output_dir / f"chart_{metric}_{model_name}_{trained_fold}_{split}{suffix}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Chart saved: {chart_file}")

    # Create a combined chart with all metrics
    plt.figure(figsize=(16, 10))

    # Create subplots for each metric
    for idx, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, idx)

        colors = plt.cm.tab10(range(len(fold_data)))

        for i, (fold_name, metrics_dict) in enumerate(fold_data.items()):
            if metric in metrics_dict and len(metrics_dict[metric]) > 0:
                color = colors[i]

                if fold_name == trained_fold:
                    linestyle = '--'
                    linewidth = 3
                    alpha = 0.8
                    label = f"{fold_name} (trained)"
                else:
                    linestyle = '-'
                    linewidth = 2
                    alpha = 0.7
                    label = fold_name

                plt.plot(
                    epochs,
                    metrics_dict[metric],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    marker='o',
                    markersize=4,
                    label=label if idx == 1 else ""  # Only show legend on first subplot
                )

        # Add vertical line at epoch 0 if baseline included
        if include_baseline and 0 in epochs:
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

        plt.title(f'{metric.title()}', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch (0 = baseline)' if include_baseline else 'Epoch')
        plt.ylabel(f'{metric.title()}')
        plt.grid(True, alpha=0.3)

        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)

        # Add legend only to the first subplot
        if idx == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    title = f'Performance Metrics Across Training Epochs\n{model_name} - {split} split'
    if include_baseline:
        title += ' (with baseline)'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save combined chart
    suffix = "_with_baseline" if include_baseline else ""
    combined_chart_file = output_dir / f"chart_combined_{model_name}_{trained_fold}_{split}{suffix}.png"
    plt.savefig(combined_chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined chart saved: {combined_chart_file}")


async def run_baseline_evaluation(
        base_path: Path,
        folder: str,
        fold_name: str,
        model_name: str,
        eval_folds: List,
        limit: Optional[int] = None,
        force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Run baseline evaluation on the non-finetuned model with caching support.

    Args:
        base_path: Base directory path
        fold_name: Training fold name
        model_name: Model name
        eval_folds: List of evaluation folds
        limit: Optional limit on samples
        force_recompute: Whether to force recomputation instead of using cache

    Returns:
        Dictionary with baseline evaluation results
    """
    from epoch_eval import (
        load_jsonl_samples,
        prepare_eval_sample,
        load_baseline_results,
        save_baseline_results
    )

    print(f"\n{'=' * 60}")
    print(f"RUNNING BASELINE EVALUATION")
    print(f"{'=' * 60}")

    # Check for cached results first (unless forced to recompute)
    if not force_recompute:
        print("Checking for cached baseline results...")
        cached_results = load_baseline_results(base_path, folder, fold_name, model_name)
        if cached_results is not None:
            print("✓ Found cached baseline results, skipping computation")
            return cached_results
        else:
            print("No cached results found, will compute baseline evaluation")

    # Get baseline model ID
    baseline_model_id = get_baseline_model_id(model_name)
    print(f"Baseline model ID: {baseline_model_id}")

    baseline_results = {}

    for eval_fold in eval_folds:
        print(f"\n  Evaluating baseline on fold: {eval_fold.name}")

        # Always use train split for consistency
        split = "train"
        samples_path = str(eval_fold.train_path).replace("train.jsonl", "_train.jsonl")

        try:
            raw_samples = load_jsonl_samples(Path(samples_path))
            print(f"    Loaded {len(raw_samples)} raw samples")

            # Prepare samples
            eval_samples = []
            for raw_sample in raw_samples:
                try:
                    sample = prepare_eval_sample(raw_sample)
                    eval_samples.append(sample)
                except Exception as e:
                    print(f"    Warning: Failed to prepare sample: {e}")

            print(f"    Prepared {len(eval_samples)} evaluation samples")

            # Run evaluation
            metrics = await evaluate_model_on_fold_single(
                model_ref=baseline_model_id,
                eval_samples=eval_samples,
                fold_name=eval_fold.name,
                epoch=0,  # Use epoch 0 for baseline
                trained_fold=fold_name,
                limit=limit,
                is_endpoint=False,
                is_baseline=True,
                run_id=None  # No run_id available for baseline evaluation
            )

            # Store results
            baseline_results[eval_fold.name] = metrics

            # Print metrics
            if 'error' not in metrics:
                print(f"    Baseline Results:")
                print(f"      - Accuracy:  {metrics['accuracy']:.3f}")
                print(f"      - F1 Score:  {metrics['f1']:.3f}")
                print(f"      - Precision: {metrics['precision']:.3f}")
                print(f"      - Recall:    {metrics['recall']:.3f}")
                print(f"      - Samples:   {metrics['num_samples']}")
            else:
                print(f"    Error: {metrics['error']}")

        except Exception as e:
            print(f"    Error loading/evaluating: {e}")
            baseline_results[eval_fold.name] = {'error': str(e)}

    # Save results to cache if computation was successful
    if baseline_results:
        # Check if we have any successful results (no errors)
        successful_results = {k: v for k, v in baseline_results.items() if 'error' not in v}
        if successful_results:
            print(f"\n💾 Caching baseline results for future use...")
            save_baseline_results(baseline_results, base_path, folder, fold_name, model_name)

    return baseline_results


async def main():
    """Main evaluation function with baseline support."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate lie detection models")
    parser.add_argument(
        "--base-path",
        type=str,
        default="../../",
        help="Base directory path (e.g., /Users/jackhopkins/PycharmProjects/lie-detector)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=".together-120b",
        help="Base directory path (e.g., /Users/jackhopkins/PycharmProjects/lie-detector)"
    )
    parser.add_argument(
        "--fold-name",
        type=str,
        default="sandbagging_other",
        #default="games",
        help="Training fold name (e.g., sandbagging_ascii)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt_oss_120b",
        help="Model name (e.g., gpt_oss_120b)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline evaluation on non-finetuned model"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run ONLY baseline evaluation (skip epoch evaluations)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Limit number of samples per evaluation (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: base_path/eval_results)"
    )
    parser.add_argument(
        "--force-recompute-baseline",
        action="store_true",
        help="Force recomputation of baseline results instead of using cache"
    )
    
    # Run management parameters
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Use specific run ID (for resume or recompute)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect and resume latest incomplete run"
    )
    parser.add_argument(
        "--log-base-dir",
        type=str,
        default="../../logs",
        help="Base directory for logs (default: ../../logs)"
    )
    parser.add_argument(
        "--force-recompute",
        type=str,
        default=None,
        help="Force recompute specific epochs (comma-separated, e.g., 'epoch_0,epoch_2,baseline')"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available runs and their status, then exit"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookup and force recomputation of all evaluations"
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached results, skip any evaluations that aren't cached"
    )
    parser.add_argument(
        "--show-cache-stats",
        action="store_true",
        help="Display cache performance statistics"
    )

    args = parser.parse_args()
    #args.baseline = False
    #args.baseline_only = True
    args.render_only = None#'/Users/jackhopkins/PycharmProjects/lie-detector/eval_results/sandbagging_other/lie_detection_sandbagging_other_gpt_oss_120b_train.json'
    args.force_recompute = None # "epoch_4"
    # Import run state manager and cache manager
    from run_state_manager import RunStateManager
    from log_cache_manager import LogCacheManager

    # Initialize run state manager
    run_manager = RunStateManager(args.log_base_dir)
    
    # Initialize cache manager (unless caching is disabled)
    cache_manager = None if args.no_cache else LogCacheManager(args.log_base_dir)
    
    if cache_manager and not args.no_cache:
        print(f"🗂️  Cache manager initialized (base dir: {args.log_base_dir})")
    elif args.no_cache:
        print(f"🚫 Caching disabled by --no-cache flag")
    
    # Handle list-runs command
    if args.list_runs:
        print(f"\n{'=' * 60}")
        print(f"AVAILABLE EVALUATION RUNS")
        print(f"{'=' * 60}")
        
        runs = run_manager.list_runs()
        if not runs:
            print("No evaluation runs found.")
            return
            
        for run_id, run_state in runs:
            summary = run_manager.get_run_summary(run_state)
            status = "✅ Complete" if summary['is_complete'] else f"⏳ {summary['completion_percentage']:.1f}% complete"
            print(f"\n{run_id}")
            print(f"  Fold: {summary['trained_fold']} | Model: {summary['model_name']}")
            print(f"  Created: {summary['created_time']}")
            print(f"  Status: {status} ({summary['completed_combinations']}/{summary['total_eval_combinations']} combinations)")
            if summary['failed_epochs'] > 0:
                print(f"  Failed epochs: {summary['failed_epochs']}")
        return


    if args.render_only:
        print(f"\n{'=' * 60}")
        print(f"RENDER-ONLY MODE")
        print(f"{'=' * 60}")
        print(f"Loading existing results from: {args.render_only}")

        # Load existing results
        with open(args.render_only, 'r') as f:
            all_results = json.load(f)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.render_only).parent

        # Generate charts
        print(f"\n{'=' * 60}")
        print(f"GENERATING CHARTS")
        print(f"{'=' * 60}")

        split = "train"  # Default to train split
        include_baseline = "baseline" in all_results

        # Generate full charts with all folds
        try:
            print("Generating full performance charts...")
            generate_performance_charts_with_baseline(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: Full chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            print("\nGenerating AUROC charts...")
            generate_auroc_charts(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: AUROC chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate simplified charts (trained vs mean of others)
        try:
            print("\nGenerating simplified performance charts...")
            generate_simplified_performance_chart(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: Simplified chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("Chart generation complete!")
        return  # Exit early, skip all evaluation logic

    # Import helper functions from the main module
    from epoch_eval import (
        find_trained_models,
        find_eval_folds
    )
    from endpoint_manager import EndpointManager

    base_path = Path(args.base_path)

    # Step 1: Find evaluation folds (needed for both baseline and epoch eval)
    print(f"\n{'=' * 60}")
    print(f"FINDING EVALUATION FOLDS")
    print(f"{'=' * 60}")

    try:
        eval_folds = find_eval_folds(base_path, args.folder, args.model_name)
        print(f"Found {len(eval_folds)} evaluation folds:")
        for fold in eval_folds:
            print(f"  - {fold.name}")
    except Exception as e:
        print(f"Error finding evaluation folds: {e}")
        sys.exit(1)

    if not eval_folds:
        print("No evaluation folds found!")
        sys.exit(1)

    all_results = {}
    include_baseline = args.baseline or args.baseline_only

    # Step 2: Run baseline evaluation if requested
    if include_baseline:
        baseline_results = await run_baseline_evaluation(
            base_path=base_path,
            folder=args.folder,
            fold_name=args.fold_name,
            model_name=args.model_name,
            eval_folds=eval_folds,
            limit=args.limit,
            force_recompute=args.force_recompute_baseline
        )
        all_results["baseline"] = baseline_results

    # Step 3: Run epoch evaluations (unless baseline-only mode)
    if not args.baseline_only:
        # Find trained models
        print(f"\n{'=' * 60}")
        print(f"FINDING TRAINED MODELS")
        print(f"{'=' * 60}")
        print(f"Fold: {args.fold_name}")
        print(f"Model: {args.model_name}")

        try:
            trained_models = find_trained_models(base_path, args.folder, args.fold_name, args.model_name)
            print(f"Found {len(trained_models)} completed models:")
            for model in trained_models:
                print(f"  - Epoch {model.epoch}: {model.model_id}")
        except Exception as e:
            print(f"Error finding trained models: {e}")
            if not include_baseline:
                sys.exit(1)
            trained_models = []

        if trained_models:
            # Generate run_id from all models if not provided
            if not args.run_id:
                baseline_model_id = None
                if include_baseline:
                    baseline_model_id = get_baseline_model_id(args.model_name)
                
                args.run_id = generate_run_id_from_models(trained_models, baseline_model_id)
                print(f"🔗 Generated run ID from {len(trained_models)} models: {args.run_id}")
            else:
                print(f"🔗 Using provided run ID: {args.run_id}")

            # Initialize endpoint manager and batch deploy all endpoints
            print(f"\n{'=' * 60}")
            print(f"ENDPOINT SETUP")
            print(f"{'=' * 60}")

            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                print("Warning: TOGETHER_API_KEY not found. Will skip models without cached endpoints.")
                endpoint_manager = None
                batch_deploy_result = None
            else:
                endpoint_manager = EndpointManager(api_key)
                print("Endpoint manager initialized")
                
                # Define training fold path
                training_fold_path = base_path / args.folder / args.model_name / args.fold_name
                
                # Batch deploy ALL endpoints for this fold with 2-hour timeout
                print(f"\n🚀 BATCH DEPLOYING ALL ENDPOINTS FOR {args.fold_name}")
                batch_deploy_result = endpoint_manager.batch_deploy_all_epochs(
                    fold_path=str(training_fold_path),
                    fold_name=args.fold_name,
                    inactive_timeout=120,  # 2 hours
                    max_concurrent=5,
                    wait_for_ready=True
                )
                
                if batch_deploy_result["status"] == "success":
                    print(f"✅ All {batch_deploy_result['ready_endpoints']} endpoints deployed successfully!")
                elif batch_deploy_result["status"] == "partial":
                    print(f"⚠️  Partial deployment: {batch_deploy_result['ready_endpoints']}/{batch_deploy_result['total_epochs']} endpoints ready")
                    if batch_deploy_result["failed_epochs"]:
                        print(f"❌ Failed epochs: {batch_deploy_result['failed_epochs']}")
                else:
                    print(f"❌ Batch deployment failed: {batch_deploy_result.get('reason', 'unknown')}")
                    if not batch_deploy_result.get("endpoints"):
                        print("No endpoints available for evaluation!")
                        endpoint_manager = None

            # Run evaluations
            print(f"\n{'=' * 60}")
            print(f"RUNNING EPOCH EVALUATIONS")
            print(f"{'=' * 60}")

            if args.limit:
                print(f"Limiting to {args.limit} samples per evaluation")

            training_fold_path = base_path / args.folder / args.model_name / args.fold_name

            for model_info in trained_models:
                print(f"\n--- Model Epoch {model_info.epoch} ---")
                print(f"Model ID: {model_info.model_id}")

                epoch_results = {}
                endpoint_name = None
                is_endpoint = False

                # Set up endpoint if endpoint manager is available
                if endpoint_manager:
                    try:
                        print(f"Looking for endpoint for model evaluation...")
                        endpoint_name = endpoint_manager.get_or_find_endpoint(
                            fold_path=str(training_fold_path),
                            epoch=model_info.epoch,
                            model_id=model_info.model_id,
                            fold_name=args.fold_name
                        )
                        if endpoint_name:
                            print(f"Using endpoint: {endpoint_name}")
                            is_endpoint = True
                        else:
                            print(f"No endpoint found for model {model_info.model_id}")
                            print("Skipping evaluation for this model (no active endpoint)")
                            all_results[f"epoch_{model_info.epoch}"] = {
                                "error": "No active endpoint found",
                                "model_id": model_info.model_id,
                                "skipped": True
                            }
                            continue
                    except Exception as e:
                        print(f"Failed to find endpoint: {e}")
                        print("Skipping evaluation for this model (endpoint discovery failed)")
                        all_results[f"epoch_{model_info.epoch}"] = {
                            "error": f"Endpoint discovery failed: {e}",
                            "model_id": model_info.model_id,
                            "skipped": True
                        }
                        continue
                else:
                    print("No endpoint manager available, skipping evaluation")
                    all_results[f"epoch_{model_info.epoch}"] = {
                        "error": "No endpoint manager (TOGETHER_API_KEY not provided)",
                        "model_id": model_info.model_id,
                        "skipped": True
                    }
                    continue

                # Run parallel evaluation on all folds for this epoch
                print(f"\n🚀 Running parallel evaluation on all folds for epoch {model_info.epoch}")
                epoch_results = await evaluate_model_on_all_folds_parallel(
                    model_ref=endpoint_name,
                    eval_folds=eval_folds,
                    epoch=model_info.epoch,
                    trained_fold=args.fold_name,
                    limit=args.limit,
                    is_endpoint=is_endpoint,
                    is_baseline=False,
                    cache_manager=cache_manager,
                    cache_only=args.cache_only,
                    run_id=args.run_id
                )

                all_results[f"epoch_{model_info.epoch}"] = epoch_results

            # Clean up endpoints if needed
            if endpoint_manager:
                print(f"\n{'=' * 60}")
                print(f"CLEANING UP ENDPOINTS")
                print(f"{'=' * 60}")

                try:
                    cleanup_results = endpoint_manager.cleanup_expired_endpoints(str(training_fold_path))
                    if cleanup_results["total_expired"] > 0:
                        print(f"Cleaned up {len(cleanup_results['cleaned_up'])} expired endpoints")
                        if cleanup_results["failed_cleanup"]:
                            print(f"Failed to clean up {len(cleanup_results['failed_cleanup'])} endpoints")
                    else:
                        print("No expired endpoints to clean up")
                except Exception as e:
                    print(f"Warning: Endpoint cleanup failed: {e}")

    # Step 4: Save results
    print(f"\n{'=' * 60}")
    print(f"SAVING RESULTS")
    print(f"{'=' * 60}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "eval_results" / args.fold_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    split = "train"  # Always using train for consistency
    suffix = "_baseline" if args.baseline_only else ("_with_baseline" if include_baseline else "")
    output_file = output_dir / f"lie_detection_{args.fold_name}_{args.model_name}_{split}{suffix}.json"

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Step 5: Print summary
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    for epoch_key, epoch_results in all_results.items():
        if epoch_key == "baseline":
            print(f"\nBaseline (non-finetuned):")
        else:
            print(f"\n{epoch_key}:")

        # Skip if this epoch was skipped
        if isinstance(epoch_results, dict) and 'skipped' in epoch_results:
            print(f"  SKIPPED: {epoch_results.get('error', 'Unknown error')}")
            continue

        # Compute average metrics across folds
        avg_metrics = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

        for fold_name, metrics in epoch_results.items():
            if 'error' not in metrics:
                try:
                    print(f"  {fold_name:25} - F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}, "
                          f"Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}")

                    avg_metrics['accuracy'].append(metrics['accuracy'])
                    avg_metrics['f1'].append(metrics['f1'])
                    avg_metrics['precision'].append(metrics['precision'])
                    avg_metrics['recall'].append(metrics['recall'])
                except Exception as e:
                    pass
            else:
                print(f"  {fold_name:25} - Error: {metrics['error'][:50]}...")

        # Print averages if we have valid results
        if avg_metrics['accuracy']:
            print(f"\n  Averages across {len(avg_metrics['accuracy'])} folds:")
            print(f"    - Accuracy:  {sum(avg_metrics['accuracy']) / len(avg_metrics['accuracy']):.3f}")
            print(f"    - F1 Score:  {sum(avg_metrics['f1']) / len(avg_metrics['f1']):.3f}")
            print(f"    - Precision: {sum(avg_metrics['precision']) / len(avg_metrics['precision']):.3f}")
            print(f"    - Recall:    {sum(avg_metrics['recall']) / len(avg_metrics['recall']):.3f}")

    # Step 6: Generate performance charts
    if not args.baseline_only and len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"GENERATING CHARTS")
        print(f"{'=' * 60}")

        try:
            print("\nGenerating AUROC charts...")
            generate_auroc_charts(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: AUROC chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate full charts with all folds
        try:
            print("Generating full performance charts...")
            generate_performance_charts_with_baseline(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: Full chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate simplified charts (trained vs mean of others)
        try:
            print("\nGenerating simplified performance charts...")
            generate_simplified_performance_chart(
                all_results=all_results,
                trained_fold=args.fold_name,
                output_dir=output_dir,
                split=split,
                model_name=args.model_name,
                include_baseline=include_baseline
            )
        except Exception as e:
            print(f"Warning: Simplified chart generation failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())