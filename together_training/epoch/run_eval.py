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
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Import Inspect AI components
from inspect_ai import Task, eval, eval_async
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem
from inspect_ai.scorer import Score, accuracy
from inspect_ai.solver import TaskState, Generate


async def evaluate_model_on_fold(
        model_ref: str,
        eval_samples: List[Sample],
        fold_name: str,
        epoch: int,
        trained_fold: str,
        limit: Optional[int] = None,
        is_endpoint: bool = False,
        is_baseline: bool = False
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
    from together_training.epoch.epoch_eval import (
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
        scorer=binary_classification_scorer(),
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

        log_dir_suffix = "baseline" if is_baseline else f"{fold_name}"
        log = await eval_async(
            task,
            model=model_string,
            temperature=0.0,  # Deterministic for evaluation
            log_dir=f"../../logs/eval-18-9-6/{trained_fold}/{log_dir_suffix}",
        )

        # Extract scores
        scores = []
        for sample in log[0].samples:
            if hasattr(sample, 'scores') and 'binary_classification_scorer' in sample.scores:
                scores.append(sample.scores['binary_classification_scorer'])

        # Compute metrics
        metrics = compute_metrics(scores)
        metrics['model_ref'] = model_ref
        metrics['is_endpoint'] = is_endpoint
        metrics['is_baseline'] = is_baseline
        metrics['fold_name'] = fold_name
        metrics['epoch'] = epoch

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
        'gpt_oss_120b': "openai/gpt-oss-120b"#'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',  # Update this with correct model ID
        # Add other model mappings as needed
    }

    if model_name in model_mapping:
        return model_mapping[model_name]

    # If no mapping found, try to use the model name directly
    # You may need to adjust this based on Together AI's model naming
    return model_name


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
        if 'skipped' in epoch_results or not isinstance(epoch_results, dict):
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
            fold_data[fold_name][metric] = [fold_data[fold_name][metric][i] for i in sorted_indices]

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
        fold_name: str,
        model_name: str,
        eval_folds: List,
        limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run baseline evaluation on the non-finetuned model.

    Args:
        base_path: Base directory path
        fold_name: Training fold name
        model_name: Model name
        eval_folds: List of evaluation folds
        limit: Optional limit on samples

    Returns:
        Dictionary with baseline evaluation results
    """
    from together_training.epoch.epoch_eval import (
        load_jsonl_samples,
        prepare_eval_sample
    )

    print(f"\n{'=' * 60}")
    print(f"RUNNING BASELINE EVALUATION")
    print(f"{'=' * 60}")

    # Get baseline model ID
    baseline_model_id = get_baseline_model_id(model_name)
    print(f"Baseline model ID: {baseline_model_id}")

    baseline_results = {}

    for eval_fold in eval_folds:
        print(f"\n  Evaluating baseline on fold: {eval_fold.name}")

        # Always use train split for consistency
        split = "train"
        samples_path = eval_fold.train_path

        try:
            raw_samples = load_jsonl_samples(samples_path)
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
            metrics = await evaluate_model_on_fold(
                model_ref=baseline_model_id,
                eval_samples=eval_samples,
                fold_name=eval_fold.name,
                epoch=0,  # Use epoch 0 for baseline
                trained_fold=fold_name,
                limit=limit,
                is_endpoint=False,
                is_baseline=True
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
        "--fold-name",
        type=str,
        default="games",
        #default="games",
        help="Training fold name (e.g., sandbagging_ascii)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt_oss_120b",
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
        default=200,
        help="Limit number of samples per evaluation (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: base_path/eval_results)"
    )

    args = parser.parse_args()
    args.baseline = True
    args.render_only = None#'/Users/jackhopkins/PycharmProjects/lie-detector/eval_results/mask-factual/lie_detection_mask-factual_gpt_oss_120b_train_with_baseline.json'
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
    from together_training.epoch.epoch_eval import (
        find_trained_models,
        find_eval_folds,
        load_jsonl_samples,
        prepare_eval_sample
    )
    from together_training.epoch.endpoint_manager import EndpointManager

    base_path = Path(args.base_path)

    # Step 1: Find evaluation folds (needed for both baseline and epoch eval)
    print(f"\n{'=' * 60}")
    print(f"FINDING EVALUATION FOLDS")
    print(f"{'=' * 60}")

    try:
        eval_folds = find_eval_folds(base_path, args.model_name)
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
            fold_name=args.fold_name,
            model_name=args.model_name,
            eval_folds=eval_folds,
            limit=args.limit
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
            trained_models = find_trained_models(base_path, args.fold_name, args.model_name)
            print(f"Found {len(trained_models)} completed models:")
            for model in trained_models:
                print(f"  - Epoch {model.epoch}: {model.model_id}")
        except Exception as e:
            print(f"Error finding trained models: {e}")
            if not include_baseline:
                sys.exit(1)
            trained_models = []

        if trained_models:
            # Initialize endpoint manager
            print(f"\n{'=' * 60}")
            print(f"ENDPOINT SETUP")
            print(f"{'=' * 60}")

            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                print("Warning: TOGETHER_API_KEY not found. Will skip models without cached endpoints.")
                endpoint_manager = None
            else:
                endpoint_manager = EndpointManager(api_key)
                print("Endpoint manager initialized")

            # Run evaluations
            print(f"\n{'=' * 60}")
            print(f"RUNNING EPOCH EVALUATIONS")
            print(f"{'=' * 60}")

            if args.limit:
                print(f"Limiting to {args.limit} samples per evaluation")

            training_fold_path = base_path / ".together-120b" / "openai" / args.model_name / args.fold_name

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

                for eval_fold in eval_folds:
                    print(f"\n  Evaluating on fold: {eval_fold.name}")

                    split = "train"
                    print(f"Using {split} split for evaluation")

                    samples_path = eval_fold.train_path

                    try:
                        raw_samples = load_jsonl_samples(samples_path)
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
                        metrics = await evaluate_model_on_fold(
                            model_ref=endpoint_name,
                            eval_samples=eval_samples,
                            fold_name=eval_fold.name,
                            epoch=model_info.epoch,
                            limit=args.limit,
                            is_endpoint=is_endpoint,
                            trained_fold=args.fold_name
                        )

                        # Store results
                        epoch_results[eval_fold.name] = metrics

                        # Print metrics
                        if 'error' not in metrics:
                            print(f"    Results:")
                            print(f"      - Accuracy:  {metrics['accuracy']:.3f}")
                            print(f"      - F1 Score:  {metrics['f1']:.3f}")
                            print(f"      - Precision: {metrics['precision']:.3f}")
                            print(f"      - Recall:    {metrics['recall']:.3f}")
                            print(f"      - Samples:   {metrics['num_samples']}")
                        else:
                            print(f"    Error: {metrics['error']}")

                    except Exception as e:
                        print(f"    Error loading/evaluating: {e}")
                        epoch_results[eval_fold.name] = {'error': str(e)}

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
                print(f"  {fold_name:25} - F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}, "
                      f"Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}")

                avg_metrics['accuracy'].append(metrics['accuracy'])
                avg_metrics['f1'].append(metrics['f1'])
                avg_metrics['precision'].append(metrics['precision'])
                avg_metrics['recall'].append(metrics['recall'])
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