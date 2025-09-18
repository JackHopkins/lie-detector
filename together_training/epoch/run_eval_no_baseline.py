#!/usr/bin/env python3
"""
Script to run lie detection evaluation using the Inspect framework.

Usage:
    python run_lie_detection_eval.py \
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
        is_endpoint: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single model on a single fold.

    Args:
        model_ref: Together AI model ID or endpoint name
        eval_samples: List of evaluation samples
        fold_name: Name of the evaluation fold
        epoch: Model epoch number
        limit: Optional limit on number of samples
        is_endpoint: Whether model_ref is an endpoint name

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

    # Create task
    task = Task(
        dataset=dataset,
        solver=lie_detection_solver(),
        scorer=binary_classification_scorer(),
        name=f"eval_{fold_name}_epoch{epoch}"
    )

    # Run evaluation
    print(f"    Running evaluation on {len(dataset.samples)} samples...")

    try:
        # Use Together AI endpoint or model ID
        model_string = f"together/{model_ref}"
        ref_type = "endpoint" if is_endpoint else "model ID"
        print(f"    Using {ref_type}: {model_ref}")
        
        log = await eval_async(
            task,
            model=model_string,
            temperature=0.0,  # Deterministic for evaluation
            log_dir=f"../../logs/eval-18-9-1/{trained_fold}/{fold_name}",  # Specify log directory
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
        metrics['fold_name'] = fold_name
        metrics['epoch'] = epoch

        return metrics

    except Exception as e:
        print(f"    Error during evaluation: {e}")
        return {
            'error': str(e),
            'model_ref': model_ref,
            'is_endpoint': is_endpoint,
            'fold_name': fold_name,
            'epoch': epoch
        }


def generate_performance_charts(
    all_results: Dict[str, Any], 
    trained_fold: str, 
    output_dir: Path,
    split: str,
    model_name: str
) -> None:
    """
    Generate line charts showing performance across epochs.
    
    Args:
        all_results: Dictionary with evaluation results by epoch
        trained_fold: Name of the fold used for training (will be dotted line)
        output_dir: Directory to save charts
        split: Train or val split being evaluated
        model_name: Name of the model
    """
    # Extract data for plotting
    epochs = []
    fold_data = defaultdict(lambda: defaultdict(list))  # fold_name -> metric -> [values]
    
    # Collect data from results
    for epoch_key, epoch_results in all_results.items():
        if 'skipped' in epoch_results or not isinstance(epoch_results, dict):
            continue
            
        # Extract epoch number from key like "epoch_1"
        try:
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
        
        # Customize the chart
        plt.title(f'{metric.title()} Across Training Epochs\n{model_name} - {split} split', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(f'{metric.title()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set y-axis limits for better visualization
        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)
        
        # Improve layout and save
        plt.tight_layout()
        
        # Save chart
        chart_file = output_dir / f"chart_{metric}_{model_name}_{trained_fold}_{split}.png"
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
        
        plt.title(f'{metric.title()}', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.title()}')
        plt.grid(True, alpha=0.3)
        
        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            plt.ylim(0, 1.05)
        
        # Add legend only to the first subplot
        if idx == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'Performance Metrics Across Training Epochs\n{model_name} - {split} split', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save combined chart
    combined_chart_file = output_dir / f"chart_combined_{model_name}_{trained_fold}_{split}.png"
    plt.savefig(combined_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined chart saved: {combined_chart_file}")


async def main():
    """Main evaluation function."""
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
        default="cot-unfaithfulness",
        help="Training fold name (e.g., sandbagging_ascii)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt_oss_120b",
        help="Model name (e.g., gpt_oss_120b)"
    )
    # parser.add_argument(
    #     "--use-train",
    #     action="store_true",
    #     help="Evaluate on train split instead of val"
    # )
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

    # Import helper functions from the main module
    from together_training.epoch.epoch_eval import (
        find_trained_models,
        find_eval_folds,
        load_jsonl_samples,
        prepare_eval_sample
    )
    from together_training.epoch.endpoint_manager import EndpointManager

    base_path = Path(args.base_path)

    # Step 1: Find trained models
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
        sys.exit(1)

    if not trained_models:
        print("No completed models found!")
        sys.exit(1)

    # Step 2: Find evaluation folds
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

    # Step 3: Initialize endpoint manager
    print(f"\n{'=' * 60}")
    print(f"ENDPOINT SETUP")
    print(f"{'=' * 60}")
    
    # Get API key from environment
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("Warning: TOGETHER_API_KEY not found. Will skip models without cached endpoints.")
        endpoint_manager = None
    else:
        endpoint_manager = EndpointManager(api_key)
        print("Endpoint manager initialized")
    
    # Step 4: Run evaluations
    print(f"\n{'=' * 60}")
    print(f"RUNNING EVALUATIONS")
    print(f"{'=' * 60}")


    if args.limit:
        print(f"Limiting to {args.limit} samples per evaluation")

    all_results = {}
    
    # Get training fold path for endpoint management
    training_fold_path = base_path / ".together-120b" / "openai" / args.model_name / args.fold_name

    for model_info in trained_models:
        print(f"\n--- Model Epoch {model_info.epoch} ---")

        # if model_info.epoch != 4:
        #     continue
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

            split = "train" #"val" if args.fold_name is eval_fold else "train"

            print(f"Using {split} split for evaluation")

            # Load samples
            if split == "train":
                samples_path = eval_fold.train_path
            else:
                samples_path = eval_fold.val_path

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
    
    # Step 5: Clean up endpoints if needed
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

    # Step 6: Save results
    print(f"\n{'=' * 60}")
    print(f"SAVING RESULTS")
    print(f"{'=' * 60}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "eval_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    output_file = output_dir / f"lie_detection_{args.fold_name}_{args.model_name}_{split}.json"

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Step 7: Print summary
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    for epoch_key, epoch_results in all_results.items():
        print(f"\n{epoch_key}:")

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

    # Step 8: Generate performance charts
    print(f"\n{'=' * 60}")
    print(f"GENERATING CHARTS")
    print(f"{'=' * 60}")
    
    try:
        generate_performance_charts(
            all_results=all_results,
            trained_fold=args.fold_name,
            output_dir=output_dir,
            split=split,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"Warning: Chart generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())