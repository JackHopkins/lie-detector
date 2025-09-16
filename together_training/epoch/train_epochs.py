#!/usr/bin/env python3
"""
CLI Interface for Epoch-by-Epoch TogetherAI Training

This script provides a command-line interface for training models
epoch-by-epoch with automatic resumption and state management.

Usage:
    python -m together_training.epoch.train_epochs <fold_path> [options]
    
Examples:
    # Train a single epoch
    python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games
    
    # Train all 5 epochs
    python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --max-epochs 5 --wait
    
    # Check status
    python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --status
    
    # Clean up failed epochs
    python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --cleanup
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .epoch_trainer import EpochTrainer


def get_api_key() -> str:
    """Get TogetherAI API key from environment."""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable is required")
        print("Set it with: export TOGETHER_API_KEY=your_api_key_here")
        sys.exit(1)
    return api_key


def print_status(status_info: dict) -> None:
    """Print formatted status information."""
    summary = status_info['summary']
    
    print(f"\n=== Training Status for {summary['fold_path']} ===")
    print(f"Base model: {summary['base_model']}")
    print(f"Total epochs tracked: {summary['total_epochs']}")
    print(f"Completed epochs: {len(summary['completed_epochs'])}")
    print(f"Running epochs: {summary['running_epochs']}")
    print(f"Failed epochs: {summary['failed_epochs']}")
    print(f"Next epoch to train: {summary['next_epoch']}")
    
    if summary['completed_epochs']:
        print(f"\nCompleted epoch models:")
        for epoch, model_id in summary['models'].items():
            print(f"  Epoch {epoch}: {model_id}")
    
    if summary['files_cached']:
        print(f"\nCached files: {summary['files_cached']}")


def print_deployment_info(deploy_info: dict) -> None:
    """Print deployment information for completed models."""
    print(f"\n=== Deployment Information ===")
    print(f"Fold: {deploy_info['fold_name']}")
    print(f"Base model: {deploy_info['base_model']}")
    print(f"Total trained epochs: {deploy_info['total_epochs']}")
    
    if deploy_info['models']:
        print(f"\nDeployable models:")
        for epoch_name, info in deploy_info['models'].items():
            print(f"  {epoch_name}:")
            print(f"    Model ID: {info['model_id']}")
            print(f"    Suggested endpoint name: {info['endpoint_name']}")
            print(f"    Description: {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Epoch-by-Epoch TogetherAI Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )
    
    # Positional arguments
    parser.add_argument(
        'fold_path',
        help='Path to fold directory containing train.jsonl and val.jsonl'
    )
    
    # Training options
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=5,
        help='Maximum number of epochs to train (default: 5)'
    )
    
    parser.add_argument(
        '--base-model',
        default='openai/gpt-oss-120b',
        help='Base model to start training from (default: openai/gpt-oss-120b)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate for training (default: 1e-5)'
    )
    
    # Workflow options
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for current epoch to complete before returning'
    )
    
    parser.add_argument(
        '--all-epochs',
        action='store_true',
        help='Train all epochs sequentially until completion'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per epoch in minutes (default: 60)'
    )
    
    # Status and management
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current training status (syncs with TogetherAI API)'
    )
    
    parser.add_argument(
        '--status-local',
        action='store_true',
        help='Show cached local status without API calls'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up failed epochs to allow retry'
    )
    
    parser.add_argument(
        '--deploy-info',
        action='store_true',
        help='Show deployment information for completed models'
    )
    
    # Optional API keys and logging
    parser.add_argument(
        '--wandb-api-key',
        help='Weights & Biases API key for experiment logging'
    )
    
    parser.add_argument(
        '--wandb-project-name',
        default='lie-detection-folds-harmony',
        help='Weights & Biases project name (default: lie-detection-megafolds-harmony)'
    )
    
    args = parser.parse_args()
    
    # Validate fold path
    fold_path = Path(args.fold_path).resolve()
    if not fold_path.exists():
        print(f"Error: Fold path does not exist: {fold_path}")
        sys.exit(1)
    
    if not fold_path.is_dir():
        print(f"Error: Fold path is not a directory: {fold_path}")
        sys.exit(1)
    
    # Check for required files
    train_file = fold_path / "train.jsonl"
    val_file = fold_path / "val.jsonl"
    
    if not train_file.exists():
        print(f"Error: train.jsonl not found in {fold_path}")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"Error: val.jsonl not found in {fold_path}")
        sys.exit(1)
    
    # Get API key
    api_key = get_api_key()
    
    # Initialize trainer
    trainer = EpochTrainer(
        api_key=api_key,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        wandb_api_key=args.wandb_api_key,
        wandb_project_name=args.wandb_project_name
    )
    
    try:
        # Handle status check
        if args.status:
            status_info = trainer.get_fold_status(str(fold_path), sync_with_api=True)
            print_status(status_info)
            return
        
        # Handle local status check  
        if args.status_local:
            status_info = trainer.get_fold_status(str(fold_path), sync_with_api=False)
            print_status(status_info)
            return
        
        # Handle cleanup
        if args.cleanup:
            trainer.cleanup_failed_training(str(fold_path))
            print("Failed epochs cleaned up. You can now retry training.")
            return
        
        # Handle deployment info
        if args.deploy_info:
            deploy_info = trainer.deploy_epoch_models(str(fold_path))
            print_deployment_info(deploy_info)
            return
        
        # Handle training workflows
        if args.all_epochs:
            print(f"Starting complete training: {args.max_epochs} epochs")
            result = trainer.train_until_complete(
                fold_path=str(fold_path),
                max_epochs=args.max_epochs,
                timeout_per_epoch=args.timeout
            )
            
            if result['status'] == 'completed':
                print(f"\n✓ All {args.max_epochs} epochs completed successfully!")
                print_deployment_info(trainer.deploy_epoch_models(str(fold_path)))
            else:
                print(f"\n✗ Training failed: {result['message']}")
                sys.exit(1)
        
        else:
            # Train single epoch
            print("Training single epoch...")
            result = trainer.train_single_epoch(
                fold_path=str(fold_path),
                max_epochs=args.max_epochs
            )
            
            if result['status'] == 'completed':
                print(f"\n✓ Training completed! All {args.max_epochs} epochs finished.")
                print_deployment_info(trainer.deploy_epoch_models(str(fold_path)))
                
            elif result['status'] == 'training':
                print(f"\n✓ Epoch {result['current_epoch']} started: {result['job_id']}")
                
                if args.wait:
                    print("Waiting for epoch to complete...")
                    monitor_result = trainer.monitor_and_wait(str(fold_path), args.timeout)
                    
                    if monitor_result['status'] == 'completed':
                        print(f"\n✓ Epoch {monitor_result['epoch']} completed!")
                        print(f"Model ID: {monitor_result['model_id']}")
                        
                        # Show status after completion
                        status_info = trainer.get_fold_status(str(fold_path))
                        print_status(status_info)
                    else:
                        print(f"\n✗ Epoch failed: {monitor_result['message']}")
                        sys.exit(1)
                else:
                    print(f"Use --wait to monitor completion, or run with --status to check later")
                    print(f"Job ID: {result['job_id']}")
            
            elif result['status'] == 'error':
                print(f"\n✗ Training failed: {result['message']}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()