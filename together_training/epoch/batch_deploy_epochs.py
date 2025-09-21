#!/usr/bin/env python3
"""
Standalone Batch Endpoint Deployment Script

This script deploys all epoch endpoints for a given model/fold simultaneously
rather than lazily. Endpoints are kept alive for 2 hours by default.

Usage:
    python batch_deploy_epochs.py --fold-name mask-factual --base-model openai/gpt-oss-120b
    python batch_deploy_epochs.py --fold-name cot-unfaithfulness --timeout 180  # 3 hours
    python batch_deploy_epochs.py --list-available  # List available folds
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from together_training.epoch.endpoint_manager import EndpointManager
from together_training.epoch.training_state import TrainingState


def find_available_folds(base_model: str = "openai/gpt-oss-120b") -> list:
    """Find all available training folds for a given base model."""
    together_dir = project_root / ".together-120b"
    # Convert model path: openai/gpt-oss-120b -> openai/gpt_oss_120b
    model_path = together_dir / base_model.replace("/", "/").replace("-", "_")
    
    available_folds = []
    if model_path.exists():
        for fold_dir in model_path.iterdir():
            if fold_dir.is_dir() and (fold_dir / "training.json").exists():
                available_folds.append(fold_dir.name)
    
    return sorted(available_folds)


def get_fold_path(fold_name: str, base_model: str = "openai/gpt-oss-120b") -> Path:
    """Get the path to a specific fold directory."""
    together_dir = project_root / ".together-120b"
    # Convert model path: openai/gpt-oss-120b -> openai/gpt_oss_120b
    model_path = together_dir / base_model.replace("/", "/").replace("-", "_")
    fold_path = model_path / fold_name
    
    if not fold_path.exists():
        raise ValueError(f"Fold directory not found: {fold_path}")
    
    if not (fold_path / "training.json").exists():
        raise ValueError(f"Training state file not found: {fold_path}/training.json")
    
    return fold_path


def main():
    parser = argparse.ArgumentParser(
        description="Deploy all epoch endpoints for a model/fold simultaneously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_deploy_epochs.py --fold-name mask-factual
  python batch_deploy_epochs.py --fold-name cot-unfaithfulness --timeout 180
  python batch_deploy_epochs.py --list-available
        """
    )
    
    parser.add_argument(
        "--fold-name",
        type=str,
        help="Name of the fold to deploy (e.g., mask-factual, cot-unfaithfulness)"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Base model identifier (default: openai/gpt-oss-120b)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Endpoint inactive timeout in minutes (default: 120 = 2 hours)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent endpoint deployments (default: 5)"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for endpoints to be ready before exiting"
    )
    
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List all available folds and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list_available:
        print(f"Available folds for {args.base_model}:")
        available_folds = find_available_folds(args.base_model)
        if available_folds:
            for fold in available_folds:
                print(f"  - {fold}")
        else:
            print("  No folds found")
        return
    
    # Validate required arguments
    if not args.fold_name:
        parser.error("--fold-name is required (or use --list-available to see options)")
    
    try:
        # Get fold path and validate
        fold_path = get_fold_path(args.fold_name, args.base_model)
        print(f"Found fold: {fold_path}")
        
        # Load training state
        training_state = TrainingState(str(fold_path))
        training_state.load_state()
        
        # Check if we have any epochs to deploy
        if not training_state.epochs:
            print("❌ No epochs found in training state")
            return
        
        print(f"Found {len(training_state.epochs)} epochs to deploy:")
        for epoch_num, epoch_info in training_state.epochs.items():
            status = epoch_info.get('status', 'unknown')
            model_id = epoch_info.get('model_id', 'unknown')
            print(f"  Epoch {epoch_num}: {model_id} ({status})")
        
        # Initialize endpoint manager
        import os
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("❌ TOGETHER_API_KEY not found in environment")
            return 1
        endpoint_manager = EndpointManager(api_key)
        
        # Batch deploy all endpoints
        print(f"\n🚀 Starting batch deployment with {args.timeout}-minute timeout...")
        print(f"   Max concurrent: {args.max_concurrent}")
        print(f"   Wait for ready: {'Yes' if not args.no_wait else 'No'}")
        
        result = endpoint_manager.batch_deploy_all_epochs(
            fold_path=str(fold_path),
            fold_name=args.fold_name,
            inactive_timeout=args.timeout,
            max_concurrent=args.max_concurrent,
            wait_for_ready=not args.no_wait
        )
        
        # Display results
        print(f"\n📊 Deployment Results:")
        print(f"   Total epochs: {result['total_epochs']}")
        print(f"   Already running: {result['existing']}")
        print(f"   Successfully deployed: {result['created']}")
        print(f"   Failed deployments: {result['failed']}")
        print(f"   Ready endpoints: {result['ready_endpoints']}")
        
        if result.get('failed_epochs'):
            print(f"\n❌ Failed Epochs: {result['failed_epochs']}")
        
        if result['failed'] == 0:
            print(f"\n✅ All endpoints deployed successfully!")
            if not args.no_wait:
                print(f"   All endpoints are ready and will stay alive for {args.timeout} minutes")
            else:
                print(f"   Endpoints are being created (use --no-wait=false to wait for readiness)")
        else:
            print(f"\n⚠️  Some deployments failed. Check the failed epochs above.")
            return 1
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())