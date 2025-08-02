#!/usr/bin/env python3
"""
Refactored run script demonstrating the improved baseline architecture.
This script is much cleaner and more maintainable than the original run.py.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from baseline.common.config import EvaluationConfig, ConfigManager, get_baseline_types
from baseline.evaluate import BaselineEvaluator, run_multiple_baselines


def main():
    """Main function demonstrating the new refactored architecture."""
    
    # Configuration using the new config management system
    config = EvaluationConfig(
        model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
        num_samples=100,  # Limit for demo
        baseline_types=["escaped_transcript", "rowans_escaped_transcript"],
        split_by_model=True,
        s3_uri=None,  # Use local data
        processed_data_dir=".data/openai/gpt_4o"
    )
    
    # Create configuration manager
    config_manager = ConfigManager(config)
    
    # Validate data directory
    print(f"\n{'='*60}")
    print(f"Using local data directory: {config.processed_data_dir}")
    print(f"{'='*60}")
    
    if not os.path.exists(config.processed_data_dir):
        print(f"Error: Local data directory {config.processed_data_dir} does not exist!")
        print(f"Available data directories in .data/:")
        data_dir = Path(".data")
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item}")
        sys.exit(1)
    
    # Run evaluations for all configured baseline types
    print(f"\nRunning evaluations for: {config.baseline_types}")
    
    results = run_multiple_baselines(
        baseline_types=config.baseline_types,
        num_samples=config.num_samples,
        model=config.model_name,
        data_dir=config.processed_data_dir,
        split_by_model=config.split_by_model
    )
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    for baseline_type, result in results.items():
        if "error" in result:
            print(f"{baseline_type}: ERROR - {result['error']}")
        elif config.split_by_model and "model_metrics" in result:
            print(f"\n{baseline_type}:")
            for model, metrics in result["model_metrics"].items():
                print(f"  {model}: Accuracy={metrics.get('accuracy', 0):.3f}, "
                     f"F1={metrics.get('f1_score', 0):.3f}")
        else:
            metrics = result.get("metrics", {})
            print(f"{baseline_type}: Accuracy={metrics.get('accuracy', 0):.3f}, "
                 f"F1={metrics.get('f1_score', 0):.3f}")


def run_single_baseline_demo():
    """Demonstrate running a single baseline with the new architecture."""
    
    print("=== SINGLE BASELINE DEMO ===")
    
    # Simple configuration for a single baseline
    evaluator = BaselineEvaluator()
    
    # Run just one baseline type with custom settings
    result = evaluator.run_evaluation(
        baseline_type="escaped_transcript",
        num_samples=10,  # Limit to 10 samples for demo
        model="openrouter/meta-llama/llama-3.1-8b-instruct",
        data_dir=".data/openai/gpt_4o",
        split_by_model=False
    )
    
    print(f"\nDemo completed. Processed {result['total_samples']} samples.")
    print(f"Final accuracy: {result['metrics']['accuracy']:.3f}")
    
    return result


def run_factory_demo():
    """Demonstrate the new factory patterns."""
    
    print("=== FACTORY PATTERN DEMO ===")
    
    from baseline.common.task_factory import TaskFactory
    from baseline.common.scorer_factory import ScorerFactory
    
    # Demonstrate task factory
    task_factory = TaskFactory()
    print(f"Supported baseline types: {task_factory.get_supported_baselines()}")
    
    # Demonstrate scorer factory
    scorer_factory = ScorerFactory()
    print(f"Supported scorer types: {scorer_factory.get_supported_types()}")
    
    # Create a task and scorer using the factory
    try:
        # This would work if we had valid data
        # task = task_factory.create_task("escaped_transcript", "some_data_dir", limit=5)
        scorer = scorer_factory.create_scorer("ab_scorer", "demo_baseline")
        print("✓ Successfully created scorer using factory")
    except Exception as e:
        print(f"Note: {e} (expected if data directory doesn't exist)")


def show_configuration_demo():
    """Demonstrate the configuration management."""
    
    print("=== CONFIGURATION DEMO ===")
    
    # Create configuration with custom settings
    config = EvaluationConfig(
        model_name="openrouter/custom-model",
        num_samples=100,
        baseline_types=["llama_chat", "escaped_transcript"],
        split_by_model=False
    )
    
    config_manager = ConfigManager(config)
    
    # Show configuration
    print("Configuration:")
    for key, value in config_manager.to_dict().items():
        print(f"  {key}: {value}")
    
    # Show data source info
    data_info = config_manager.get_data_source_info()
    print(f"\nData source info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Check command line arguments for different demo modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "single":
            run_single_baseline_demo()
        elif mode == "factory":
            run_factory_demo()
        elif mode == "config":
            show_configuration_demo()
        elif mode == "help":
            print("Available modes:")
            print("  python run_refactored.py single   - Run single baseline demo")
            print("  python run_refactored.py factory  - Demonstrate factory patterns")
            print("  python run_refactored.py config   - Show configuration management")
            print("  python run_refactored.py          - Run full evaluation (default)")
        else:
            print(f"Unknown mode: {mode}. Use 'help' for available modes.")
            main()
    else:
        # Default: run full evaluation
        main()