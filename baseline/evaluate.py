"""
Refactored evaluation implementation using the new factory patterns and utilities.
This dramatically simplifies the original evaluate.py file by eliminating duplication.
"""

from inspect_ai import eval
from dotenv import load_dotenv
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from baseline.common.config import ConfigManager, EvaluationConfig
from baseline.common.task_factory import TaskFactory
from baseline.common.evaluation_utils import EvaluationSummary, ResultsSaver, ResultsAnalyzer

load_dotenv()


class BaselineEvaluator:
    """Main class for running baseline evaluations with clean architecture."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config_manager = ConfigManager(config)
        self.task_factory = TaskFactory()
    
    def run_evaluation(
        self, 
        baseline_type: str,
        num_samples: int = None,
        model: str = None,
        data_dir: str = None,
        split_by_model: bool = None
    ) -> dict:
        """
        Run evaluation for a specific baseline type.
        
        Args:
            baseline_type: Type of baseline to evaluate
            num_samples: Number of samples to evaluate (None for all)
            model: Model to use for evaluation (None for config default)
            data_dir: Data directory (None for config default)
            split_by_model: Whether to split by model (None for config default)
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        # Apply parameter overrides
        config = self.config_manager.config
        if num_samples is not None:
            config.num_samples = num_samples
        if model is not None:
            config.model_name = model
        if data_dir is not None:
            config.processed_data_dir = data_dir
        if split_by_model is not None:
            config.split_by_model = split_by_model
        
        # Validate configuration
        if not self.config_manager.validate_data_directory():
            raise ValueError(f"Data directory does not exist: {config.processed_data_dir}")
        
        # Setup directories
        self.config_manager.setup_directories(baseline_type)
        
        # Print configuration
        self._print_evaluation_header(baseline_type)
        
        if config.split_by_model:
            return self._run_by_model_evaluation(baseline_type)
        else:
            return self._run_single_evaluation(baseline_type)
    
    def _print_evaluation_header(self, baseline_type: str):
        """Print evaluation header with configuration."""
        config = self.config_manager.config
        print(f"\n{'='*80}")
        print(f"=== RUNNING {baseline_type.upper()} EVALUATION ===")
        print(f"Model: {config.model_name}")
        print(f"Samples: {'All available' if config.num_samples is None else config.num_samples}")
        print(f"Data Directory: {config.processed_data_dir}")
        print(f"Split by Model: {config.split_by_model}")
        print(f"{'='*80}")
    
    def _run_single_evaluation(self, baseline_type: str) -> dict:
        """Run evaluation for a single baseline type (all models combined)."""
        config = self.config_manager.config
        
        # Create task
        task = self.task_factory.create_task(
            baseline_type, 
            config.processed_data_dir, 
            config.num_samples
        )
        
        # Run evaluation
        log_dir = config.get_log_dir(baseline_type)
        log = eval(task, model=config.model_name, log_dir=log_dir)
        results = log[0].samples if log and log[0].samples else []
        
        print(f"\nProcessed {len(results)} samples")
        
        # Analyze and save results
        return self._process_results(baseline_type, results)
    
    def _run_by_model_evaluation(self, baseline_type: str) -> dict:
        """Run evaluation split by model."""
        config = self.config_manager.config
        
        print(f"Running {baseline_type} evaluation by model...")
        
        # Create model-specific tasks
        model_tasks = self.task_factory.create_tasks_by_model(
            baseline_type,
            config.processed_data_dir,
            config.num_samples
        )
        
        print(f"\nFound {len(model_tasks)} models to evaluate:")
        for model_key in model_tasks.keys():
            print(f"  - {model_key}")
        
        # Run evaluation for each model
        all_results = {}
        all_metrics = {}
        
        for model_key, task in model_tasks.items():
            print(f"\n{'='*60}")
            print(f"EVALUATING MODEL: {model_key}")
            print(f"{'='*60}")
            
            # Create model-specific log directory
            log_dir = os.path.join(config.get_log_dir(baseline_type), 
                                 model_key.replace('/', '_').replace(':', '_'))
            
            # Run evaluation
            log = eval(task, model=config.model_name, log_dir=log_dir)
            results = log[0].samples if log and log[0].samples else []
            
            print(f"Processed {len(results)} samples for {model_key}")
            
            if len(results) > 0:
                # Process results for this model
                model_results = self._process_results(
                    baseline_type, 
                    results, 
                    model_suffix=model_key.replace('/', '_').replace(':', '_')
                )
                all_results[model_key] = model_results
                all_metrics[model_key] = model_results['metrics']
        
        # Print overall summary
        self._print_model_summary(all_metrics)
        
        return {
            'baseline_type': baseline_type,
            'split_by_model': True,
            'model_results': all_results,
            'model_metrics': all_metrics
        }
    
    def _process_results(self, baseline_type: str, results: list, model_suffix: str = "") -> dict:
        """Process evaluation results and save outputs."""
        config = self.config_manager.config
        
        # Create directory paths
        results_dir = config.get_results_dir(baseline_type)
        transcripts_dir = config.get_transcripts_dir(baseline_type)
        
        # Create suffix for file names
        file_suffix = f"{model_suffix}_{baseline_type}" if model_suffix else baseline_type
        
        # Initialize results processor
        results_saver = ResultsSaver(transcripts_dir, baseline_type)
        
        # Save prompts
        prompts_file = results_saver.save_prompts(results, file_suffix)
        print(f"📝 Saved prompts to: {prompts_file}")
        
        # Analyze results
        summary = EvaluationSummary(results, baseline_type)
        summary.print_filtering_summary()
        summary.print_classification_summary()
        metrics, confusion_matrix = summary.print_final_metrics()
        
        # Combine metrics for saving
        combined_metrics = {**metrics, **confusion_matrix}
        analyzer = ResultsAnalyzer(baseline_type)
        
        # Count parseable vs filtered samples
        parseable_count = 0
        filtered_count = 0
        for sample in results:
            sample_data = analyzer.analyze_sample(sample)
            if sample_data["is_parseable"]:
                parseable_count += 1
            else:
                filtered_count += 1
        
        combined_metrics.update({
            "parseable_samples": parseable_count,
            "filtered_samples": filtered_count
        })
        
        # Save detailed results
        results_saver_json = ResultsSaver(results_dir, baseline_type)
        json_filepath = results_saver_json.save_results_json(results, combined_metrics, file_suffix)
        txt_filepath = results_saver_json.save_summary_txt(results, combined_metrics, file_suffix)
        
        print(f"📝 Saved results to: {json_filepath}")
        print(f"📝 Saved summary to: {txt_filepath}")
        
        return {
            'baseline_type': baseline_type,
            'total_samples': len(results),
            'metrics': combined_metrics,
            'files': {
                'prompts': prompts_file,
                'results_json': json_filepath,
                'results_txt': txt_filepath
            }
        }
    
    def _print_model_summary(self, all_metrics: dict):
        """Print summary across all models."""
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY BY MODEL")
        print(f"{'='*80}")
        
        for model_key, metrics in all_metrics.items():
            print(f"\n{model_key}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1: {metrics.get('f1_score', 0):.4f}")
            print(f"  Parseable: {metrics.get('parseable_samples', 0)}")


def main(
    num_samples: int = None, 
    model: str = None, 
    data_dir: str = "processed_data", 
    baseline_type: str = "base_transcript"
):
    """
    Main evaluation function with simplified interface.
    
    Args:
        num_samples: Number of samples to evaluate
        model: Model to use for evaluation
        data_dir: Directory containing the data
        baseline_type: Type of baseline to evaluate
    """
    # Create evaluator with default configuration
    evaluator = BaselineEvaluator()
    
    # Run evaluation
    return evaluator.run_evaluation(
        baseline_type=baseline_type,
        num_samples=num_samples,
        model=model,
        data_dir=data_dir,
        split_by_model=False
    )


def main_by_model(
    num_samples: int = None,
    model: str = None, 
    data_dir: str = "processed_data",
    baseline_type: str = "base_transcript"
):
    """
    Main evaluation function split by model.
    
    Args:
        num_samples: Number of samples to evaluate per model
        model: Model to use for evaluation
        data_dir: Directory containing the data
        baseline_type: Type of baseline to evaluate
    """
    # Create evaluator with default configuration
    evaluator = BaselineEvaluator()
    
    # Run evaluation
    return evaluator.run_evaluation(
        baseline_type=baseline_type,
        num_samples=num_samples,
        model=model,
        data_dir=data_dir,
        split_by_model=True
    )


# Convenience function for running multiple baseline types
def run_multiple_baselines(
    baseline_types: list,
    num_samples: int = None,
    model: str = None,
    data_dir: str = "processed_data",
    split_by_model: bool = True
):
    """
    Run evaluation for multiple baseline types.
    
    Args:
        baseline_types: List of baseline types to evaluate
        num_samples: Number of samples to evaluate
        model: Model to use for evaluation
        data_dir: Directory containing the data
        split_by_model: Whether to split by model
    """
    evaluator = BaselineEvaluator()
    results = {}
    
    for baseline_type in baseline_types:
        print(f"\n{'='*40}\nRunning baseline: {baseline_type}\n{'='*40}")
        
        try:
            result = evaluator.run_evaluation(
                baseline_type=baseline_type,
                num_samples=num_samples,
                model=model,
                data_dir=data_dir,
                split_by_model=split_by_model
            )
            results[baseline_type] = result
        except Exception as e:
            print(f"Error evaluating {baseline_type}: {e}")
            results[baseline_type] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # For backward compatibility, run with default settings
    main()