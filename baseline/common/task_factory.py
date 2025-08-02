"""
Task factory to eliminate duplication in task creation across different baseline types.
"""

from typing import Dict, Optional, Callable, Any
from inspect_ai import Task
from inspect_ai.solver import generate

from .dataset_factory import DatasetFactory
from .scorer_factory import ScorerFactory


class TaskConfig:
    """Configuration for different baseline task types."""
    
    # Map baseline types to their configurations
    BASELINE_CONFIGS = {
        "baseline": {
            "dataset_func": "baseline_dataset",
            "scorer_type": "ab_scorer",
            "description": "Baseline lie detection task"
        },
        "escaped_transcript": {
            "dataset_func": "conversation_dataset", 
            "scorer_type": "score_scorer",
            "description": "Escaped transcript format lie detection task"
        },
        "llama_chat": {
            "dataset_func": "llama_chat_dataset",
            "scorer_type": "ab_scorer", 
            "description": "LLaMA chat format lie detection task"
        },
        "llama_chat_reasoning": {
            "dataset_func": "llama_chat_reasoning_dataset",
            "scorer_type": "ab_scorer",
            "description": "LLaMA chat reasoning format lie detection task"
        },
        "base_transcript_reasoning": {
            "dataset_func": "base_transcript_reasoning_dataset",
            "scorer_type": "ab_scorer",
            "description": "Base transcript reasoning format lie detection task"
        },
        "rowans_escaped_transcript": {
            "dataset_func": "rowans_escaped_transcript_dataset",
            "scorer_type": "score_scorer",
            "description": "Rowan's escaped transcript format lie detection task"
        }
    }


class TaskFactory:
    """Factory for creating baseline tasks with consistent patterns."""
    
    def __init__(self):
        self.dataset_factory = DatasetFactory()
        self.scorer_factory = ScorerFactory()
    
    def create_task(
        self, 
        baseline_type: str, 
        data_dir: str, 
        limit: Optional[int] = None
    ) -> Task:
        """
        Create a task for the specified baseline type.
        
        Args:
            baseline_type: Type of baseline (e.g., "escaped_transcript", "llama_chat")
            data_dir: Directory containing the data
            limit: Optional limit on number of samples
            
        Returns:
            Configured Task object
            
        Raises:
            ValueError: If baseline_type is not supported
        """
        if baseline_type not in TaskConfig.BASELINE_CONFIGS:
            raise ValueError(f"Unsupported baseline type: {baseline_type}")
        
        config = TaskConfig.BASELINE_CONFIGS[baseline_type]
        
        # Create dataset using factory
        dataset = self.dataset_factory.create_dataset(
            config["dataset_func"], data_dir, limit
        )
        
        # Create scorer using factory
        scorer = self.scorer_factory.create_scorer(
            config["scorer_type"], baseline_type
        )
        
        return Task(
            dataset=dataset,
            solver=generate(),
            scorer=scorer
        )
    
    def create_tasks_by_model(
        self, 
        baseline_type: str, 
        data_dir: str, 
        limit: Optional[int] = None
    ) -> Dict[str, Task]:
        """
        Create separate tasks for each model for the specified baseline type.
        
        Args:
            baseline_type: Type of baseline
            data_dir: Directory containing the data
            limit: Optional limit on number of samples per model
            
        Returns:
            Dictionary mapping model names to Task objects
        """
        if baseline_type not in TaskConfig.BASELINE_CONFIGS:
            raise ValueError(f"Unsupported baseline type: {baseline_type}")
        
        config = TaskConfig.BASELINE_CONFIGS[baseline_type]
        
        # Create model-specific datasets
        model_datasets = self.dataset_factory.create_datasets_by_model(
            config["dataset_func"], data_dir, limit
        )
        
        # Create scorer
        scorer = self.scorer_factory.create_scorer(
            config["scorer_type"], baseline_type
        )
        
        # Create tasks for each model
        model_tasks = {}
        for model, dataset in model_datasets.items():
            model_tasks[model] = Task(
                dataset=dataset,
                solver=generate(),
                scorer=scorer
            )
        
        return model_tasks
    
    def get_supported_baselines(self) -> list:
        """Get list of supported baseline types."""
        return list(TaskConfig.BASELINE_CONFIGS.keys())
    
    def get_baseline_description(self, baseline_type: str) -> str:
        """Get description for a baseline type."""
        config = TaskConfig.BASELINE_CONFIGS.get(baseline_type)
        return config["description"] if config else "Unknown baseline type"


# Convenience functions to maintain backward compatibility
def create_task_for_baseline(baseline_type: str, data_dir: str, limit: Optional[int] = None) -> Task:
    """Convenience function to create a task for a baseline type."""
    factory = TaskFactory()
    return factory.create_task(baseline_type, data_dir, limit)


def create_tasks_by_model_for_baseline(
    baseline_type: str, 
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """Convenience function to create model-specific tasks for a baseline type.""" 
    factory = TaskFactory()
    return factory.create_tasks_by_model(baseline_type, data_dir, limit)