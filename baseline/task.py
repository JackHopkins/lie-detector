"""
Refactored task implementation using the new factory pattern.
This eliminates the massive duplication in the original task.py file.
"""

import sys
import os
from typing import Optional, Dict
from inspect_ai import task, Task

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from common.generalization_enums import MotivationFramework, KnowledgeFramework
from baseline.common.task_factory import TaskFactory, create_task_for_baseline, create_tasks_by_model_for_baseline

# Create factory instance
_task_factory = TaskFactory()

# Legacy task functions for backward compatibility
# These now use the factory pattern instead of duplicating code

@task
def baseline_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """Baseline lie detection task."""
    return _task_factory.create_task("baseline", data_dir, limit)


@task 
def escaped_transcript_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """Escaped transcript format lie detection task."""
    return _task_factory.create_task("escaped_transcript", data_dir, limit)


@task
def llama_chat_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """LLaMA chat format lie detection task."""
    return _task_factory.create_task("llama_chat", data_dir, limit)


@task
def llama_chat_reasoning_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """LLaMA chat reasoning format lie detection task."""
    return _task_factory.create_task("llama_chat_reasoning", data_dir, limit)


@task
def base_transcript_reasoning_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """Base transcript reasoning format lie detection task."""
    return _task_factory.create_task("base_transcript_reasoning", data_dir, limit)


@task
def rowans_escaped_transcript_task(data_dir: str, limit: Optional[int] = None) -> Task:
    """Rowan's escaped transcript format lie detection task."""
    return _task_factory.create_task("rowans_escaped_transcript", data_dir, limit)


# Legacy "by_model" functions for backward compatibility
def baseline_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """Baseline lie detection task split by model."""
    return _task_factory.create_tasks_by_model("baseline", data_dir, limit)


def escaped_transcript_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """Escaped transcript format lie detection task split by model."""
    return _task_factory.create_tasks_by_model("escaped_transcript", data_dir, limit)


def llama_chat_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """LLaMA chat format lie detection task split by model."""
    return _task_factory.create_tasks_by_model("llama_chat", data_dir, limit)


def llama_chat_reasoning_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """LLaMA chat reasoning format lie detection task split by model."""
    return _task_factory.create_tasks_by_model("llama_chat_reasoning", data_dir, limit)


def base_transcript_reasoning_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """Base transcript reasoning format lie detection task split by model."""
    return _task_factory.create_tasks_by_model("base_transcript_reasoning", data_dir, limit)


def rowans_escaped_transcript_task_by_model(data_dir: str, limit: Optional[int] = None) -> Dict[str, Task]:
    """Rowan's escaped transcript format lie detection task split by model."""
    return _task_factory.create_tasks_by_model("rowans_escaped_transcript", data_dir, limit)


# New factory-based approach - recommended for new code
def create_task_for_baseline_type(baseline_type: str, data_dir: str, limit: Optional[int] = None) -> Task:
    """
    Create a task for the specified baseline type using the factory pattern.
    
    Args:
        baseline_type: Type of baseline (e.g., "escaped_transcript", "llama_chat")
        data_dir: Directory containing the data
        limit: Optional limit on number of samples
        
    Returns:
        Configured Task object
        
    Example:
        task = create_task_for_baseline_type("escaped_transcript", "/path/to/data")
    """
    return _task_factory.create_task(baseline_type, data_dir, limit)


def create_tasks_by_model_for_baseline_type(
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
        
    Example:
        tasks = create_tasks_by_model_for_baseline_type("llama_chat", "/path/to/data")
    """
    return _task_factory.create_tasks_by_model(baseline_type, data_dir, limit)


# Helper functions
def get_supported_baseline_types() -> list:
    """Get list of supported baseline types."""
    return _task_factory.get_supported_baselines()


def get_baseline_description(baseline_type: str) -> str:
    """Get description for a baseline type."""
    return _task_factory.get_baseline_description(baseline_type)


# Task creation with validation
def create_validated_task(baseline_type: str, data_dir: str, limit: Optional[int] = None) -> Task:
    """
    Create a task with validation of baseline type.
    
    Args:
        baseline_type: Type of baseline
        data_dir: Directory containing the data  
        limit: Optional limit on number of samples
        
    Returns:
        Configured Task object
        
    Raises:
        ValueError: If baseline_type is not supported or data_dir doesn't exist
    """
    if baseline_type not in get_supported_baseline_types():
        raise ValueError(f"Unsupported baseline type: {baseline_type}. "
                        f"Supported types: {get_supported_baseline_types()}")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    return _task_factory.create_task(baseline_type, data_dir, limit)