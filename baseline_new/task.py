from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, Solver
from typing import Optional, Dict
import sys
import os

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)



# Import from current directory only (baseline_new)
from dataset import baseline_dataset, conversation_dataset, llama_chat_dataset, llama_chat_reasoning_dataset, base_transcript_reasoning_dataset, rowans_escaped_transcript_dataset
from dataset import baseline_dataset_by_model, conversation_dataset_by_model, llama_chat_dataset_by_model, llama_chat_reasoning_dataset_by_model, base_transcript_reasoning_dataset_by_model, rowans_escaped_transcript_dataset_by_model, gt_labelling_rowan_modified_dataset, gt_labelling_rowan_modified_dataset_by_model
from dataset import baseline_dataset_by_model_from_file, conversation_dataset_by_model_from_file, llama_chat_dataset_by_model_from_file, llama_chat_reasoning_dataset_by_model_from_file
from scorer import truth_scorer, escaped_transcript_scorer, llama_chat_scorer, llama_chat_reasoning_scorer, base_transcript_reasoning_scorer
from hooks import BaselineRawDataHook

@task
def baseline_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Baseline lie detection task.
    """
    return Task(
        dataset=baseline_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=truth_scorer(),
    ) 

@task
def escaped_transcript_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Escaped transcript format lie detection task.
    """
    return Task(
        dataset=conversation_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=escaped_transcript_scorer()
    ) 

@task
def llama_chat_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    LLaMA chat format lie detection task.
    """
    return Task(
        dataset=llama_chat_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=llama_chat_scorer(),
    ) 

@task
def llama_chat_reasoning_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    LLaMA chat reasoning format lie detection task.
    """
    return Task(
        dataset=llama_chat_reasoning_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=llama_chat_reasoning_scorer()
    )

@task
def base_transcript_reasoning_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Base transcript reasoning format lie detection task.
    """
    return Task(
        dataset=base_transcript_reasoning_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=base_transcript_reasoning_scorer()
    )

@task
def rowans_escaped_transcript_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    Rowan's escaped transcript format lie detection task.
    """
    return Task(
        dataset=rowans_escaped_transcript_dataset(data_dir, limit=limit),
        solver=generate(),
        scorer=escaped_transcript_scorer(),
    ) 

def baseline_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Baseline lie detection task split by model.
    """
    model_datasets = baseline_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=truth_scorer(),
        )
    
    return model_tasks

def baseline_task_by_model_from_file(
    data_file: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Baseline lie detection task split by model from a specific file.
    """
    model_datasets = baseline_dataset_by_model_from_file(data_file, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=truth_scorer(),
        )
    
    return model_tasks

def escaped_transcript_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Escaped transcript format lie detection task split by model.
    """
    model_datasets = conversation_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=escaped_transcript_scorer(),
        )
    
    return model_tasks

def escaped_transcript_task_by_model_from_file(
    data_file: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Escaped transcript format lie detection task split by model from a specific file.
    """
    model_datasets = conversation_dataset_by_model_from_file(data_file, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=escaped_transcript_scorer(),
        )
    
    return model_tasks

def llama_chat_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    LLaMA chat format lie detection task split by model.
    """
    model_datasets = llama_chat_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=llama_chat_scorer()
        )
    
    return model_tasks

def llama_chat_task_by_model_from_file(
    data_file: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    LLaMA chat format lie detection task split by model from a specific file.
    """
    model_datasets = llama_chat_dataset_by_model_from_file(data_file, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=llama_chat_scorer()
        )
    
    return model_tasks

def llama_chat_reasoning_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    LLaMA chat reasoning format lie detection task split by model.
    """
    model_datasets = llama_chat_reasoning_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=llama_chat_reasoning_scorer()
        )
    
    return model_tasks

def llama_chat_reasoning_task_by_model_from_file(
    data_file: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    LLaMA chat reasoning format lie detection task split by model from a specific file.
    """
    model_datasets = llama_chat_reasoning_dataset_by_model_from_file(data_file, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=llama_chat_reasoning_scorer()
        )
    
    return model_tasks

def base_transcript_reasoning_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Base transcript reasoning format lie detection task split by model.
    """
    model_datasets = base_transcript_reasoning_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=base_transcript_reasoning_scorer()
        )
    
    return model_tasks

def rowans_escaped_transcript_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    Rowan's escaped transcript format lie detection task split by model.
    """
    model_datasets = rowans_escaped_transcript_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(),
            scorer=escaped_transcript_scorer()
        )
    
    return model_tasks


@task
def gt_labelling_rowan_modified_task(
    data_dir: str, 
    limit: Optional[int] = None
) -> Task:
    """
    GT labelling with Rowan's escaped transcript format + lie definitions.
    Prompts the model to classify whether the assistant is lying or telling the truth.
    """
    return Task(
        dataset=gt_labelling_rowan_modified_dataset(data_dir, limit=limit),
        solver=generate(max_tokens=8000, temperature=0.6),
        scorer=escaped_transcript_scorer(),  # Use the same scorer as rowans_escaped_transcript
        # Note: The target is automatically appended by inspect_ai's generate() solver
        # This is expected behavior for evaluation, but for GT labelling we need to be careful
        # about target leakage in the prompt construction
    )


def gt_labelling_rowan_modified_task_by_model(
    data_dir: str, 
    limit: Optional[int] = None
) -> Dict[str, Task]:
    """
    GT labelling with Rowan's escaped transcript format + lie definitions, split by model.
    """
    model_datasets = gt_labelling_rowan_modified_dataset_by_model(data_dir, limit=limit)
    model_tasks = {}
    
    for model, dataset in model_datasets.items():
        model_tasks[model] = Task(
            dataset=dataset,
            solver=generate(max_tokens=8000, temperature=0.6),
            scorer=escaped_transcript_scorer()  # Use the same scorer as rowans_escaped_transcript
        )
    
    return model_tasks