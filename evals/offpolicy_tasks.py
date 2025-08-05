"""Task definitions for off-policy lie detection evaluation."""

from typing import Optional, List

from inspect_ai import task, Task, eval

from pipelines.offpolicy.doluschat.dataset import doluschat_dataset
from pipelines.offpolicy.solver import offpolicy_passthrough_solver
from pipelines.offpolicy.hook import OffPolicyHook
from pipelines.offpolicy.truth_is_universal.dataset import truth_is_universal_dataset
from pipelines.offpolicy.halueval.dataset import halueval_dataset
from pipelines.offpolicy.truthfulqa.dataset import truthfulqa_dataset


@task
def doluschat_task(
        dataset_name: str = "AlignmentResearch/DolusChat",
        split: str = "train",
        limit: Optional[int] = None,
) -> Task:
    """
    Minimal passthrough task that just extracts and saves dataset samples.
    This is useful for data extraction without any generation.

    Args:
        dataset_name: HuggingFace dataset to use
        split: Dataset split
        limit: Maximum samples
        model: Model name (for logging only)

    Returns:
        Passthrough task
    """

    return Task(
        name="off-policy_dolus_chat",
        dataset=doluschat_dataset(
            dataset_name=dataset_name,
            split=split,
            limit=limit
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

@task
def truth_is_universal_task(
    limit: Optional[int] = None,
) -> Task:
    """
    Minimal passthrough task that just extracts and saves dataset samples.
    This is useful for data extraction without any generation.

    Args:
        dataset_name: HuggingFace dataset to use
        split: Dataset split
        limit: Maximum samples
        model: Model name (for logging only)

    Returns:
        Passthrough task
    """

    return Task(
        name="off-policy_truth_is_universal",
        dataset=truth_is_universal_dataset(limit=limit),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        #hooks=[OffPolicyHook()]
    )

@task
def halueval_task(
    subsets: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    HaluEval off-policy task for hallucination detection evaluation.
    
    Args:
        subsets: List of HaluEval subsets to use (e.g., ["dialogue", "qa"])
                If None, uses default subsets
        limit: Maximum samples per subset
        
    Returns:
        Passthrough task for HaluEval dataset
    """
    
    return Task(
        name="off-policy_halueval",
        dataset=halueval_dataset(
            subsets=subsets,
            limit=limit
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

@task
def truthfulqa_task(
    split: str = "validation",
    limit: Optional[int] = None,
    balance_labels: bool = True,
    include_alternatives: bool = False
) -> Task:
    """
    TruthfulQA off-policy task for evaluating truthfulness in responses.
    
    TruthfulQA tests whether models give truthful answers to questions
    designed to elicit false answers based on common misconceptions.
    
    Args:
        split: Dataset split to use ("validation" is the main split)
        limit: Maximum number of questions to load
        balance_labels: Whether to balance truthful and deceptive samples
        include_alternatives: Whether to include alternative correct answers
        
    Returns:
        Passthrough task for TruthfulQA dataset
    """
    
    return Task(
        name="off-policy_truthfulqa",
        dataset=truthfulqa_dataset(
            split=split,
            limit=limit,
            balance_labels=balance_labels,
            include_alternatives=include_alternatives
        ),
        solver=offpolicy_passthrough_solver(),
        scorer=[],
        hooks=[OffPolicyHook()]
    )

if __name__ == "__main__":
    results = eval(
        tasks=[truthfulqa_task()],#[truth_is_universal_task(), doluschat_task(), halueval_task()],
        #model=["openai/gpt-4o"],
        model=["openrouter/google/gemma-3-4b-it",
               "openrouter/google/gemma-3-12b-it",
               "openrouter/google/gemma-3-27b-it"],  # Using OpenRouter model
        limit=2000,  # Limit samples for faster debugging
        log_dir="../logs",  # Specify log directory
        max_connections=100
        # debug_errors=True,  # Optional: enable debug mode
    )
