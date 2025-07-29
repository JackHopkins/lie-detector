from typing import List, Dict, Any

from inspect_ai._eval.eval import EvalLogs
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalResults
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


def create_bytetrash_question_dataset(dataset: List[Dict[str, Any]]) -> MemoryDataset:
    """Create samples for the bytetrash baseline.

    This function prepares the conversation up to the intervention point,
    where we'll inject the random character request.
    """
    samples = []
    for item in dataset:
        trace = item.get("messages", [])

        # Convert trace to list of ChatMessage objects
        messages = []
        for turn in trace:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role.lower() == "system":
                # Skip system messages for this format
                continue
            elif role.lower() == "user":
                messages.append(ChatMessageUser(content=content))
            elif role.lower() == "assistant":
                messages.append(ChatMessageAssistant(content=content))

        # Store the original did_lie value and other metadata
        metadata = {
            "did_lie": item.get("meta", {}).get("did_lie"),
            "task": item.get("meta", {}).get("task", "unknown"),
            "sample_id": item.get("meta", {}).get("sample_id", "unknown"),
            "task_id": item.get("meta", {}).get("task_id", "unknown"),
            "original_trace": trace,
            "bytetrash_chars": None  # Will be populated during solving
        }

        # For bytetrash, we pass the messages as input and will handle the intervention in the solver
        samples.append(Sample(input=messages, metadata=metadata))

    dataset = MemoryDataset(samples)

    return dataset

def create_bytetrash_classification_dataset(result: EvalLogs) -> MemoryDataset:
    eval_samples = result[0].samples
    samples = []
    for sample in eval_samples:
        samples.append(Sample(sample.output.completion, metadata=sample.metadata))
    return MemoryDataset(samples)