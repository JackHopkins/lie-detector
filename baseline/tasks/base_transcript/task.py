from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.scorer import choice, f1
from inspect_ai.solver import multiple_choice

from baseline.dataset import load_jsonl_data
from baseline.tasks import get_samples
from baseline.tasks.base_transcript.dataset import create_base_transcript_dataset


def create_base_transcript_task(model_name: str, cot: bool = False):
    """Create a task for a specific model"""
    samples = get_samples(model_name)
    return Task(
        name=f"base_transcript" + ("_reasoning" if cot else ''),
        dataset=create_base_transcript_dataset(samples),
        solver=[multiple_choice(cot=cot)],
        scorer=[choice(), f1()]
    )


if __name__ == '__main__':
    model = 'openai/gpt-4o'
    result = eval(
        tasks=[create_base_transcript_task(model_name=model, cot=False),
               create_base_transcript_task(model_name=model, cot=True)],
        model=model,
        log_dir="../../logs",
        limit=50
    )
    pass