from pathlib import Path

from inspect_ai import task, Task, eval
from inspect_ai.scorer import choice, f1
from inspect_ai.solver import multiple_choice

from baseline.dataset import load_jsonl_data_by_model, load_jsonl_data
from baseline.tasks import get_samples
from baseline.tasks.escaped_transcript.dataset import create_escaped_transcript_dataset
from baseline.tasks.llama_chat.dataset import create_llama_chat_dataset




@task
def create_escaped_transcript_task(model_name:str, cot=False, condensed=False):
    samples = get_samples(model_name)
    return Task(
        name="escaped_transcript" + ("_reasoning" if cot else '') + ("_condensed" if condensed else ''),
        dataset=create_escaped_transcript_dataset(samples, condensed),
        solver=[
            multiple_choice(cot=cot)
        ],
        scorer=[
            choice(),
            f1()
        ]
    )

if __name__ == '__main__':
    model = 'openai/gpt-4o'
    result = eval(
        tasks=[create_escaped_transcript_task(model_name=model, cot=False),
               create_escaped_transcript_task(model_name=model, cot=True),
               create_escaped_transcript_task(model_name=model,cot=False, condensed=True),
               create_escaped_transcript_task(model_name=model, cot=True, condensed=True)],
        model=model,
        log_dir="../../logs",
        limit=1000
    )
    pass