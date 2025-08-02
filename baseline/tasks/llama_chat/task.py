from inspect_ai import Task, eval
from inspect_ai.scorer import choice, f1
from inspect_ai.solver import multiple_choice

from baseline.tasks import get_samples
from baseline.tasks.llama_chat.dataset import create_llama_chat_dataset


def create_llama_chat_task(model_name, cot=False):
    samples = get_samples(model_name)
    return Task(
        name="llama_chat" + "_reasoning" if cot else None,
        dataset=create_llama_chat_dataset(samples),
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
        tasks=[create_llama_chat_task(model_name=model, cot=False),
               create_llama_chat_task(model_name=model, cot=True)],
        model=model,
        log_dir="../../logs",
        limit=50
    )
    pass