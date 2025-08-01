import os
import sys
from inspect_ai import task, Task

# Add the project root to path to import sandbagging module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.generalization_enums import MotivationFramework, KnowledgeFramework

from pipelines.sandbagging.selective_compliance.answer_accuracy.dataset import create_sandbagging_contrastive_dataset
from pipelines.sandbagging.selective_compliance.answer_accuracy.solver import sandbagging_contrastive_solver, sandbagging_with_refusal_detection
from pipelines.sandbagging.selective_compliance.answer_accuracy.scorer import sandbagging_detection_scorer, sandbagging_control_accuracy_scorer, sandbagging_harmful_accuracy_scorer
from pipelines.sandbagging.selective_compliance.answer_accuracy.models import SandbaggingDomain
from pipelines.sandbagging.selective_compliance.answer_accuracy.hooks import *



@task
def sandbagging_contrastive():
    """Main sandbagging detection task using contrastive pairs."""
    dataset = create_sandbagging_contrastive_dataset()
    
    return Task(
        name="sandbagging_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[
            sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()
        ],
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )



@task
def sandbagging_physical_security_contrastive():
    """Contrastive sandbagging task for physical security domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.PHYSICAL_SECURITY])
    
    return Task(
        name="sandbagging_physical_security_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )



@task
def sandbagging_digital_security_contrastive():
    """Contrastive sandbagging task for digital security domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.DIGITAL_SECURITY])
    
    return Task(
        name="sandbagging_digital_security_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        sandbox="local",
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )




@task
def sandbagging_chemistry_biology_contrastive():
    """Contrastive sandbagging task for chemistry/biology domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.CHEMISTRY_BIOLOGY])
    
    return Task(
        name="sandbagging_chemistry_biology_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        sandbox="local",
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )




@task
def sandbagging_automotive_mechanical_contrastive():
    """Contrastive sandbagging task for automotive/mechanical domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.AUTOMOTIVE_MECHANICAL])
    
    return Task(
        name="sandbagging_automotive_mechanical_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        sandbox="local",
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )



@task
def sandbagging_medical_pharmaceutical_contrastive():
    """Contrastive sandbagging task for medical/pharmaceutical domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.MEDICAL_PHARMACEUTICAL])
    
    return Task(
        name="sandbagging_medical_pharmaceutical_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        sandbox="local",
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )


@task
def sandbagging_financial_systems_contrastive():
    """Contrastive sandbagging task for financial systems domain."""
    dataset = create_sandbagging_contrastive_dataset(categories=[SandbaggingDomain.FINANCIAL_SYSTEMS])
    
    return Task(
        name="sandbagging_financial_systems_contrastive",
        dataset=dataset,
        solver=sandbagging_contrastive_solver(),
        scorer=[sandbagging_detection_scorer(),
            sandbagging_control_accuracy_scorer(),
            sandbagging_harmful_accuracy_scorer()],
        sandbox="local",
        metadata={
            "motivation_framework": MotivationFramework.INTERNALIZED_DRIVE_TO_LIE,
            "knowledge_framework": KnowledgeFramework.IN_CONTEXT_KNOWLEDGE
        }
    )



if __name__ == "__main__":
    # Example usage - run judge-based lie detection evaluation
    from inspect_ai import eval
    
    all_tasks = [
      sandbagging_financial_systems_contrastive(),
      sandbagging_medical_pharmaceutical_contrastive(),
      sandbagging_automotive_mechanical_contrastive(),
      sandbagging_chemistry_biology_contrastive(),
      sandbagging_digital_security_contrastive(),
      sandbagging_physical_security_contrastive()
    ]
    model = "openrouter/openai/gpt-4o"
    #model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    #model = "openrouter/meta-llama/llama-3.1-8b-base"
    #model = "openrouter/meta-llama/llama-3.1-405b"
    # model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    model2 = ["openrouter/anthropic/claude-3.7-sonnet",
             "openrouter/anthropic/claude-sonnet-4",
             "openrouter/google/gemma-3-27b-it",
             "openrouter/google/gemma-3-9b-it"
             ]
    
    # Run evaluation with a specific model
    results = eval(
        model=model,
        tasks=all_tasks,#all_tasks,
        #token_limit=1,
        retry_on_error=8,
        limit=125,  # Limit samples for faster debugging
        log_dir="../logs",  # Specify log directory
    )