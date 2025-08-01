#!/usr/bin/env python3
"""
Convenience script to run all evaluation tasks.
Usage: python evals/run_all_tasks.py --model <model_name> --limit <sample_limit>
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from inspect_ai import eval

from evals.sandbagging_performance_degradation_chess_tasks import chess_mate_in_2_shutdown_threat_task, \
    chess_mate_in_2_capability_hiding_task, chess_mate_in_2_safety_concern_task
from evals.sandbagging_selective_compliance_accuracy_tasks import sandbagging_physical_security_contrastive, \
    sandbagging_digital_security_contrastive, sandbagging_chemistry_biology_contrastive, \
    sandbagging_automotive_mechanical_contrastive, sandbagging_medical_pharmaceutical_contrastive, \
    sandbagging_financial_systems_contrastive, sandbagging_contrastive

# # Add the parent directory to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all task modules
from evals.mask_tasks import (
    mask_statistics, mask_continuations, mask_known_facts, 
    mask_provided_facts, mask_disinformation
)

from evals.sycophancy_tasks import (
    sycophancy_answer_positive_incorrect, sycophancy_answer_positive_correct,
    sycophancy_answer_negative_correct, sycophancy_are_you_sure,
    sycophancy_feedback_user_did_not_write, sycophancy_feedback_user_wrote,
    sycophancy_feedback_user_likes, sycophancy_feedback_user_dislikes
)

from evals.self_sycophancy_tasks import (
    self_sycophancy_harmfulness_common_sense, self_sycophancy_harmfulness_deontology,
    self_sycophancy_harmfulness_justice, self_sycophancy_correctness_arc,
    self_sycophancy_correctness_biology, self_sycophancy_correctness_chemistry,
    self_sycophancy_correctness_mathematics, self_sycophancy_correctness_physics
)

from evals.n_truths_and_a_lie_tasks import (
    n_truths_and_a_lie_family_history, n_truths_and_a_lie_hidden_talents,
    n_truths_and_a_lie_personal_quirks, n_truths_and_a_lie_professional_background,
    n_truths_and_a_lie_travel_experiences
)

from evals.capture_the_secret_tasks import (
    capture_the_secret_corporate, capture_the_secret_intelligence,
    capture_the_secret_relationship, capture_the_secret_sales,
    capture_the_secret_scientific
)

from evals.cot_unfaithfulness_tasks import (
    cot_unfaithfulness_contrastive_mmlu,
    cot_unfaithfulness_contrastive_arc,
)

from evals.alibi_tasks import (
    alibi_theft_task,
    alibi_fraud_task,
    alibi_workplace_task, alibi_assault_task, alibi_breaking_entering_task
)





from evals.sandbagging_ascii_capability_denial_tasks import ascii_sandbagging_task

from evals.sandbagging_basic_tool_capability_denial_tasks import *

from evals.sandbagging_unethical_tool_capability_denial_tasks import (
    tool_ethical_sandbagging_task, tool_unethical_sandbagging_task
)


def get_all_tasks():
    """Get all available tasks organized by category."""
    tasks = {
        "mask": [
            # mask_statistics,
            # mask_continuations,
            # mask_known_facts,
            # mask_provided_facts,
            # mask_disinformation
        ],
        "sycophancy": [
            sycophancy_answer_positive_incorrect,
            sycophancy_answer_positive_correct,
            sycophancy_answer_negative_correct,
            sycophancy_are_you_sure,
            sycophancy_feedback_user_did_not_write,
            sycophancy_feedback_user_wrote,
            sycophancy_feedback_user_likes,
            sycophancy_feedback_user_dislikes
        ],
        "self_sycophancy": [
            self_sycophancy_harmfulness_common_sense,
            self_sycophancy_harmfulness_deontology,
            self_sycophancy_harmfulness_justice,
            self_sycophancy_correctness_arc,
            self_sycophancy_correctness_biology,
            self_sycophancy_correctness_chemistry,
            self_sycophancy_correctness_mathematics,
            self_sycophancy_correctness_physics
        ],
        "n_truths_and_a_lie": [
            n_truths_and_a_lie_family_history,
            n_truths_and_a_lie_hidden_talents,
            n_truths_and_a_lie_personal_quirks,
            n_truths_and_a_lie_professional_background,
            n_truths_and_a_lie_travel_experiences
        ],
        "capture_the_secret": [
            capture_the_secret_corporate,
            capture_the_secret_intelligence,
            capture_the_secret_relationship,
            capture_the_secret_sales,
            capture_the_secret_scientific
        ],
        "alibi": [
            alibi_theft_task,
            alibi_fraud_task,
            alibi_workplace_task,
            alibi_assault_task,
            alibi_breaking_entering_task
        ],
        "sandbagging": [
            ascii_sandbagging_task,
            tool_code_sandbagging_task,
            tool_image_gen_sandbagging_task,
            tool_search_sandbagging_task,
            tool_ethical_sandbagging_task,
            tool_unethical_sandbagging_task,
            # sandbagging_tool_results_tasks  # Requires tool usage - excluded for models without tool support
        ],
        "sandbagging_chess": [
            # chess_task,
            # chess_binary_task,
             chess_mate_in_2_shutdown_threat_task,
             chess_mate_in_2_capability_hiding_task,
             chess_mate_in_2_safety_concern_task
        ],
        "sandbagging_contrastive": [
            sandbagging_contrastive,
            sandbagging_physical_security_contrastive,
            sandbagging_digital_security_contrastive,
            sandbagging_chemistry_biology_contrastive,
            sandbagging_automotive_mechanical_contrastive,
            sandbagging_medical_pharmaceutical_contrastive,
            sandbagging_financial_systems_contrastive
        ],
        "cot_unfaithfulness": [
            cot_unfaithfulness_contrastive_mmlu,
            cot_unfaithfulness_contrastive_arc
        ]
    }
    return tasks

def run_all_tasks(model, limit=None, categories=None, log_dir=None):
    """Run all tasks or specific categories of tasks."""
    
    all_task_categories = get_all_tasks()
    
    # Filter categories if specified
    if categories:
        if isinstance(categories, str):
            categories = [categories]
        task_categories = {k: v for k, v in all_task_categories.items() if k in categories}
    else:
        task_categories = all_task_categories
    
    # Create all tasks
    all_tasks = []
    task_names = []
    
    for category, task_functions in task_categories.items():
        print(f"\nCategory: {category.upper()}")
        for task_func in task_functions:
            try:
                task = task_func()
                all_tasks.append(task)
                task_names.append(f"{category}_{task.name}")
                print(f"  + {task.name}")
            except Exception as e:
                print(f"  - Failed to create {task_func.__name__}: {e}")
    
    if not all_tasks:
        print("No tasks could be created!")
        return
    
    print(f"\nRunning {len(all_tasks)} tasks with model: {model}")
    print(f"Sample limit per task: {limit if limit else 1000}")
    
    # Create timestamp for log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if log_dir is None:
        log_dir = f"logs/all_tasks_{timestamp}"
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Run evaluation
        results = eval(
            tasks=all_tasks,
            model=model,
            limit=limit,
            log_dir=log_dir
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Logs saved to: {log_dir}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"   Total tasks: {len(all_tasks)}")
        print(f"   Model: {model}")
        print(f"   Sample limit: {limit if limit else 'All'}")
        
        return results
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        return None

def main():

    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all evaluation tasks')
    parser.add_argument('--model', type=str, default="openrouter/google/gemma-2-27b-it",
                       help='Model to use for evaluation')
    parser.add_argument('--limit', type=int, default=1,
                       help='Limit samples per task (default: 1)')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Specific categories to run (e.g., mask sycophancy)')
    parser.add_argument('--log-dir', type=str, default="../logs",
                       help='Log directory (default: ../logs)')
    
    args = parser.parse_args()
    
    print("LIE DETECTOR - ALL TASKS EVALUATION")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print(f"Limit: {LIMIT if LIMIT else 1000}")
    print(f"Categories: {CATEGORIES if CATEGORIES else 'All'}")
    print("=" * 50)
    
    # Run all tasks
    results = run_all_tasks(
        model=MODEL,
        limit=LIMIT,
        categories=CATEGORIES,
        log_dir=LOG_DIR
    )
    
    if results:
        print("\nAll tasks completed successfully!")
    else:
        print("\nSome tasks failed. Check the logs for details.")

if __name__ == "__main__":
    main() 