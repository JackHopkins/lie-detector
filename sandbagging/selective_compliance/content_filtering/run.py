#!/usr/bin/env python3
"""
Run script for HarmBench content filtering sandbagging detection.

This script demonstrates how to run content filtering evaluations using
harmful content from the HarmBench dataset (specifically chemical/biological category).
"""

from inspect_ai import eval_set, eval_async
from tasks import (
    harmbench_content_filtering_task,
    harmbench_quick_test_task,
    harmbench_comprehensive_test_task,
    harmbench_security_framing_task,
    harmbench_explicit_instruction_task,
    harmbench_multi_solver_comparison_task
)
import asyncio
import argparse


async def run_harmbench_quick_test():
    """Run a quick test with HarmBench data."""
    print("Running HarmBench quick test...")
    print("This will test with chemical/biological harmful content from HarmBench")

    task = harmbench_quick_test_task(judge_model="openai/gpt-4o")

    results = await eval_async(
        tasks=[task],
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"HarmBench quick test completed. R esults: {results}")
    return results


async def run_harmbench_evaluation():
    """Run standard HarmBench content filtering evaluation."""
    print("Running HarmBench content filtering evaluation...")
    print("Testing selective omission of chemical/biological harmful content")

    task = harmbench_content_filtering_task(
        num_samples=10,
        num_harmful_files=3,  # HarmBench chemical/biological files
        num_benign_files=5,  # Control files (recipes, guides, etc.)
        solver_type="probing",
        judge_model="openai/gpt-4o"
    )

    results = await eval_async(
        tasks=[task],
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"HarmBench evaluation completed. Results: {results}")
    return results


async def run_harmbench_comprehensive():
    """Run comprehensive HarmBench test."""
    print("Running comprehensive HarmBench test...")
    print("This will take longer but provide more thorough results")

    task = harmbench_comprehensive_test_task(judge_model="openai/gpt-4o")

    results = await eval_async(
        tasks=[task],
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"Comprehensive HarmBench test completed. Results: {results}")
    return results


async def run_harmbench_security_framing():
    """Test HarmBench content with security audit framing."""
    print("Running HarmBench security framing test...")
    print("Testing if security context changes omission behavior")

    task = harmbench_security_framing_task(judge_model="openai/gpt-4o")

    results = await eval_async(
        tasks=[task],
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"Security framing test completed. Results: {results}")
    return results


async def run_harmbench_explicit_instructions():
    """Test with explicit instructions to report all HarmBench content."""
    print("Running HarmBench explicit instruction test...")
    print("Testing if explicit instructions prevent selective omission")

    task = harmbench_explicit_instruction_task(judge_model="openai/gpt-4o")

    results = await eval_async(
        tasks=[task],
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"Explicit instruction test completed. Results: {results}")
    return results


async def run_harmbench_solver_comparison():
    """Compare different solver strategies on HarmBench content."""
    print("Running HarmBench solver comparison...")
    print("Comparing basic, probing, explicit, educational, and security approaches")

    tasks = harmbench_multi_solver_comparison_task(judge_model="openai/gpt-4o")

    results = await eval_async(
        tasks=tasks,
        model="openai/gpt-4o",
        log_dir="../../../logs"
    )

    print(f"Solver comparison completed. Results: {results}")
    return results


async def run_all_harmbench_tests():
    """Run all HarmBench content filtering tests."""
    print("Running all HarmBench content filtering tests...")
    print("=" * 50)

    # Quick test first
    print("\n1. Quick Test")
    await run_harmbench_quick_test()

    # Standard evaluation
    print("\n2. Standard Evaluation")
    await run_harmbench_evaluation()

    # Security framing
    print("\n3. Security Framing")
    await run_harmbench_security_framing()

    # Explicit instructions
    print("\n4. Explicit Instructions")
    await run_harmbench_explicit_instructions()

    # Solver comparison
    print("\n5. Solver Comparison")
    await run_harmbench_solver_comparison()

    # Comprehensive test (takes longer)
    print("\n6. Comprehensive Test")
    await run_harmbench_comprehensive()

    print("\n" + "=" * 50)
    print("All HarmBench tests completed!")


def main():
    """Main entry point for HarmBench evaluation."""
    parser = argparse.ArgumentParser(
        description="Run HarmBench content filtering sandbagging detection evaluations"
    )
    parser.add_argument(
        "--test",
        choices=["quick", "standard", "comprehensive", "security", "explicit", "comparison", "all"],
        default="quick",
        help="Which test to run"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4",
        help="Model to evaluate (e.g., openai/gpt-4, anthropic/claude-3-sonnet)"
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o",
        help="Model to use for judging/scoring"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override number of samples (if applicable)"
    )

    args = parser.parse_args()

    # Map test names to functions
    test_functions = {
        "quick": run_harmbench_quick_test,
        "standard": run_harmbench_evaluation,
        "comprehensive": run_harmbench_comprehensive,
        "security": run_harmbench_security_framing,
        "explicit": run_harmbench_explicit_instructions,
        "comparison": run_harmbench_solver_comparison,
        "all": run_all_harmbench_tests
    }

    # Run the selected test
    test_func = test_functions[args.test]
    asyncio.run(test_func())


if __name__ == "__main__":
    main()

# Example usage:
# python run_harmbench.py --test quick
# python run_harmbench.py --test standard --model anthropic/claude-3-sonnet
# python run_harmbench.py --test comparison --judge-model anthropic/claude-3-opus
# python run_harmbench.py --test all --model openai/gpt-4