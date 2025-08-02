"""
Refactored scorer implementation using the new base scorer functionality.
This eliminates the massive code duplication in the original scorer.py file.
"""

from baseline.common.base_scorer import create_ab_scorer, create_score_based_scorer
from baseline.common.scorer_factory import ScorerFactory

# Create scorer factory instance
_scorer_factory = ScorerFactory()

# Legacy scorer functions for backward compatibility
# These now use the new factories instead of duplicating code

def truth_scorer():
    """
    Scorer that calculates accuracy, precision, recall, and F1 score components.
    Uses the new base scorer with A/B parsing logic.
    """
    return create_ab_scorer("truth_scorer")()


def escaped_transcript_scorer():
    """
    Scorer that extracts scores from <score> tags and calculates lie detection metrics.
    Expects target to be 'A' (lie) or 'B' (truthful).
    """
    return create_score_based_scorer("escaped_transcript_scorer")()


def llama_chat_scorer():
    """
    Scorer for LLaMA chat baseline that extracts A/B answers.
    Uses the same logic as truth_scorer since both ask for A/B responses.
    """
    return create_ab_scorer("llama_chat_scorer")()


def llama_chat_reasoning_scorer():
    """
    Scorer for LLaMA chat reasoning baseline that extracts A/B answers from <answer> tags.
    Uses the same logic as llama_chat_scorer but prioritizes <answer> tags.
    """
    return create_ab_scorer("llama_chat_reasoning_scorer")()


def base_transcript_reasoning_scorer():
    """
    Scorer for base transcript reasoning baseline that extracts A/B answers from <answer> tags.
    Uses the same logic as truth_scorer but prioritizes <answer> tags.
    """
    return create_ab_scorer("base_transcript_reasoning_scorer")()


# New factory-based approach - recommended for new code
def create_scorer_for_baseline(baseline_type: str):
    """
    Create a scorer for the specified baseline type using the factory pattern.
    
    Args:
        baseline_type: Type of baseline (e.g., "escaped_transcript", "llama_chat")
        
    Returns:
        Configured scorer function
        
    Example:
        scorer = create_scorer_for_baseline("escaped_transcript")
    """
    # Map baseline types to scorer types
    baseline_to_scorer_type = {
        "baseline": "ab_scorer",
        "escaped_transcript": "score_scorer",
        "llama_chat": "ab_scorer",
        "llama_chat_reasoning": "ab_scorer", 
        "base_transcript_reasoning": "ab_scorer",
        "rowans_escaped_transcript": "score_scorer"
    }
    
    scorer_type = baseline_to_scorer_type.get(baseline_type, "ab_scorer")
    return _scorer_factory.create_scorer(scorer_type, baseline_type)


# Helper function to get all available scorer types
def get_available_scorers():
    """Get list of available scorer types from the factory."""
    return _scorer_factory.get_supported_types()


# Helper function to register custom scorer types
def register_custom_scorer(scorer_type: str, creator_func):
    """
    Register a custom scorer type with the factory.
    
    Args:
        scorer_type: Name of the scorer type
        creator_func: Function that creates the scorer given a baseline name
    """
    _scorer_factory.register_scorer_type(scorer_type, creator_func)