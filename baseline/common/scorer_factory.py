"""
Scorer factory to eliminate duplication across different baseline scorers.
"""

from typing import Dict, Callable
from inspect_ai.scorer import Scorer

from .base_scorer import create_ab_scorer, create_score_based_scorer


class ScorerFactory:
    """Factory for creating different types of scorers."""
    
    def __init__(self):
        self._scorer_registry = {
            "ab_scorer": self._create_ab_scorer,
            "score_scorer": self._create_score_scorer
        }
    
    def create_scorer(self, scorer_type: str, baseline_name: str) -> Scorer:
        """
        Create a scorer of the specified type.
        
        Args:
            scorer_type: Type of scorer ("ab_scorer" or "score_scorer")
            baseline_name: Name of the baseline for scorer identification
            
        Returns:
            Configured scorer function
            
        Raises:
            ValueError: If scorer_type is not supported
        """
        if scorer_type not in self._scorer_registry:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")
        
        return self._scorer_registry[scorer_type](baseline_name)
    
    def _create_ab_scorer(self, baseline_name: str) -> Scorer:
        """Create an A/B answer scorer."""
        return create_ab_scorer(f"{baseline_name}_scorer")
    
    def _create_score_scorer(self, baseline_name: str) -> Scorer:
        """Create a score-based scorer that extracts numeric scores."""
        return create_score_based_scorer(f"{baseline_name}_scorer")
    
    def register_scorer_type(self, scorer_type: str, creator_func: Callable[[str], Scorer]):
        """
        Register a new scorer type.
        
        Args:
            scorer_type: Name of the scorer type
            creator_func: Function that creates the scorer given a baseline name
        """
        self._scorer_registry[scorer_type] = creator_func
    
    def get_supported_types(self) -> list:
        """Get list of supported scorer types."""
        return list(self._scorer_registry.keys())


# Legacy scorer functions for backward compatibility
def truth_scorer():
    """Legacy truth scorer - use ScorerFactory instead."""
    return create_ab_scorer("truth_scorer")()


def escaped_transcript_scorer():
    """Legacy escaped transcript scorer - use ScorerFactory instead."""
    return create_score_based_scorer("escaped_transcript_scorer")()


def llama_chat_scorer():
    """Legacy LLaMA chat scorer - use ScorerFactory instead."""
    return create_ab_scorer("llama_chat_scorer")()


def llama_chat_reasoning_scorer():
    """Legacy LLaMA chat reasoning scorer - use ScorerFactory instead."""
    return create_ab_scorer("llama_chat_reasoning_scorer")()


def base_transcript_reasoning_scorer():
    """Legacy base transcript reasoning scorer - use ScorerFactory instead."""
    return create_ab_scorer("base_transcript_reasoning_scorer")()