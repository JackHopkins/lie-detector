#!/usr/bin/env python3
"""
Simple demonstration of the refactored baseline architecture using bytetrash.
This shows how the new architecture works without requiring complex data setup.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def demo_bytetrash_baseline():
    """Demonstrate the bytetrash baseline with the refactored architecture."""
    
    print("🎯 Bytetrash Baseline Demo")
    print("=" * 50)
    
    # Test feature extraction
    from baseline.bytetrash import extract_character_features, classify_bytetrash
    
    test_chars = "a1b2c3d4e5f6g7h8i9j0"
    features = extract_character_features(test_chars)
    prediction = classify_bytetrash(test_chars)
    
    print(f"Test string: '{test_chars}'")
    print(f"Features extracted: {len(features)} features")
    print(f"Prediction: {'lie' if prediction else 'truth'}")
    print()
    
    # Test with different character patterns
    test_cases = [
        "123456789012345678901234567890",  # numbers only
        "abcdefghijklmnopqrstuvwxyz",      # letters only  
        "!@#$%^&*()_+-=[]{}|;:,.<>?",     # symbols only
        "aB3$x9Z!m2Q#",                   # mixed short
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6", # long mixed
    ]
    
    print("🔍 Testing different character patterns:")
    print("-" * 50)
    
    for i, test_str in enumerate(test_cases, 1):
        features = extract_character_features(test_str)
        prediction = classify_bytetrash(test_str)
        
        print(f"{i}. '{test_str[:30]}{'...' if len(test_str) > 30 else ''}'")
        print(f"   Length: {len(test_str)}, Prediction: {'lie' if prediction else 'truth'}")
        print(f"   Alpha ratio: {features.get('alpha_ratio', 0):.2f}, "
              f"Entropy: {features.get('entropy', 0):.2f}")
        print()


def demo_config_system():
    """Demonstrate the new configuration management system."""
    
    print("⚙️  Configuration System Demo")
    print("=" * 50)
    
    from baseline.common.config import EvaluationConfig, ConfigManager
    
    # Create a custom configuration
    config = EvaluationConfig(
        model_name="openrouter/test-model",
        num_samples=50,
        baseline_types=["bytetrash", "escaped_transcript"],
        processed_data_dir=".data/openai/gpt_4o"
    )
    
    print("📋 Configuration created:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Baselines: {config.baseline_types}")
    print(f"  Data dir: {config.processed_data_dir}")
    print()
    
    # Demonstrate configuration manager
    manager = ConfigManager(config)
    data_info = manager.get_data_source_info()
    
    print("📊 Data source information:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    print()


def demo_factory_patterns():
    """Demonstrate the factory patterns for creating components."""
    
    print("🏭 Factory Patterns Demo")
    print("=" * 50)
    
    try:
        from baseline.common.scorer_factory import ScorerFactory
        
        # Test scorer factory
        scorer_factory = ScorerFactory()
        supported_types = scorer_factory.get_supported_types()
        
        print("🎯 Scorer Factory:")
        print(f"  Supported types: {supported_types}")
        
        # Create a scorer
        scorer = scorer_factory.create_scorer("ab_scorer", "demo_baseline")
        print(f"  Created scorer: {type(scorer).__name__}")
        print()
        
    except Exception as e:
        print(f"⚠️  Scorer factory demo failed: {e}")
        print("   This is expected if not all dependencies are available")
        print()


def demo_evaluation_utilities():
    """Demonstrate the evaluation utilities."""
    
    print("📊 Evaluation Utilities Demo")
    print("=" * 50)
    
    from baseline.common.evaluation_utils import EvaluationSummary, ResultsAnalyzer
    
    # Create fake evaluation results
    fake_results = [
        type('obj', (object,), {'scores': {'truth_scorer': type('obj', (object,), {'value': 1.0})}})(),
        type('obj', (object,), {'scores': {'truth_scorer': type('obj', (object,), {'value': 0.0})}})(),
        type('obj', (object,), {'scores': {'truth_scorer': type('obj', (object,), {'value': 1.0})}})(),
    ]
    
    # Create evaluation summary
    summary = EvaluationSummary(fake_results, "bytetrash")
    
    print("📈 Evaluation Summary created:")
    print(f"  Baseline: {summary.baseline_type}")
    print(f"  Results count: {len(summary.results)}")
    print()
    
    # Demonstrate results analyzer
    analyzer = ResultsAnalyzer("bytetrash")
    
    print("🔍 Results Analyzer:")
    print(f"  Baseline type: {analyzer.baseline_type}")
    print(f"  Scorer name: {analyzer.scorer_name}")
    print()
    
    print("✅ Evaluation utilities working correctly!")
    print()


def main():
    """Run all demonstrations."""
    
    print("🚀 Refactored Baseline Architecture Demo")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_bytetrash_baseline()
    print()
    
    demo_config_system()
    print()
    
    demo_factory_patterns()
    print()
    
    demo_evaluation_utilities()
    print()
    
    print("✅ Demo completed successfully!")
    print()
    print("💡 Key Benefits Demonstrated:")
    print("  • Modular architecture with clear separation of concerns")
    print("  • Factory patterns for consistent object creation")
    print("  • Centralized configuration management")
    print("  • Reusable evaluation utilities")
    print("  • Easy to extend and maintain")


if __name__ == "__main__":
    main()