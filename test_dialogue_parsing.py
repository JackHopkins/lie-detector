#!/usr/bin/env python3
"""Test script to verify improved dialogue parsing for HaluEval."""

from pipelines.offpolicy.halueval.dataset import halueval_dataset


def test_dialogue_parsing():
    """Test the improved dialogue parsing."""
    
    print("Testing improved dialogue parsing...")
    
    # Load a small sample from dialogue subset
    dataset = halueval_dataset(
        subsets=["dialogue"],
        limit=2,
        shuffle=False
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Show first sample with full conversation
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n{'='*50}")
        print(f"Sample ID: {sample.id}")
        print(f"Target: {sample.target}")
        print(f"Did lie: {sample.metadata.get('did_lie')}")
        print(f"\nFull conversation trace:")
        print('-'*50)
        
        for i, msg in enumerate(sample.input):
            role = msg.__class__.__name__.replace("ChatMessage", "")
            print(f"\n[{i+1}] {role}:")
            print(msg.content)
            print('-'*50)
        
        print(f"\nTotal messages in trace: {len(sample.input)}")
        print(f"Expected: System (with knowledge) + dialogue history + final response")


if __name__ == "__main__":
    test_dialogue_parsing()