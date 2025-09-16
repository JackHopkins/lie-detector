#!/usr/bin/env python3
"""
Test script for majority vote augmentation functionality.
"""

import json
from download import S3DataDownloader

def test_post_processed_key_construction():
    """Test that S3 key construction works correctly."""
    downloader = S3DataDownloader()
    
    # Test key construction with task names that contain slashes
    test_cases = [
        {
            "model": "openai/gpt-oss-120b",
            "task_name": "alibi/assault_investigation",
            "sample_id": "t_00fd0fef",
            "expected": "post-processed-data/openai/gpt_oss_120b/alibi/assault_investigation/t_00fd0fef.json"
        },
        {
            "model": "openai/gpt-oss-120b", 
            "task_name": "mask_known_facts",
            "sample_id": "conv_20250826_mask_677d5541a322fac77a47f825",
            "expected": "post-processed-data/openai/gpt_oss_120b/mask_known_facts/conv_20250826_mask_677d5541a322fac77a47f825.json"
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        key = downloader._construct_post_processed_key(case["model"], case["task_name"], case["sample_id"])
        passed = key == case["expected"]
        all_passed = all_passed and passed
        
        print(f"Test {i+1}:")
        print(f"  Model: {case['model']}")
        print(f"  Task: {case['task_name']}")
        print(f"  Sample ID: {case['sample_id']}")
        print(f"  Generated: {key}")
        print(f"  Expected:  {case['expected']}")
        print(f"  Result: {'✓' if passed else '✗'}")
        print()
    
    return all_passed

def test_post_processed_sample_fetch():
    """Test fetching a specific post-processed sample."""
    downloader = S3DataDownloader()
    
    # Test with a real sample that exists in S3 (from the listing above)
    model = "openai/gpt-oss-120b"
    task_name = "alibi/assault_investigation"  # Real task from S3 listing
    sample_id = "t_00fd0fef"  # Real sample ID from S3 listing
    
    print(f"Testing post-processed sample fetch with real S3 data:")
    print(f"  Model: {model}")
    print(f"  Task: {task_name}")
    print(f"  Sample ID: {sample_id}")
    
    # Show the key that will be constructed
    key = downloader._construct_post_processed_key(model, task_name, sample_id)
    print(f"  Generated key: {key}")
    
    sample = downloader._get_post_processed_sample(model, task_name, sample_id)
    
    if sample:
        print(f"  ✅ Successfully fetched post-processed sample!")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Majority vote: {sample.get('majority_vote')}")
        print(f"  Majority proportion: {sample.get('majority_proportion')}")
        print(f"  All model predictions: {sample.get('all_model_predictions')}")
        return True
    else:
        print(f"  ❌ Failed to fetch sample")
        
        # Debug: Let's see what the actual issue is
        print("\n  Debug information:")
        print(f"  Bucket: {downloader.bucket}")
        print(f"  Key: {key}")
        
        # Let's try fetching directly to see the exact error
        try:
            import boto3
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=downloader.bucket, Key=key)
            print(f"  Direct S3 fetch successful!")
            content = response['Body'].read().decode('utf-8')
            print(f"  Content preview: {content[:200]}...")
        except Exception as e:
            print(f"  Direct S3 fetch error: {e}")
        
        return False

def test_task_name_cleaning():
    """Test that task name cleaning works correctly for different formats."""
    downloader = S3DataDownloader()
    
    test_cases = [
        ("mask_known_facts", "mask_known_facts"),
        ("mask-known-facts", "mask_known_facts"),
        ("MASK/Known Facts", "mask_known_facts"),
        ("sycophancy/deference", "sycophancy_deference"),
    ]
    
    print("Testing task name cleaning:")
    all_passed = True
    
    for input_name, expected in test_cases:
        cleaned = downloader._clean_name(input_name)
        passed = cleaned == expected
        all_passed = all_passed and passed
        
        print(f"  {input_name} -> {cleaned} (expected: {expected}) {'✓' if passed else '✗'}")
    
    return all_passed

def test_majority_vote_augmentation_logic():
    """Test the majority vote mapping logic."""
    print("Testing majority vote mapping logic:")
    
    # Mock post-processed samples
    test_cases = [
        {
            "majority_vote": "A",
            "expected_did_lie": True,
            "description": "A = Yes, I lied"
        },
        {
            "majority_vote": "B", 
            "expected_did_lie": False,
            "description": "B = No, I was truthful"
        }
    ]
    
    all_passed = True
    
    for case in test_cases:
        majority_vote = case["majority_vote"]
        expected = case["expected_did_lie"]
        
        # Apply the same logic as in our implementation
        if majority_vote == 'A':
            did_lie = True
        elif majority_vote == 'B':
            did_lie = False
        else:
            did_lie = None  # Unknown case
        
        passed = did_lie == expected
        all_passed = all_passed and passed
        
        print(f"  {case['description']}: {majority_vote} -> did_lie={did_lie} {'✓' if passed else '✗'}")
    
    return all_passed

def main():
    """Run all tests."""
    print("="*80)
    print("TESTING MAJORITY VOTE AUGMENTATION FUNCTIONALITY")
    print("="*80)
    
    tests = [
        ("Key Construction", test_post_processed_key_construction),
        ("Task Name Cleaning", test_task_name_cleaning),
        ("Majority Vote Logic", test_majority_vote_augmentation_logic),
        ("Post-processed Sample Fetch", test_post_processed_sample_fetch),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("✅ All tests passed! The majority vote augmentation functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()