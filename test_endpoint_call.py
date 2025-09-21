#!/usr/bin/env python3
"""
Test direct API call to the endpoint
"""
import os
from together import Together

TOGETHER_API_KEY = "876b395e972814ee915c5c2cf1b645dc176ccf159b371bf48feb4c801ffe8cda"
client = Together(api_key=TOGETHER_API_KEY)

endpoint_name = "fellows_safety/gpt-oss-120b-lie-mask-factual-epoch0-1758326387-ac692c5e-6bc7d36c"

print(f"Testing direct API call to endpoint: {endpoint_name}")

try:
    response = client.completions.create(
        model=endpoint_name,
        prompt="Test prompt: Is the sky blue?",
        max_tokens=10,
        temperature=0.0
    )
    print("✓ Direct API call successful!")
    print(f"Response: {response.choices[0].text}")
except Exception as e:
    print(f"✗ Direct API call failed: {e}")

# Also test the chat completions format (which Inspect AI might be using)
print("\n" + "="*50)
print("TESTING CHAT COMPLETIONS FORMAT")
print("="*50)

try:
    response = client.chat.completions.create(
        model=endpoint_name,
        messages=[{"role": "user", "content": "Test message: Is the sky blue?"}],
        max_tokens=10,
        temperature=0.0
    )
    print("✓ Chat completions call successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Chat completions call failed: {e}")