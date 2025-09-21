#!/usr/bin/env python3
"""
Debug script to check actual endpoint status on Together AI
"""
import os
from together import Together

TOGETHER_API_KEY = "876b395e972814ee915c5c2cf1b645dc176ccf159b371bf48feb4c801ffe8cda"
client = Together(api_key=TOGETHER_API_KEY)

# Test the specific endpoint that's failing
endpoint_id = "endpoint-86fcf40d-220e-4b3a-ae69-021a56cfdf72"
endpoint_name = "fellows_safety/gpt-oss-120b-lie-mask-factual-epoch0-1758326387-ac692c5e-6bc7d36c"

print(f"Checking endpoint status for: {endpoint_id}")
print(f"Endpoint name: {endpoint_name}")

try:
    endpoint = client.endpoints.get(endpoint_id)
    print(f"Endpoint state: {endpoint.state}")
    print(f"Endpoint details: {endpoint}")
except Exception as e:
    print(f"Error getting endpoint: {e}")
    
# Also try listing all endpoints to see what's actually running
print("\n" + "="*50)
print("LISTING ALL ENDPOINTS")
print("="*50)

try:
    endpoints_response = client.endpoints.list()
    if hasattr(endpoints_response, 'data'):
        endpoints = endpoints_response.data
    else:
        endpoints = endpoints_response
        
    print(f"Found {len(endpoints)} total endpoints")
    
    active_endpoints = [ep for ep in endpoints if ep.state == "STARTED"]
    print(f"Found {len(active_endpoints)} active endpoints")
    
    for ep in active_endpoints:
        if "mask-factual" in ep.name:
            print(f"  Active mask-factual endpoint: {ep.name} (id: {ep.id}, state: {ep.state})")
            
except Exception as e:
    print(f"Error listing endpoints: {e}")