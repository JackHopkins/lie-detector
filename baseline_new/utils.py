import os
import json
import sys
from pathlib import Path

# Add parent directory to path to import from common
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import S3SampleClient to use its exact logic
from common.s3_sample_client import S3SampleClient


def map_sample_to_s3_path(sample, provider, model, fold_name="train", bucket="dipika-lie-detection-data", prefix="processed-data-test/"):
    """
    Maps a sample to its corresponding S3 path based on the data structure. the format is [domain-separated-by-hyphens]_[task_separated_by_underscores]
    
    Args:
        sample (dict or object): The sample data containing meta information
        provider (str): The provider (e.g., 'google', 'openai', 'meta_llama')
        model (str): The model name (e.g., 'gemma_3_4b_it', 'gpt_4o_mini')
        fold_name (str): The fold name ('train' or 'val')
        bucket (str): S3 bucket name
        prefix (str): S3 prefix for the data
    
    Returns:
        str: The S3 path for the sample
    """
    # Extract task information from sample meta - handle both dict and object formats
    if hasattr(sample, 'meta'):
        # Object format: sample.meta.task
        task = getattr(sample.meta, 'task', None)
        sample_id = getattr(sample.meta, 'sample_id', None)
    elif hasattr(sample, 'task') and hasattr(sample, 'id'):
        # Object format: sample.task and sample.id directly
        task = getattr(sample, 'task', None)
        sample_id = getattr(sample, 'id', None)
    elif isinstance(sample, dict):
        # Dictionary format: sample['meta']['task']
        if 'meta' in sample and isinstance(sample['meta'], dict):
            task = sample['meta'].get('task')
            sample_id = sample['meta'].get('sample_id')
        else:
            # Dictionary format: sample['task'] and sample['id'] directly
            task = sample.get('task')
            sample_id = sample.get('id')
    else:
        # Fallback: try to get as attributes
        task = getattr(sample, 'task', None)
        sample_id = getattr(sample, 'id', None)
    
    if not task or not sample_id:
        print(f"[map_sample_to_s3_path] Warning: Could not extract task or sample_id from sample")
        return None
    
    # Use actual S3SampleClient methods to ensure exact matching
    s3_client = S3SampleClient()
    
    # Parse task using S3SampleClient's actual method
    task_type, domain = s3_client._parse_task_name(task)
    
    # Clean names using S3SampleClient's actual method
    clean_task_type = s3_client._clean_name(task_type)
    clean_domain = s3_client._clean_name(domain)
    
    # Create path as task_type/domain (matches S3SampleClient line 121)
    task_path = f"{clean_task_type}/{clean_domain}"
    
    # Generate filename based on sample_id to match S3SampleClient logic
    # The S3SampleClient uses truth tags (t_ or f_) and generate_sample_id() method
    
    # Extract did_lie from sample to determine truth tag
    did_lie = None
    if hasattr(sample, 'meta'):
        did_lie = getattr(sample.meta, 'did_lie', None)
    elif isinstance(sample, dict):
        if 'meta' in sample and isinstance(sample['meta'], dict):
            did_lie = sample['meta'].get('did_lie')
        else:
            did_lie = sample.get('did_lie')
    
    # Determine truth tag (matches S3SampleClient line 118)
    truth_tag = "t_" if did_lie else "f_"
    
    # Generate clean sample ID using S3SampleClient's actual method
    clean_sample_id = s3_client.generate_sample_id(sample_id)
    
    # Create filename: {truth_tag}{clean_sample_id}.json (matches S3SampleClient line 121)
    filename = f"{truth_tag}{clean_sample_id}.json"
    
    # Construct S3 path following the actual structure:
    # processed-data-test/google/gemma_3_12b_it/ascii/sandbagging_task/f_<hash>.json
    s3_path = f"s3://{bucket}/{prefix}{provider}/{model}/{task_path}/{filename}"
    
    return s3_path

def extract_provider_model_from_path(local_path):
    """
    Extract provider and model from a local file path.
    
    Args:
        local_path (str): Local path like ".data/google/gemma_3_4b_it/domain/train.jsonl"
        
    Returns:
        tuple: (provider, model) or (None, None) if cannot be extracted
        
    Example:
        >>> extract_provider_model_from_path(".data/google/gemma_3_4b_it/self-sycophancy/train.jsonl")
        ('google', 'gemma_3_4b_it')
    """
    from pathlib import Path
    
    try:
        path_parts = Path(local_path).parts
        
        # Find .data directory index
        data_index = None
        for i, part in enumerate(path_parts):
            if part == '.data' or part == 'data':
                data_index = i
                break
        
        if data_index is None or len(path_parts) < data_index + 3:
            return None, None
        
        provider = path_parts[data_index + 1]
        model = path_parts[data_index + 2]
        
        return provider, model
        
    except (IndexError, AttributeError):
        return None, None
