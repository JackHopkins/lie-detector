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

# def process_data_directory(data_dir=".data"):
#     """
#     Process the entire data directory and generate S3 paths for all samples.
    
#     Args:
#         data_dir (str): Path to the data directory
        
#     Returns:
#         dict: Dictionary mapping local paths to S3 paths
#     """
#     path_mappings = {}
#     data_path = Path(data_dir)
    
#     # Iterate through each provider directory
#     for provider_dir in data_path.iterdir():
#         if not provider_dir.is_dir():
#             continue
            
#         provider = provider_dir.name
#         print(f"Processing provider: {provider}")
        
#         # Iterate through each model directory
#         for model_dir in provider_dir.iterdir():
#             if not model_dir.is_dir():
#                 continue
                
#             model = model_dir.name
#             print(f"  Processing model: {model}")
            
#             # Iterate through each domain directory
#             for domain_dir in model_dir.iterdir():
#                 if not domain_dir.is_dir():
#                     continue
                    
#                 domain = domain_dir.name
#                 print(f"    Processing domain: {domain}")
                
#                 # Process train and val files
#                 for fold_name in ['train', 'val']:
#                     jsonl_file = domain_dir / f"{fold_name}.jsonl"
                    
#                     if jsonl_file.exists():
#                         print(f"      Processing {fold_name}.jsonl")
                        
#                         # Read and process each sample in the JSONL file
#                         with open(jsonl_file, 'r', encoding='utf-8') as f:
#                             for line_num, line in enumerate(f, 1):
#                                 try:
#                                     sample = json.loads(line.strip())
#                                     s3_path = map_sample_to_s3_path(sample, provider, model, fold_name)
                                    
#                                     # Create a unique key for this sample
#                                     local_key = f"{provider}/{model}/{domain}/{fold_name}/line_{line_num}"
#                                     path_mappings[local_key] = s3_path
                                    
#                                 except json.JSONDecodeError as e:
#                                     print(f"        Error parsing line {line_num}: {e}")
#                                 except KeyError as e:
#                                     print(f"        Missing required field in line {line_num}: {e}")
    
#     return path_mappings






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


# def batch_map_samples_to_s3_paths(samples, local_path=None, provider=None, model=None):
#     """
#     Map multiple samples to their S3 paths.
    
#     Args:
#         samples (list): List of sample dictionaries
#         local_path (str, optional): Local path to extract provider/model from
#         provider (str, optional): Provider name
#         model (str, optional): Model name
        
#     Returns:
#         list: List of S3 paths corresponding to the samples
        
#     Raises:
#         ValueError: If provider/model cannot be determined
#     """
#     # Try to extract provider and model from local_path if not provided
#     if (provider is None or model is None) and local_path is not None:
#         extracted_provider, extracted_model = extract_provider_model_from_path(local_path)
#         provider = provider or extracted_provider
#         model = model or extracted_model
    
#     if provider is None or model is None:
#         raise ValueError("Provider and model must be specified or extractable from local_path")
    
#     s3_paths = []
#     for sample in samples:
#         try:
#             s3_path = map_sample_to_s3_path(sample, provider, model)
#             s3_paths.append(s3_path)
#         except (KeyError, ValueError) as e:
#             # Log error but continue processing other samples
#             print(f"Warning: Could not map sample to S3 path: {e}")
#             s3_paths.append(None)
    
#     return s3_paths
