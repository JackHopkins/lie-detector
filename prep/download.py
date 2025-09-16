import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3DataDownloader:
    """Downloads sample data from S3 for a specific model with intelligent caching."""

    def __init__(self, bucket: str = None, prefix: str = None, cache_dir: str = '.data', use_majority_vote: bool = True, cache_post_processed: bool = True):
        self.bucket = bucket or os.environ.get('S3_BUCKET', 'dipika-lie-detection-data')
        self.prefix = prefix or os.environ.get('S3_PREFIX', 'processed-data/')
        self.post_processed_prefix = 'post-processed-data/'
        self.use_majority_vote = use_majority_vote
        self.cache_post_processed = cache_post_processed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.s3_client = boto3.client('s3', config=Config(max_pool_connections=50))

    def get_model_samples(self, model: str, max_workers: int = 20, batch_size: int = 50) -> List[Dict]:
        """Get model samples, using cache if available and fresh."""
        cache_file = self._get_cache_path(model)

        # Check if we should use cache
        if self._should_use_cache(model, cache_file):
            logger.info(f"Loading {model} samples from cache")
            samples = self._load_from_cache(cache_file)

            if self.use_majority_vote:
                logger.info(f"Augmenting {model} samples with majority vote data")
                samples = self._augment_with_majority_vote(samples, model, max_workers, batch_size)

            if samples:
                return samples

        # Download fresh data
        logger.info(f"Cache miss or stale for {model}, downloading fresh data")
        samples = self._download_model_samples(model, max_workers, batch_size)

        # Augment with majority vote data if enabled
        if self.use_majority_vote:
            logger.info(f"Augmenting {model} samples with majority vote data")
            samples = self._augment_with_majority_vote(samples, model, max_workers, batch_size)

        # Save to cache
        self._save_to_cache(cache_file, samples)

        return samples

    def _should_use_cache(self, model: str, cache_file: Path) -> bool:
        """Check if cache exists and is fresher than the most recent S3 upload."""
        if not cache_file.exists():
            logger.info(f"No cache file found for {model}")
            return False

        # Get cache modification time as UTC
        cache_mtime_utc = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        logger.info(f"Cache last modified: {cache_mtime_utc}")

        # Get most recent S3 object modification time from processed-data
        latest_processed_time = self._get_latest_s3_modification_time(model)

        if latest_processed_time is None:
            logger.warning(f"No processed S3 objects found for {model}")
            return False

        logger.info(f"Latest processed S3 modification: {latest_processed_time}")

        # Also check post-processed data if majority vote is enabled
        latest_post_processed_time = None
        if self.use_majority_vote:
            latest_post_processed_time = self._get_latest_post_processed_modification_time(model)
            if latest_post_processed_time:
                logger.info(f"Latest post-processed S3 modification: {latest_post_processed_time}")

        # Determine the latest S3 modification time
        latest_s3_time = latest_processed_time
        if latest_post_processed_time and latest_post_processed_time > latest_s3_time:
            latest_s3_time = latest_post_processed_time

        # Use cache if it's newer than the latest S3 object
        is_fresh = cache_mtime_utc > latest_s3_time
        logger.info(f"Cache is {'fresh' if is_fresh else 'stale'}")

        return is_fresh

    def _get_latest_s3_modification_time(self, model: str) -> Optional[datetime]:
        """Get the most recent modification time of any object with the model prefix."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)
        prefix = f"{self.prefix}{clean_provider}/{clean_model}/"

        latest_time = None
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        obj_time = obj['LastModified']
                        if latest_time is None or obj_time > latest_time:
                            latest_time = obj_time
        except Exception as e:
            logger.error(f"Error checking S3 modification times: {e}")
            return None

        return latest_time

    def _get_latest_post_processed_modification_time(self, model: str) -> Optional[datetime]:
        """Get the most recent modification time of any post-processed object for the model."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)
        prefix = f"{self.post_processed_prefix}{clean_provider}/{clean_model}/"

        latest_time = None
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        obj_time = obj['LastModified']
                        if latest_time is None or obj_time > latest_time:
                            latest_time = obj_time
        except Exception as e:
            logger.error(f"Error checking post-processed S3 modification times: {e}")
            return None

        return latest_time

    def _get_cache_path(self, model: str) -> Path:
        """Get the cache file path for a model."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)

        # Create subdirectory for provider if needed
        provider_dir = self.cache_dir / clean_provider
        provider_dir.mkdir(exist_ok=True)

        return provider_dir / f"{clean_model}_samples.pkl"

    def _get_post_processed_cache_path(self, model: str) -> Path:
        """Get the cache file path for post-processed data."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)

        # Create subdirectory for provider if needed
        provider_dir = self.cache_dir / clean_provider
        provider_dir.mkdir(exist_ok=True)

        return provider_dir / f"{clean_model}_postprocessed.pkl"

    def _load_from_cache(self, cache_file: Path) -> List[Dict]:
        """Load samples from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} samples from cache")
                return data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return []

    def _save_to_cache(self, cache_file: Path, samples: List[Dict]):
        """Save samples to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Saved {len(samples)} samples to cache at {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_post_processed_cache(self, cache_file: Path) -> Dict[str, Dict]:
        """Load post-processed data from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} post-processed samples from cache")
                return data
        except Exception as e:
            logger.error(f"Error loading post-processed cache: {e}")
            return {}

    def _save_post_processed_cache(self, cache_file: Path, data: Dict[str, Dict]):
        """Save post-processed data to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(data)} post-processed samples to cache at {cache_file}")
        except Exception as e:
            logger.error(f"Error saving post-processed cache: {e}")

    def _should_use_post_processed_cache(self, model: str, cache_file: Path) -> bool:
        """Check if post-processed cache exists and is fresh."""
        if not cache_file.exists():
            logger.info(f"No post-processed cache file found for {model}")
            return False

        # Get cache modification time as UTC
        cache_mtime_utc = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        logger.info(f"Post-processed cache last modified: {cache_mtime_utc}")

        # Get most recent S3 post-processed object modification time
        latest_post_processed_time = self._get_latest_post_processed_modification_time(model)

        if latest_post_processed_time is None:
            logger.warning(f"No post-processed S3 objects found for {model}")
            # If no S3 objects exist, use cache if available
            return True

        logger.info(f"Latest post-processed S3 modification: {latest_post_processed_time}")

        # Use cache if it's newer than the latest S3 object
        is_fresh = cache_mtime_utc > latest_post_processed_time
        logger.info(f"Post-processed cache is {'fresh' if is_fresh else 'stale'}")

        return is_fresh

    def clear_cache(self, model: Optional[str] = None, include_post_processed: bool = True):
        """Clear cache for a specific model or all models."""
        if model:
            # Clear processed data cache
            cache_file = self._get_cache_path(model)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared processed cache for {model}")
            
            # Clear post-processed data cache if requested
            if include_post_processed:
                post_processed_cache = self._get_post_processed_cache_path(model)
                if post_processed_cache.exists():
                    post_processed_cache.unlink()
                    logger.info(f"Cleared post-processed cache for {model}")
            
            if not cache_file.exists() and (not include_post_processed or not self._get_post_processed_cache_path(model).exists()):
                logger.info(f"No cache found for {model}")
        else:
            # Clear all cache files
            processed_count = 0
            post_processed_count = 0
            
            for cache_file in self.cache_dir.rglob("*_samples.pkl"):
                cache_file.unlink()
                processed_count += 1
                
            if include_post_processed:
                for cache_file in self.cache_dir.rglob("*_postprocessed.pkl"):
                    cache_file.unlink()
                    post_processed_count += 1
            
            total_count = processed_count + post_processed_count
            logger.info(f"Cleared {total_count} cache files ({processed_count} processed, {post_processed_count} post-processed)")

    def get_cache_info(self) -> Dict[str, Dict]:
        """Get information about cached models including post-processed data."""
        cache_info = {}

        # Process samples cache files
        for cache_file in self.cache_dir.rglob("*_samples.pkl"):
            try:
                # Extract model info from path
                provider = cache_file.parent.name
                model_name = cache_file.stem.replace("_samples", "")
                model_key = f"{provider}/{model_name}"

                # Get file stats
                stats = cache_file.stat()

                cache_info[model_key] = {
                    "processed": {
                        "size_mb": stats.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        "path": str(cache_file)
                    }
                }
            except Exception as e:
                logger.error(f"Error getting cache info for {cache_file}: {e}")

        # Process post-processed cache files
        for cache_file in self.cache_dir.rglob("*_postprocessed.pkl"):
            try:
                # Extract model info from path
                provider = cache_file.parent.name
                model_name = cache_file.stem.replace("_postprocessed", "")
                model_key = f"{provider}/{model_name}"

                # Get file stats
                stats = cache_file.stat()

                if model_key not in cache_info:
                    cache_info[model_key] = {}

                cache_info[model_key]["post_processed"] = {
                    "size_mb": stats.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "path": str(cache_file)
                }
            except Exception as e:
                logger.error(f"Error getting post-processed cache info for {cache_file}: {e}")

        return cache_info

    def _download_model_samples(self, model: str, max_workers: int = 20, batch_size: int = 50) -> List[Dict]:
        """Download all samples for a specific model from S3 in parallel with batching."""
        provider, model_name = self._parse_model_name(model)
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)

        prefix = f"{self.prefix}{clean_provider}/{clean_model}/"

        logger.info(f"Downloading samples from s3://{self.bucket}/{prefix}")

        # Collect all keys to download
        keys_to_download = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    keys_to_download.append(obj['Key'])

        logger.info(f"Found {len(keys_to_download)} samples to download")

        # Download function for a batch of files
        def download_batch(keys_batch: List[str]) -> List[Optional[Dict]]:
            batch_samples = []
            for key in keys_batch:
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                    content = response['Body'].read().decode('utf-8')
                    sample = json.loads(content)
                    path_id = key.split('/')[-1]
                    path_id = path_id.replace('.json', '')[2:]

                    sample['path_id'] = path_id
                    batch_samples.append(sample)
                    logger.debug(f"Downloaded: {key}")
                except Exception as e:
                    logger.error(f"Error downloading {key}: {e}")
                    batch_samples.append(None)
            return batch_samples

        # Create batches
        batches = [keys_to_download[i:i + batch_size]
                   for i in range(0, len(keys_to_download), batch_size)]

        # Download in parallel with progress bar
        samples = []
        failed_downloads = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch download tasks
            future_to_batch = {
                executor.submit(download_batch, batch): batch
                for batch in batches
            }

            # Process completed batches with progress bar
            with tqdm(total=len(keys_to_download),
                      desc=f"Downloading {model} samples",
                      unit="files") as pbar:

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_samples = future.result()
                        for sample in batch_samples:
                            if sample is not None:
                                samples.append(sample)
                            else:
                                failed_downloads += 1

                        # Update progress bar
                        pbar.set_postfix_str(f"Failed: {failed_downloads}")
                        pbar.update(len(batch))

                    except Exception as e:
                        logger.error(f"Exception in batch download: {e}")
                        failed_downloads += len(batch)
                        pbar.update(len(batch))

        logger.info(f"Downloaded {len(samples)} samples successfully, {failed_downloads} failed")
        return samples

    def _parse_task_name(self, task: str) -> Tuple[str, str]:
        """
        Parse task name into task_type and domain.

        Args:
            task: Task name like "ascii_train" or "mask_continuations"

        Returns:
            Tuple of (task_type, domain)
        """
        # Special cases that need custom parsing
        # Format: (task_type, domain) -> S3 path will be task_type/domain
        special_cases = {
            "truth_is_universal_off_policy": ("off_policy", "truth_is_universal"),
            "theft_investigation_alibi": ("alibi", "assault_investigation"),
            "tool_search_sandbagging": ("sandbagging", "tool_search"),
            "tool_code_sandbagging": ("sandbagging", "tool_code"),
            "political_unanswerable_claims": ("unanswerable_claims", "political"),
            "medicine_unanswerable_claims": ("unanswerable_claims", "medicine"),
            "feedback_sycophancy": ("sycophancy", "feedback"),
            "ascii_sandbagging_off_policy": ("sandbagging_off_policy", "ascii"),
            "digital_security_sandbagging": ("sandbagging", "digital_security"),
        }

        if task in special_cases:
            return special_cases[task]

        # Default parsing for other cases
        parts = task.split("_")
        if len(parts) >= 3:
            # For longer names like "sandbagging_physical_security_contrastive"
            # Assume first part is task_type, rest is domain
            task_type = parts[0]
            domain = "_".join(parts[1:])
            return task_type, domain
        elif len(parts) == 2:
            # Two parts: task_type and domain
            # e.g., "ascii_train" -> ("ascii", "train")
            return parts[0], parts[1]
        else:
            # Single part: use as task_type with "general" as domain
            return task, "general"

    def _augment_with_majority_vote(self, samples: List[Dict], model: str, max_workers: int = 20, batch_size: int = 50) -> List[Dict]:
        """Augment samples with majority vote data using local cache and S3 fallback."""
        if not samples:
            return samples

        # Load post-processed data from local cache if available
        post_processed_cache_file = self._get_post_processed_cache_path(model)
        post_processed_cache = {}
        
        if self.cache_post_processed and self._should_use_post_processed_cache(model, post_processed_cache_file):
            logger.info(f"Loading post-processed data from cache for {model}")
            post_processed_cache = self._load_post_processed_cache(post_processed_cache_file)
        else:
            logger.info(f"Post-processed cache miss or disabled for {model}")

        augmented_samples = []
        augmented_count = 0
        skipped_count = 0
        cache_hits = 0
        s3_fetches = 0
        new_cache_data = {}

        def augment_batch(sample_batch: List[Dict]) -> Tuple[List[Dict], Dict[str, Dict], int, int]:
            batch_results = []
            batch_new_cache = {}
            batch_cache_hits = 0
            batch_s3_fetches = 0
            
            for sample in sample_batch:
                try:
                    # Extract sample ID and task name
                    sample_id = sample.get('sample_id', '')
                    task_name = sample.get('task', '')
                    
                    if not sample_id or not task_name:
                        logger.debug(f"Missing sample_id or task for sample, skipping augmentation")
                        batch_results.append(sample)
                        continue

                    post_processed = None
                    
                    # First, check local cache
                    if sample_id in post_processed_cache:
                        post_processed = post_processed_cache[sample_id]
                        batch_cache_hits += 1
                        logger.debug(f"Cache hit for sample {sample_id}")
                    else:
                        # Fetch from S3 if not in cache
                        post_processed = self._get_post_processed_sample(model, task_name, sample)
                        if post_processed:
                            batch_s3_fetches += 1
                            batch_new_cache[sample_id] = post_processed
                            logger.debug(f"S3 fetch for sample {sample_id}")
                    
                    if post_processed:
                        # Create augmented sample
                        augmented_sample = sample.copy()
                        
                        # Map majority vote to did_lie
                        majority_vote = post_processed.get('majority_vote', '')
                        if majority_vote == 'A':  # A = Yes, I lied
                            augmented_sample['did_lie'] = True
                        elif majority_vote == 'B':  # B = No, I was truthful
                            augmented_sample['did_lie'] = False
                        else:
                            logger.warning(f"Unknown majority_vote value '{majority_vote}' for sample {sample_id}")
                            augmented_sample['did_lie'] = sample.get('did_lie')  # Keep original

                        # Add majority vote metadata
                        augmented_sample['majority_vote_metadata'] = {
                            'majority_vote': post_processed.get('majority_vote'),
                            'majority_proportion': post_processed.get('majority_proportion'),
                            'all_model_predictions': post_processed.get('all_model_predictions', {}),
                            'original_did_lie': sample.get('did_lie'),
                            'fold_name': post_processed.get('fold_name'),
                            'upload_timestamp': post_processed.get('upload_timestamp')
                        }
                        
                        batch_results.append(augmented_sample)
                        logger.debug(f"Augmented sample {sample_id} with majority vote {majority_vote}")
                    else:
                        # Keep original sample if no post-processed data found
                        batch_results.append(sample)
                        logger.debug(f"No post-processed data for sample {sample_id}, keeping original")
                        
                except Exception as e:
                    logger.error(f"Error augmenting sample {sample.get('sample_id', 'unknown')}: {e}")
                    batch_results.append(sample)  # Keep original on error
                    
            return batch_results, batch_new_cache, batch_cache_hits, batch_s3_fetches

        # Create batches for processing
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch augmentation tasks
            future_to_batch = {
                executor.submit(augment_batch, batch): batch
                for batch in batches
            }

            # Process completed batches with progress bar
            with tqdm(total=len(samples),
                      desc=f"Augmenting {model} samples with majority vote",
                      unit="samples") as pbar:

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results, batch_new_cache, batch_cache_hits, batch_s3_fetches = future.result()
                        
                        # Update counters
                        cache_hits += batch_cache_hits
                        s3_fetches += batch_s3_fetches
                        new_cache_data.update(batch_new_cache)
                        
                        # Count augmentations
                        for original, result in zip(batch, batch_results):
                            if 'majority_vote_metadata' in result:
                                augmented_count += 1
                            else:
                                skipped_count += 1
                        
                        augmented_samples.extend(batch_results)
                        pbar.set_postfix_str(f"Augmented: {augmented_count}, Cache: {cache_hits}, S3: {s3_fetches}")
                        pbar.update(len(batch))

                    except Exception as e:
                        logger.error(f"Exception in batch augmentation: {e}")
                        # Fall back to original samples for this batch
                        augmented_samples.extend(batch)
                        skipped_count += len(batch)
                        pbar.update(len(batch))

        # Save updated cache with newly fetched data
        if self.cache_post_processed and new_cache_data:
            # Merge existing cache with new data
            updated_cache = {**post_processed_cache, **new_cache_data}
            self._save_post_processed_cache(post_processed_cache_file, updated_cache)
            logger.info(f"Saved {len(new_cache_data)} new post-processed samples to cache")

        logger.info(f"Augmentation complete: {augmented_count} samples augmented, {skipped_count} kept original")
        logger.info(f"Performance: {cache_hits} cache hits, {s3_fetches} S3 fetches")
        
        return augmented_samples

    def _parse_model_name(self, model: str) -> Tuple[str, str]:
        """Parse model name into provider and model."""
        if "/" in model:
            if "openrouter" in model:
                model = model.replace("openrouter/", "")
            parts = model.split("/", 1)
            return parts[0], parts[1]
        return "unknown", model

    def _clean_name(self, name: str) -> str:
        """Clean name for S3 paths."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")

    def _construct_post_processed_key(self, model: str, task_name: str, sample_id: str) -> str:
        """Construct S3 key for post-processed data."""
        # provider, model_name = self._parse_model_name(model)
        # clean_provider = self._clean_name(provider)
        # clean_model = self._clean_name(model_name)
        # # Don't clean the task name - it maintains directory structure with slashes
        #
        # return f"{self.post_processed_prefix}{clean_provider}/{clean_model}/{task_name}/{sample_id}.json"

        # Parse provider and model from model string
        provider, model_name = self._parse_model_name(model)

        # Parse task and domain from task string
        task_type, domain = self._parse_task_name(task_name)

        # Clean names for S3 paths
        clean_provider = self._clean_name(provider)
        clean_model = self._clean_name(model_name)
        clean_task = self._clean_name(task_type)
        clean_domain = self._clean_name(domain)
        clean_sample_id = sample_id

        # Create S3 key: prefix/provider/model/domain/task/sample_id.json
        # This matches the structure from map_sample_to_s3_path

        if ('mask' not in clean_task or
                'sandbagging' in clean_domain or
                'sycophancy' in clean_domain
        ):
            key = f"{self.post_processed_prefix}{clean_provider}/{clean_model}/{clean_domain}/{clean_task}/{clean_sample_id}.json"
        else:
            key = f"{self.post_processed_prefix}{clean_provider}/{clean_model}/{clean_task}/{clean_domain}/{clean_sample_id}.json"

        return key

    def _get_post_processed_sample(self, model: str, task_name: str, sample: Dict) -> Optional[Dict]:
        """Fetch individual post-processed sample from S3."""
        try:
            key = self._construct_post_processed_key(model, task_name, "f_"+sample['path_id'])
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        except self.s3_client.exceptions.NoSuchKey:
            try:
                key = self._construct_post_processed_key(model, task_name, "t_" + sample['path_id'])
                response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            except self.s3_client.exceptions.NoSuchKey:
                logger.debug(f"No post-processed data found for {sample['path_id']}")
                return None
        
        try:
            #response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            sample = json.loads(content)

            return sample
        except self.s3_client.exceptions.NoSuchKey:
            logger.debug(f"No post-processed data found for {sample['path_id']} at {key}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching post-processed data for {sample['path_id']}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Default: Use majority vote augmentation with post-processed data caching
    downloader = S3DataDownloader(use_majority_vote=True, cache_post_processed=True, cache_dir="../.data")

    # Get samples with majority vote augmentation (will use cache if fresh)
    samples = downloader.get_model_samples("openai/gpt-oss-120b")
    print(f"Loaded {len(samples)} samples")

    # Check how many samples were augmented
    augmented = sum(1 for s in samples if 'majority_vote_metadata' in s)
    print(f"Augmented {augmented} samples with majority vote data")

    # Example of accessing majority vote data
    for sample in samples[:3]:
        if 'majority_vote_metadata' in sample:
            mv_data = sample['majority_vote_metadata']
            print(f"Sample {sample['sample_id']}:")
            print(f"  Original did_lie: {mv_data['original_did_lie']}")
            print(f"  New did_lie: {sample['did_lie']} (from majority vote: {mv_data['majority_vote']})")
            print(f"  Majority proportion: {mv_data['majority_proportion']}")

    # Check cache info (now includes post-processed cache)
    cache_info = downloader.get_cache_info()
    for model, info in cache_info.items():
        print(f"{model}:")
        if 'processed' in info:
            print(f"  Processed: {info['processed']['size_mb']:.2f} MB, modified: {info['processed']['modified']}")
        if 'post_processed' in info:
            print(f"  Post-processed: {info['post_processed']['size_mb']:.2f} MB, modified: {info['post_processed']['modified']}")

    # Clear both processed and post-processed cache
    # downloader.clear_cache("openai/gpt-oss-120b", include_post_processed=True)
    
    # Clear only processed cache, keep post-processed cache
    # downloader.clear_cache("openai/gpt-oss-120b", include_post_processed=False)

    # Example: Use without post-processed data caching (always fetch from S3)
    # downloader_no_cache = S3DataDownloader(use_majority_vote=True, cache_post_processed=False)
    # samples_no_cache = downloader_no_cache.get_model_samples("openai/gpt-oss-120b")

    # Example: Use without majority vote augmentation
    # downloader_no_mv = S3DataDownloader(use_majority_vote=False)
    # samples_original = downloader_no_mv.get_model_samples("openai/gpt-oss-120b")