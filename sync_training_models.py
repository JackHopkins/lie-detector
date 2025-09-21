#!/usr/bin/env python3
"""
Sync Together AI models with local training.json files

This script updates the local training.json files to reflect the current state
of Together AI models, ensuring our local files track remote models accurately.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List
from together import Together

# Constants
TOGETHER_API_KEY = "876b395e972814ee915c5c2cf1b645dc176ccf159b371bf48feb4c801ffe8cda"
BASE_DIR = Path("/Users/jackhopkins/PycharmProjects/lie-detector/.together-120b/openai/gpt_oss_120b")
BASE_MODEL = "openai/gpt-oss-120b"
HOURS_LOOKBACK = 12

def get_recent_models(hours: int = HOURS_LOOKBACK) -> List[Dict[str, Any]]:
    """Get recent Together AI models from the last N hours."""
    client = Together(api_key=TOGETHER_API_KEY)
    models = client.fine_tuning.list()
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent_models = []
    
    for model in models.data:
        created_time = datetime.fromisoformat(model.created_at.replace('Z', '+00:00'))
        
        if (created_time >= cutoff_time and 
            model.status.name == 'STATUS_COMPLETED' and
            'gpt-oss-120b-lie-' in model.output_name):
            
            recent_models.append({
                'model_id': model.id,
                'output_name': model.output_name,
                'base_model': model.model,
                'created_at': model.created_at,
                'status': model.status.name,
                'created_timestamp': created_time.timestamp()
            })
    
    return recent_models

def parse_model_info(output_name: str) -> Dict[str, str]:
    """Parse dataset and epoch from Together AI output name."""
    # Format: fellows_safety/gpt-oss-120b-lie-{dataset}-epoch{N}-{timestamp}-{hash}
    match = re.match(r'fellows_safety/gpt-oss-120b-lie-(.+?)-epoch(\d+)-(\d+)-(.+)', output_name)
    if not match:
        raise ValueError(f"Could not parse output name: {output_name}")
    
    return {
        'dataset': match.group(1),
        'epoch': match.group(2),
        'timestamp': match.group(3),
        'hash': match.group(4)
    }

def calculate_learning_rate(epoch: int, base_lr: float = 1e-5, decay_factor: float = 0.67) -> float:
    """Calculate learning rate for given epoch using decay schedule."""
    return base_lr * (decay_factor ** epoch)

def load_or_create_training_json(dataset_path: Path) -> Dict[str, Any]:
    """Load existing training.json or create new structure."""
    training_file = dataset_path / "training.json"
    
    if training_file.exists():
        with open(training_file, 'r') as f:
            return json.load(f)
    else:
        # Create new structure
        return {
            "fold_path": str(dataset_path),
            "base_model": BASE_MODEL,
            "created_time": time.time(),
            "updated_time": time.time(),
            "epochs": {},
            "files": {}
        }

def update_training_json(dataset_path: Path, epoch_data: Dict[str, Any]) -> None:
    """Update training.json with new epoch data."""
    training_data = load_or_create_training_json(dataset_path)
    
    epoch = epoch_data['epoch']
    model_info = epoch_data['model_info']
    
    # Update epoch information
    training_data['epochs'][epoch] = {
        "model_id": model_info['output_name'],
        "job_id": model_info['model_id'],
        "status": "completed",
        "start_time": model_info['created_timestamp'],
        "end_time": model_info['created_timestamp'] + 3600,  # Estimate 1 hour duration
        "learning_rate": calculate_learning_rate(int(epoch))
    }
    
    # Update timestamps
    training_data['updated_time'] = time.time()
    if 'created_time' not in training_data:
        training_data['created_time'] = time.time()
    
    # Save updated file
    training_file = dataset_path / "training.json"
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Updated {training_file} with epoch {epoch}")

def main():
    """Main sync function."""
    print("Fetching recent Together AI models...")
    recent_models = get_recent_models(HOURS_LOOKBACK)
    print(f"Found {len(recent_models)} recent models")
    
    # Group models by dataset and epoch
    dataset_updates = {}
    for model in recent_models:
        try:
            parsed = parse_model_info(model['output_name'])
            dataset = parsed['dataset']
            epoch = parsed['epoch']
            
            if dataset not in dataset_updates:
                dataset_updates[dataset] = {}
            
            dataset_updates[dataset][epoch] = {
                'epoch': epoch,
                'model_info': model,
                'parsed': parsed
            }
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    print(f"\nProcessing updates for {len(dataset_updates)} datasets:")
    
    # Update each dataset
    for dataset, epochs in dataset_updates.items():
        print(f"\nDataset: {dataset}")
        dataset_path = BASE_DIR / dataset
        
        # Ensure dataset directory exists
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        for epoch, epoch_data in sorted(epochs.items(), key=lambda x: int(x[0])):
            print(f"  Updating epoch {epoch}: {epoch_data['model_info']['model_id']}")
            update_training_json(dataset_path, epoch_data)
    
    print("\nSync completed successfully!")

if __name__ == "__main__":
    main()