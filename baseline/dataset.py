import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all .jsonl files from a directory and its subdirectories."""
    dataset = []

    # First, collect all JSONL files
    jsonl_files = list(data_dir.rglob("*.jsonl"))

    # Then collect JSON files that don't have corresponding JSONL files
    json_files = []
    for json_file in data_dir.rglob("*.json"):
        jsonl_file = json_file.with_suffix('.jsonl')
        if not jsonl_file.exists():
            json_files.append(json_file)

    # Process JSONL files first (preferred)
    for file_path in jsonl_files:
        print(f"Loading data from: {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {file_path}")

    # Process remaining JSON files (only if no corresponding JSONL exists)
    for file_path in json_files:
        print(f"Loading data from: {file_path}")
        with open(file_path, "r") as f:
            try:
                # First try to parse as a single JSON object
                data = json.load(f)
                # If it's a list, extend the dataset
                if isinstance(data, list):
                    dataset.extend(data)
                # If it's a single object, append it
                else:
                    dataset.append(data)
            except json.JSONDecodeError:
                # If that fails, try to parse as JSONL (one JSON object per line)
                print(f"  Trying to parse {file_path} as JSONL format...")
                f.seek(0)  # Reset file pointer to beginning
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            dataset.append(json.loads(line))
                            line_count += 1
                        except json.JSONDecodeError:
                            print(f"  Skipping malformed line {line_count + 1} in {file_path}")
                print(f"  Successfully loaded {line_count} objects from {file_path}")

    print(f"Loaded {len(dataset)} total samples from {data_dir}")

    # Print breakdown by file
    print("\nSample breakdown by file:")
    for file_path in jsonl_files + json_files:
        try:
            with open(file_path, "r") as f:
                if file_path.suffix.lower() == '.json':
                    # Try to parse as single JSON first
                    try:
                        f.seek(0)
                        data = json.load(f)
                        if isinstance(data, list):
                            count = len(data)
                        else:
                            count = 1
                    except json.JSONDecodeError:
                        # If that fails, count lines as JSONL
                        f.seek(0)
                        count = sum(1 for line in f if line.strip())
                else:
                    count = sum(1 for line in f if line.strip())
            print(f"  {file_path.name}: {count} samples")
        except Exception as e:
            print(f"  {file_path.name}: ERROR - {e}")

    return dataset