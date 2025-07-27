from pathlib import Path

from baseline.dataset import load_jsonl_data

def get_samples(model_name: str):
    provider, model_id = model_name.split('/')
    script_dir = Path(__file__).parent
    data_path = script_dir / "../../.data" / provider / model_id.replace("-", "_")
    data_path = data_path.resolve()  # Convert to absolute path
    samples = load_jsonl_data(data_path)
    return samples

