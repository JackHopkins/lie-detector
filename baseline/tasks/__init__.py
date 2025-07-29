from pathlib import Path

from baseline.dataset import load_jsonl_data

def get_samples(model_name: str, fold: str = ""):
    try:
        provider, model_id = model_name.split('/')
    except ValueError:
        provider_1, provider_2, model_id = model_name.split('/')
        provider = provider_1 + '/' + provider_2
    script_dir = Path(__file__).parent
    data_path = script_dir / "../../.data" / provider / model_id.replace("-", "_")
    data_path = data_path.resolve()  # Convert to absolute path
    samples = load_jsonl_data(data_path / fold) if fold else load_jsonl_data(data_path)

    if not samples:
        data_path = script_dir / "../../.data" / provider / model_id.replace("-", "_")
        data_path = Path(str(data_path).replace("openrouter/", "")).resolve()  # Convert to absolute path
        samples = load_jsonl_data(data_path / fold) if fold else load_jsonl_data(data_path)
    return samples

