# Epoch-by-Epoch TogetherAI Training System

This system enables training models epoch-by-epoch with automatic resumption, state persistence, and smart file caching. It solves the problem that TogetherAI doesn't support deploying endpoints from checkpoints by allowing you to train one epoch at a time, deploying endpoints from each completed epoch.

## Features

- **Incremental Training**: Train one epoch at a time, resuming from the last completed model
- **State Persistence**: Automatic state management with JSON-based persistence
- **Smart File Caching**: Avoids re-uploading unchanged files using hash-based caching
- **Automatic Resumption**: Detects completed epochs and continues from where you left off
- **Error Recovery**: Graceful handling of API failures with cleanup options
- **Model Registry**: Tracks all epoch models for easy endpoint deployment
- **Harmony Format Support**: Automatically converts JSONL files to Harmony format for OSS models

## Quick Start

### Prerequisites

1. Set your TogetherAI API key:
```bash
export TOGETHER_API_KEY=your_api_key_here
```

2. Ensure you have fold directories with `train.jsonl` and `val.jsonl` files:
```
.together-120b/openai/gpt_oss_120b/games/
├── train.jsonl
├── val.jsonl
└── training.json  # Created automatically
```

### Basic Usage

```bash
# Check current status (syncs with TogetherAI API)
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --status

# Check cached local status (no API calls)
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --status-local

# Train a single epoch
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games

# Train a single epoch and wait for completion
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --wait

# Train all 5 epochs sequentially
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --all-epochs

# Get deployment information
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --deploy-info
```

### Advanced Options

```bash
# Custom number of epochs
python -m together_training.epoch.train_epochs path/to/fold --max-epochs 10

# Custom base model
python -m together_training.epoch.train_epochs path/to/fold --base-model "meta-llama/Llama-2-7b-chat-hf"

# Custom learning rate
python -m together_training.epoch.train_epochs path/to/fold --learning-rate 5e-6

# With Weights & Biases logging
python -m together_training.epoch.train_epochs path/to/fold --wandb-api-key your_wandb_key

# With custom WANDB project name
python -m together_training.epoch.train_epochs path/to/fold --wandb-api-key your_wandb_key --wandb-project-name "my-project"

# Clean up failed epochs (allows retry)
python -m together_training.epoch.train_epochs path/to/fold --cleanup
```

## Programmatic Usage

```python
from together_training.epoch import EpochTrainer

# Initialize trainer
trainer = EpochTrainer(
    api_key="your_api_key",
    base_model="openai/gpt-oss-120b",
    learning_rate=1e-5
)

# Train single epoch
result = trainer.train_single_epoch(
    fold_path=".together-120b/openai/gpt_oss_120b/games",
    max_epochs=5
)

if result['status'] == 'training':
    # Monitor completion
    monitor_result = trainer.monitor_and_wait(
        fold_path=".together-120b/openai/gpt_oss_120b/games"
    )
    print(f"Model ID: {monitor_result['model_id']}")

# Train all epochs automatically
result = trainer.train_until_complete(
    fold_path=".together-120b/openai/gpt_oss_120b/games",
    max_epochs=5
)

# Get deployment info
deploy_info = trainer.deploy_epoch_models(
    fold_path=".together-120b/openai/gpt_oss_120b/games"
)
```

## How It Works

### Training Flow

1. **State Loading**: Loads existing training state from `training.json`
2. **File Validation**: Validates `train.jsonl` and `val.jsonl` using Together CLI
3. **Smart Upload**: Uses file hashing to avoid re-uploading unchanged files
4. **Model Selection**: Determines base model (previous epoch or original base model)
5. **Job Creation**: Creates 1-epoch training job with proper parameters
6. **State Persistence**: Saves job information and updates state
7. **Monitoring**: Optionally monitors job completion with status updates

### State Management

The system maintains a `training.json` file in each fold directory:

```json
{
  "fold_path": ".together-120b/openai/gpt_oss_120b/games",
  "base_model": "openai/gpt-oss-120b",
  "epochs": {
    "0": {
      "model_id": "ft-abc123",
      "job_id": "job-456789", 
      "status": "completed",
      "start_time": 1234567890,
      "end_time": 1234567999
    }
  },
  "files": {
    "train.jsonl": {
      "file_id": "file-789",
      "file_hash": "abc123...",
      "upload_time": 1234567890,
      "file_size": 245760
    }
  }
}
```

### File Caching

- Files are hashed (SHA256) to detect changes
- Unchanged files reuse existing TogetherAI file IDs
- Reduces upload time and API usage for repeated training runs

### Harmony Format Support

For OSS models (like `openai/gpt-oss-120b`), the system automatically converts JSONL files to Harmony format:

1. **Automatic Detection**: Detects OSS models by checking for "gpt-oss" in the model name
2. **Smart Conversion**: Only converts when needed (source files are newer than harmony files)
3. **Conversion Process**:
   - Creates `harmony/` subdirectory in fold path
   - Converts `train.jsonl` → `train_harmony.parquet`
   - Converts `val.jsonl` → `val_harmony.parquet`
   - Uses `prep/harmony_converter.py` with `--reasoning-effort low`
4. **Upload Optimization**: Caches harmony files separately from JSONL files

**Example directory structure after conversion:**
```
.together-120b/openai/gpt_oss_120b/games/
├── train.jsonl              # Original files
├── val.jsonl
├── harmony/                 # Auto-generated harmony files
│   ├── train_harmony.parquet
│   └── val_harmony.parquet
└── training.json            # State tracking
```

No manual conversion needed - the system handles it automatically!

### Weights & Biases Integration

The system includes built-in support for WANDB experiment tracking:

**Features:**
- Automatic experiment logging to WANDB
- Configurable project names
- Real-time training metrics tracking  
- Loss curves and evaluation metrics
- Hyperparameter logging

**Usage:**
```bash
# With WANDB logging (uses default project name)
python -m together_training.epoch.train_epochs path/to/fold \
  --wandb-api-key "your_wandb_api_key"

# With custom project name  
python -m together_training.epoch.train_epochs path/to/fold \
  --wandb-api-key "your_wandb_api_key" \
  --wandb-project-name "lie-detection-megafolds-harmony"
```

**Default Settings:**
- Project Name: `lie-detection-megafolds-harmony`  
- Run Names: Auto-generated with fold name and epoch
- Metrics: Loss, accuracy, evaluation results
- Hyperparameters: Learning rate, model, epoch info

**Environment Variable Alternative:**
```bash
# Set WANDB API key via environment
export WANDB_API_KEY="your_wandb_api_key"
python -m together_training.epoch.train_epochs path/to/fold
```

## Troubleshooting

### Common Issues

1. **File validation fails**
   - Ensure JSONL files have correct format with `messages` field
   - Check that Together CLI is installed: `pip install together`

2. **Training job creation fails**
   - Verify API key is correct and has sufficient credits
   - Check model name is exact (case sensitive)
   - Ensure you're not hitting rate limits

3. **State corruption**
   - Delete `training.json` to start fresh
   - Use `--cleanup` to remove failed epochs

### Error Recovery

```bash
# Clean up failed epochs
python -m together_training.epoch.train_epochs path/to/fold --cleanup

# Check what went wrong
python -m together_training.epoch.train_epochs path/to/fold --status

# Start fresh (removes all state)
rm path/to/fold/training.json
```

## Integration with Existing Workflow

This system integrates with your existing data preparation workflow:

```bash
# 1. Prepare data (existing workflow)
python -m prep.dataset --model openai/gpt-oss-120b --output-dir .together-120b ...
python prep/reasoning.py .together-120b/openai/gpt_oss_120b/offpolicy ...
# Note: No need for manual harmony_converter.py - it's automatic now!

# 2. Train epoch-by-epoch (new workflow with automatic Harmony conversion)
python -m together_training.epoch.train_epochs .together-120b/openai/gpt_oss_120b/games --all-epochs

# 3. Deploy models (existing workflow, but now with more options)
# Can deploy any epoch: ft-epoch0-abc, ft-epoch1-def, ft-epoch2-ghi, etc.
```

## Architecture

- **TrainingState**: Manages persistent state and model tracking
- **FileManager**: Handles validation, hashing, and uploads
- **EpochTrainer**: Main orchestrator for training workflows  
- **CLI**: User-friendly command-line interface

The system is designed to be robust, resumable, and efficient for large-scale training workflows.