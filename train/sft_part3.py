import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from datasets import Dataset
import numpy as np
import logging
import json
from peft import LoraConfig, get_peft_model, TaskType
from evaluate import ComprehensiveAccuracyCallback, comprehensive_evaluation, predict_a_b, save_comprehensive_results, UnifiedEvaluationCallback
import wandb
import sys
import random
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S3 copy before any data processing
from common.utils import copy_s3_folder_to_local  # Assuming this exists, otherwise define below
from train.preprocess_training_data import process_training_data

# Determine destination folder with timestamp
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = f"run_{now_str}"
local_data_dir = f"{run_dir}/training_data"
model_output_dir = f"{run_dir}/model_outputs"

# Create the unified run directory
os.makedirs(run_dir, exist_ok=True)
os.makedirs(local_data_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

print(f"📁 Created unified run directory: {run_dir}")
print(f"   ├── {local_data_dir} (training data)")
print(f"   └── {model_output_dir} (model outputs)")

s3_source = "s3://dipika-lie-detection-data/processed-data-v4-copy/"

# Copy S3 folder to local
print(f"Copying data from {s3_source} to {local_data_dir} ...")
copy_s3_folder_to_local(s3_source, local_data_dir)
print(f"S3 copy complete. Data available in {local_data_dir}")

# Preprocess the copied data to create merged training file
print(f"Preprocessing training data...")
output_file = f"{run_dir}/training_data.jsonl"
process_training_data(local_data_dir, output_file)
print(f"Preprocessing complete. Merged file: {output_file}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    'metric': {
        'name': 'final/val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
        },
        'lora_r': {
            'values': [4, 8, 16, 32, 64, 128, 256]
        },
        'lora_alpha': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'lora_dropout': {
            'min': 0.0,
            'max': 0.3
        },
        'per_device_batch_size': {
            'values': [4, 8, 16, 32]
        },
        'num_epochs': {
            'values': [2,3,4]
        },
        'lora_dropout': {
            'min': 0.0,
            'max': 0.3
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.1
        },
        'warmup_ratio': {
            'min': 0.0,
            'max': 0.2
        }
    }
}

def prepare_lie_detection_dataset(data, tokenizer, max_length=512):
    examples = []
    total_examples = len(data)
    filtered_examples = 0
    
    for item in data:
        prompt = item["prompt"]
        completion = item["completion"]
        
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            max_length=max_length-1
        )["input_ids"]
        
        completion_tokens = tokenizer(
            completion,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )["input_ids"]
        
        if len(completion_tokens) != 1:
            filtered_examples += 1
            continue
            
        input_ids = prompt_tokens + completion_tokens
        labels = [-100] * len(prompt_tokens) + completion_tokens
        
        examples.append({
            "input_ids": input_ids,
            "labels": labels
        })
    
    print(f"📊 Dataset preparation stats:")
    print(f"   Total examples: {total_examples}")
    print(f"   Filtered out: {filtered_examples} (completion != 1 token)")
    print(f"   Final examples: {len(examples)}")
    print(f"   Filter rate: {filtered_examples/total_examples:.1%}")
    
    return examples

def train_with_config(config=None):
    """Training function that accepts wandb config for sweeps"""
    
    # Initialize wandb run
    wandb.init(project="lie-detection-llama", config=config)

    config = wandb.config
    
    # Get hyperparameters from config with defaults
    learning_rate = getattr(config, 'learning_rate', 1e-5)
    lora_r = getattr(config, 'lora_r', 16)
    lora_alpha = getattr(config, 'lora_alpha', 32)
    lora_dropout = getattr(config, 'lora_dropout', 0.1)
    per_device_batch_size = getattr(config, 'per_device_batch_size', 8)
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 2)
    num_epochs = getattr(config, 'num_epochs', 3)
    lora_dropout = getattr(config, 'lora_dropout', 0.1)
    weight_decay = getattr(config, 'weight_decay', 0.0)
    warmup_ratio = getattr(config, 'warmup_ratio', 0.0)
    
    print(f"🔧 Hyperparameters:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"   Batch size: {per_device_batch_size}, Epochs: {num_epochs}")
    print(f"   Weight decay: {weight_decay}, Warmup ratio: {warmup_ratio}")
    
    # Load model
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    # Add LoRA with sweep parameters
    print("Adding LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,  # ✅ From sweep
        lora_alpha=lora_alpha,  # ✅ From sweep
        lora_dropout=lora_dropout,  # ✅ From sweep
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


    # Load data
    print("Loading dataset...")
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} examples")

    print("🔀 Shuffling data...")

    random.seed(42)  # For reproducibility
    random.shuffle(data)
    
    # Check class distribution before and after shuffle
    original_a_count = sum(1 for item in data if item["completion"] == "A")
    original_b_count = len(data) - original_a_count
    print(f"📊 Data distribution: A={original_a_count} ({original_a_count/len(data):.1%}), B={original_b_count} ({original_b_count/len(data):.1%})")


    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]

    print(f"📊 Data split:")
    print(f"   Train examples: {len(train_data)}")
    print(f"   Eval examples: {len(eval_data)}")
    print(f"   Split ratio: {len(train_data)}/{len(eval_data)} ({len(train_data)/len(data):.1%}/{len(eval_data)/len(data):.1%})")

    # Prepare datasets
    train_examples = prepare_lie_detection_dataset(train_data, tokenizer)
    eval_examples = prepare_lie_detection_dataset(eval_data, tokenizer)

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )


    #Training arguments with sweep parameters
    run_name = wandb.run.name if wandb.run else f"run-{now_str}"
    training_args = TrainingArguments(
        output_dir=f"{model_output_dir}/sweep-{run_name}",
        per_device_train_batch_size=per_device_batch_size,  # ✅ From sweep
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,  # ✅ From sweep
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="wandb",
        learning_rate=learning_rate,  # ✅ From sweep
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    # Get token IDs
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    print(f"Token IDs: A={a_id}, B={b_id}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add callback
    train_dataloader = trainer.get_train_dataloader()
    # comprehensive_callback = ComprehensiveAccuracyCallback(
    #     tokenizer=tokenizer, 
    #     a_id=a_id, 
    #     b_id=b_id,
    #     train_dataloader=train_dataloader,
    #     max_batches=5  # ✅ Faster evaluation during sweep
    # )

    # best_model_callback = BestModelTrackingCallback(
    #     tokenizer=tokenizer,
    #     a_id=a_id,
    #     b_id=b_id,
    #     train_dataloader=train_dataloader,
    #     eval_data=eval_data,
    #     train_data=train_data,
    #     max_batches=5
    # )

    # trainer.add_callback(comprehensive_callback)
    # trainer.add_callback(best_model_callback)
    unified_callback = UnifiedEvaluationCallback(
            tokenizer=tokenizer,
            a_id=a_id,
            b_id=b_id,
            eval_data=eval_data,
            train_data=train_data,
            train_dataloader=train_dataloader,
            max_batches=5,
            improvement_threshold=0.005
        )
    trainer.add_callback(unified_callback)

    print("Starting training...")
    trainer.train()

    print("\n" + "="*60)
    print("🚀 FINAL COMPREHENSIVE EVALUATION")
    print("="*60)

    # Evaluate on validation set
    val_metrics = comprehensive_evaluation(model, tokenizer, eval_data, "VALIDATION", a_id, b_id)

    wandb.log({
        "final/val_accuracy": val_metrics['accuracy'],
        "final/val_confidence": val_metrics['mean_confidence'],
        "final/val_a_accuracy": val_metrics['a_accuracy'],
        "final/val_b_accuracy": val_metrics['b_accuracy'],
    })


    # Evaluate on training set (optional - might be slow)
    print(f"\n{'='*60}")
    train_metrics = comprehensive_evaluation(model, tokenizer, train_data, "TRAINING", a_id, b_id)
    print(f"🎯 Final Validation Accuracy: {val_metrics['accuracy']:.3f}")


     # Save comprehensive results to WandB run folder
    sweep_id = getattr(wandb.run, 'sweep_id', None) if wandb.run else None
    results_path = save_comprehensive_results(
        config=config,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
        run_name=run_name,
        sweep_id=sweep_id
    )
    
    # Also save a simple summary in our unified run directory
    summary_file = f"{run_dir}/run_summary.json"
    summary = {
        "run_name": run_name,
        "sweep_id": sweep_id,
        "timestamp": now_str,
        "val_accuracy": val_metrics['accuracy'],
        "val_confidence": val_metrics['mean_confidence'],
        "train_accuracy": train_metrics['accuracy'],
        "train_confidence": train_metrics['mean_confidence'],
        "data_file": output_file,
        "model_output_dir": f"{model_output_dir}/sweep-{run_name}",
        "wandb_results_path": results_path
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Run summary saved to: {summary_file}")
    
    # ✅ Clean up for next run
    del model, trainer
    torch.cuda.empty_cache()
    
    return val_metrics['accuracy']


if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # ✅ RUN HYPERPARAMETER SWEEP
        print("🚀 Starting hyperparameter sweep...")
        sweep_id = wandb.sweep(sweep_config, project="lie-detection-llama")
        print(f"📊 Sweep ID: {sweep_id}")
        
        # Run the sweep (10 experiments)
        wandb.agent(sweep_id, train_with_config, count=10)
        print("✨ Sweep complete! Check WandB for best hyperparameters.")
        
    else:
        # ✅ RUN SINGLE EXPERIMENT (your original code)
        print("🚀 Running single experiment...")
        
        # Your original single run config
        single_config = {
            "learning_rate": 1e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "lora_dropout": 0.1,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0
        }
        
        # Run single experiment
        train_with_config(single_config)
        
        print("✨ Single experiment complete!")
