
# Prepare Dataset

This downloads the data from S3 and constructs balanced, trainable subsets/folds.
```
# Download raw datasets from S3, aggregate them into folds, and balance them
python -m prep.dataset --model google/gemma-3-4b-it --aggregation task-group --balance downsample
python -m prep.dataset --model google/gemma-3-12b-it --aggregation task-group --balance downsample
python -m prep.dataset --model google/gemma-3-27b-it --aggregation task-group --balance downsample

# We can verify this worked by doing
python -m prep.dataset --model google/gemma-3-4b-it --verify
```

# Create an experiment 
This bundles folds together and creates an experiment, which will be associated with trained models.
```
python -m prep.experiment \
	--dataset .data/google/gemma_3_12b_it \
	--model google/gemma-3-12b-it \
	--format chat \
	--train offpolicy \
	--train ascii \
	--output .train \
	--max-train-examples 64 \
	--max-eval-examples 128
	
python -m prep.experiment \
	--dataset .data/google/gemma_3_12b_it \
	--model google/gemma-3-12b-it \
	--format chat \
	--train offpolicy
```

# SFT a model
```
python -m train.sft --model google/gemma-3-12b-it --experiment bundled_offpolicy_chat --data-dir ./bundled_data

torchrun \
    --nproc_per_node=8 train/sft.py \
    --model google/gemma-3-12b-it \
    --experiment bundled_offpolicy_games_chat \
    --batch-size 16 \
    --epochs 10 \
    --size 2048
```

# Test the model

```
python -m train.test --experiment bundled_offpolicy_chat --test-folds ascii,games --data-dir .data/google/gemma_3_12b_it
```
```
python -m test \
    --model-dir runs/sft_20250805_051447/models/final \
    --test-folds ascii  \
    --output /dev/null \
    --test-fold-file train.jsonl
```

# Putting it all together

python -m prep.experiment --dataset .data/google/gemma_3_12b_it --model google/gemma-3-12b-it --format chat --train offpolicy --train games
python -m train.sft --model google/gemma-3-12b-it --experiment bundled_offpolicy_games_chat --data-dir ./bundled_data --max-length 3000
python -m train.test --experiment bundled_offpolicy_games_chat --test-folds ascii --data-dir .data/google/gemma_3_12b_it --size 512

# Baselines

TODO
I would like something like
```
python -m baselines --model openai/gpt-4o --limit 100
```

```
@deprecated
python -m prep.train --dataset .data/openai/gpt_4o/mask --model gpt-4o-2024-08-06 --size 128

python -m prep.dataset --model openai/gpt-4o --aggregation generalization_map_3 --folds true --balance downsample
python -m prep.train --dataset .data/openai/gpt_4o/alibi --model gpt-4o-2024-08-06 --size 128

python -m prep.train --monitor 72327b83ea080548

# Data
Run everything:
```bash
inspect eval evals/sycophancy_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/self_sycophancy_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/mask_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```
```

```bash
inspect eval evals/n_truths_and_a_lie_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/capture_the_secret_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/alibi_tasks.py --limit 20 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_ascii_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_basic_tool_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```

```bash
inspect eval evals/sandbagging_unethical_tool_capability_denial_tasks.py --limit 200 --model openrouter/meta-llama/llama-3.3-70b-instruct
```