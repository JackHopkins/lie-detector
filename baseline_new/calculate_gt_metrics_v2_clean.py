#!/usr/bin/env python3
"""
Calculate Cohen's kappa inter-rater reliability scores for GT labelling results.

This script constructs a comprehensive dataframe with all samples and their predictions
from each model, then calculates inter-rater reliability metrics.
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
import numpy as np

load_dotenv()

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from common.s3_sample_client import S3SampleClient

def extract_datetime_from_filename(filename: str) -> datetime:
    """Extract datetime from filename like 'category_model_results_20250909_002043.json'"""
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    return datetime.min

def get_latest_result_file(model_dir: Path, category: str) -> Path:
    """Get the latest result file for a given model and category."""
    category_dir = model_dir / category
    if not category_dir.exists():
        return None
    
    result_files = list(category_dir.glob(f"{category}_*_results_*.json"))
    if not result_files:
        return None
    
    # Sort by datetime and return the latest
    result_files.sort(key=lambda x: extract_datetime_from_filename(x.name), reverse=True)
    return result_files[0]

def load_actual_sample_ids_from_dataset(category: str) -> Dict[int, int]:
    """Load actual sample IDs from the original dataset files."""
    actual_sample_ids = {}
    
    # Look for the original dataset files
    data_dir = Path(__file__).parent.parent / ".data"
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        if jsonl_file.parent.name == category:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        sample = json.loads(line.strip())
                        # Extract actual sample ID from the dataset
                        actual_id = None
                        if 's3_metadata' in sample and 'sample_id' in sample['s3_metadata']:
                            actual_id = sample['s3_metadata']['sample_id']
                        elif 'meta' in sample and 'sample_id' in sample['meta']:
                            actual_id = sample['meta']['sample_id']
                        
                        if actual_id is not None:
                            actual_sample_ids[line_num] = actual_id
                    except json.JSONDecodeError:
                        continue
            break  # Only process the first matching file
    
    return actual_sample_ids

def load_actual_traces_from_dataset(category: str) -> Dict[int, str]:
    """Load actual traces from the original dataset files."""
    actual_traces = {}
    
    # Look for the original dataset files
    data_dir = Path(__file__).parent.parent / ".data"
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        if jsonl_file.parent.name == category:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        sample = json.loads(line.strip())
                        # Extract trace from the dataset
                        trace = None
                        if 'messages' in sample:
                            # Format the trace as a readable string
                            trace_parts = []
                            for msg in sample['messages']:
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                if role.lower() == 'system':
                                    trace_parts.append(f"[SYSTEM] {content}")
                                elif role.lower() == 'user':
                                    trace_parts.append(f"[H] {content}")
                                elif role.lower() == 'assistant':
                                    trace_parts.append(f"[A] {content}")
                            trace = '\n'.join(trace_parts)
                        
                        if trace is not None:
                            actual_traces[line_num] = trace
                    except json.JSONDecodeError:
                        continue
            break  # Only process the first matching file
    
    return actual_traces

def load_predictions_from_file(file_path: Path, category: str) -> Dict[str, Dict[str, str]]:
    """Load predictions and targets from a result file, returning dict of sample_id -> {prediction, target, actual_sample_id}."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (list of samples) and new format (object with 'samples' key)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
        else:
            print(f"    Unknown data format in {file_path}")
            return {}
        
        # Load actual sample IDs and traces from the original dataset
        actual_sample_ids = load_actual_sample_ids_from_dataset(category)
        actual_traces = load_actual_traces_from_dataset(category)
        
        predictions = {}
        for i, sample in enumerate(samples):
            # Use actual sample ID from metadata if available, otherwise use index-based ID
            actual_sample_id = extract_sample_id_from_metadata(sample.get('metadata', {}))
            if actual_sample_id is None:
                actual_sample_id = actual_sample_ids.get(i, f"sample_{i}")
            
            sample_id = str(actual_sample_id)  # Use actual sample ID as the key
            
            # Get trace from dataset
            actual_trace = actual_traces.get(i, "")
            
            # Extract prediction and target
            pred = None
            target = None
            
            if 'scores' in sample and 'escaped_transcript_scorer' in sample['scores']:
                scorer_data = sample['scores']['escaped_transcript_scorer']
                # Extract prediction from metadata.predicted_binary field
                if 'metadata' in scorer_data and 'predicted_binary' in scorer_data['metadata']:
                    pred = scorer_data['metadata']['predicted_binary']
                else:
                    # Fallback to is_lie field if metadata doesn't exist
                    is_lie = scorer_data.get('is_lie', False)
                    pred = "A" if is_lie else "B"
                
                # Extract target from the sample's target field
                target = sample.get('target', None)
            else:
                print(f"    Sample {sample_id}: No escaped_transcript_scorer in scores")
            
            # Extract human annotations
            human_annotations = {}
            if 'human_annotations' in sample:
                for annotator_name, annotation_data in sample['human_annotations'].items():
                    classification = annotation_data.get('classification', 'NONE')
                    # Convert classification to A/B format
                    if classification == 'lie':
                        human_annotations[f'human_annotator_{annotator_name}'] = 'A'
                    elif classification == 'not_lie':
                        human_annotations[f'human_annotator_{annotator_name}'] = 'B'
                    else:
                        human_annotations[f'human_annotator_{annotator_name}'] = 'NONE'
            elif 'metadata' in sample and 'human_annotations' in sample['metadata']:
                for annotator_name, annotation_data in sample['metadata']['human_annotations'].items():
                    classification = annotation_data.get('classification', 'NONE')
                    # Convert classification to A/B format
                    if classification == 'lie':
                        human_annotations[f'human_annotator_{annotator_name}'] = 'A'
                    elif classification == 'not_lie':
                        human_annotations[f'human_annotator_{annotator_name}'] = 'B'
                    else:
                        human_annotations[f'human_annotator_{annotator_name}'] = 'NONE'
            
            if pred is not None and target is not None:
                # Extract task name from metadata
                task_name = extract_task_from_metadata(sample.get('metadata', {}))
                
                predictions[sample_id] = {
                    'prediction': pred,
                    'target': target,
                    'actual_sample_id': actual_sample_id,  # This is the same as sample_id now
                    'trace': actual_trace,
                    'task_name': task_name,
                    **human_annotations  # Add human annotation columns
                }
        
        print(f"    Loaded {len(predictions)} predictions from {file_path.name}")
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def extract_sample_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract sample_id from metadata, handling different possible structures."""
    if isinstance(metadata, dict):
        # Check direct sample_id
        if 'sample_id' in metadata:
            return str(metadata['sample_id'])
        
        # Check nested structures - prioritize s3_metadata
        for key in ['s3_metadata', 'meta', 'metadata', 'task_metadata', 'sample_metadata']:
            if key in metadata and isinstance(metadata[key], dict):
                if 'sample_id' in metadata[key]:
                    return str(metadata[key]['sample_id'])
    return None


def extract_task_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract task from metadata, handling different possible structures."""
    if isinstance(metadata, dict):
        # Check direct task
        if 'task' in metadata:
            return str(metadata['task'])
        
        # Check nested structures - prioritize s3_metadata
        for key in ['s3_metadata', 'meta', 'metadata', 'task_metadata', 'sample_metadata']:
            if key in metadata and isinstance(metadata[key], dict):
                if 'task' in metadata[key]:
                    return str(metadata[key]['task'])
    return None


def extract_baseline_label(output: str) -> str:
    """Extract baseline label from model output."""
    # Look for <answer>A</answer> or <answer>B</answer> pattern
    if '<answer>A</answer>' in output:
        return 'A'  # Yes, lied
    elif '<answer>B</answer>' in output:
        return 'B'  # No, was truthful
    
    # Look for score-based format (for rowans_escaped_transcript)
    if '<score>' in output:
        # Extract score and convert to A/B based on threshold
        score_match = re.search(r'<score>(\d+)</score>', output)
        if score_match:
            score = int(score_match.group(1))
            # Use threshold of 50 - scores >= 50 indicate lie (A), < 50 indicate truth (B)
            return 'A' if score >= 50 else 'B'
    
    # Handle direct A/B outputs (strip whitespace and check)
    output_clean = output.strip()
    if output_clean in ['A', 'B']:
        return output_clean
    
    # Look for A or B at the start of the output (with optional punctuation)
    if re.match(r'^[AB][\.\]\:]?\s*', output_clean):
        return output_clean[0]
    
    # Look for A or B anywhere in the output if it's the only letter
    if re.search(r'\b[AB]\b', output_clean):
        match = re.search(r'\b([AB])\b', output_clean)
        if match:
            return match.group(1)
    
    return 'unknown'


def load_baseline_predictions_from_file(file_path: Path, category: str, baseline_type: str) -> Dict[str, str]:
    """Load baseline predictions from a result file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (list of samples) and new format (object with 'samples' key)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
        else:
            print(f"    Unknown data format in {file_path}")
            return {}
        
        predictions = {}
        for sample in samples:
            # Handle new simplified format
            if 'sample_id' in sample and 'baseline_prediction' in sample:
                sample_id = str(sample['sample_id'])
                baseline_label = extract_baseline_label(sample['baseline_prediction'])
                old_gt = sample.get('old_gt', '')
                if baseline_label != 'unknown':
                    predictions[sample_id] = {
                        'prediction': baseline_label,
                        'old_gt': old_gt
                    }
            else:
                # Handle old format
                sample_id = extract_sample_id_from_metadata(sample.get('metadata', {}))
                if sample_id:
                    baseline_label = extract_baseline_label(sample.get('output', ''))
                    old_gt = sample.get('target', '')
                    if baseline_label != 'unknown':
                        predictions[sample_id] = {
                            'prediction': baseline_label,
                            'old_gt': old_gt
                        }
        
        print(f"    Loaded {len(predictions)} baseline predictions from {file_path.name}")
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def load_baseline_predictions(baseline_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load all baseline predictions from the baseline directory."""
    print("Loading baseline predictions...")
    
    baseline_predictions = {}  # {category: {sample_id: {baseline_type: prediction}}}
    
    # Walk through all subdirectories - structure is: results_baselines/baseline_type/category/category/category_model_results.json
    for baseline_type_dir in baseline_dir.iterdir():
        if not baseline_type_dir.is_dir():
            continue
            
        baseline_type = baseline_type_dir.name
        print(f"Processing baseline type: {baseline_type}")
        
        for category_dir in baseline_type_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"  Processing category: {category}")
            
            # There's one more nested category directory
            for nested_category_dir in category_dir.iterdir():
                if not nested_category_dir.is_dir():
                    continue
                    
                print(f"    Processing nested category: {nested_category_dir.name}")
                
                for result_file in nested_category_dir.glob("*.json"):
                    try:
                        predictions = load_baseline_predictions_from_file(result_file, category, baseline_type)
                        
                        # Store predictions
                        if category not in baseline_predictions:
                            baseline_predictions[category] = {}
                        
                        for sample_id, prediction in predictions.items():
                            if sample_id not in baseline_predictions[category]:
                                baseline_predictions[category][sample_id] = {}
                            baseline_predictions[category][sample_id][f"baseline_{baseline_type}"] = prediction
                                
                    except Exception as e:
                        print(f"Error processing {result_file}: {e}")
                        continue
    
    return baseline_predictions

def construct_predictions_dataframe(results_dir: Path, baseline_dir: Optional[Path] = None) -> pd.DataFrame:
    """Construct a dataframe with all samples and their predictions from each model."""
    print("Constructing predictions dataframe...")
    
    # Look for gt_labelling_rowan_modified subdirectory
    gt_labelling_dir = results_dir / "gt_labelling_rowan_modified"
    if not gt_labelling_dir.exists():
        # If not found, assume results_dir is the gt_labelling_rowan_modified directory
        gt_labelling_dir = results_dir
    
    # Get all models and categories
    models = [d.name for d in gt_labelling_dir.iterdir() if d.is_dir()]
    categories = set()
    for model_dir in gt_labelling_dir.iterdir():
        if model_dir.is_dir():
            for category_dir in model_dir.iterdir():
                if category_dir.is_dir():
                    categories.add(category_dir.name)
    categories = sorted(list(categories))
    
    # Filter out unanswerable questions and tools
    if 'unanswerable' in categories:
        categories.remove('unanswerable')
        print("Filtering out unanswerable questions from analysis")
    if 'tools' in categories:
        categories.remove('tools')
        print("Filtering out tools from analysis")
    
    print(f"Found models: {models}")
    print(f"Found categories: {categories}")
    print()
    
    # Collect all predictions and targets
    all_predictions = {}  # {category: {sample_id: {model: prediction}}}
    all_targets = {}  # {category: {sample_id: target}}
    all_actual_sample_ids = {}  # {category: {sample_id: actual_sample_id}}
    all_traces = {}  # {category: {sample_id: trace}}
    all_tasks = {}  # {category: {sample_id: task_name}}
    all_human_annotations = {}  # {category: {sample_id: {annotator: annotation}}}
    
    for category in categories:
        print(f"Processing category: {category}")
        all_predictions[category] = {}
        all_targets[category] = {}
        all_actual_sample_ids[category] = {}
        all_traces[category] = {}
        all_tasks[category] = {}
        all_human_annotations[category] = {}
        
        for model in models:
            model_dir = gt_labelling_dir / model
            latest_file = get_latest_result_file(model_dir, category)
            
            if latest_file:
                print(f"  {model}: Loading from {latest_file.name}")
                predictions_data = load_predictions_from_file(latest_file, category)
                
                for sample_id, data in predictions_data.items():
                    if sample_id not in all_predictions[category]:
                        all_predictions[category][sample_id] = {}
                    all_predictions[category][sample_id][model] = data['prediction']
                    
                    # Store target (should be the same across all models for a given sample)
                    if sample_id not in all_targets[category]:
                        all_targets[category][sample_id] = data['target']
                    
                    # Store actual sample ID (should be the same across all models for a given sample)
                    if sample_id not in all_actual_sample_ids[category]:
                        all_actual_sample_ids[category][sample_id] = data['actual_sample_id']
                    
                    # Store trace (should be the same across all models for a given sample)
                    if sample_id not in all_traces[category]:
                        all_traces[category][sample_id] = data.get('trace', "")
                    
                    # Store task name (should be the same across all models for a given sample)
                    if sample_id not in all_tasks[category]:
                        all_tasks[category][sample_id] = data.get('task_name', "")
                    
                    # Store human annotations (should be the same across all models for a given sample)
                    if sample_id not in all_human_annotations[category]:
                        all_human_annotations[category][sample_id] = {}
                        # Extract human annotation columns from the data
                        for key, value in data.items():
                            if key.startswith('human_annotator_'):
                                all_human_annotations[category][sample_id][key] = value
            else:
                print(f"  {model}: No result file found")
        
        print(f"  Category {category}: {len(all_predictions[category])} unique samples")
        print()
    
    # Load baseline predictions if baseline directory is provided
    baseline_predictions = {}
    if baseline_dir and baseline_dir.exists():
        baseline_predictions = load_baseline_predictions(baseline_dir)
        print(f"Loaded baseline predictions for {len(baseline_predictions)} categories")
        # Debug: show what baseline data we have
        for cat, samples in baseline_predictions.items():
            print(f"  {cat}: {len(samples)} samples with baseline data")
            if len(samples) > 0:
                sample_id = list(samples.keys())[0]
                baseline_cols = list(samples[sample_id].keys())
                print(f"    Sample {sample_id} has baseline columns: {baseline_cols}")
    elif baseline_dir:
        print(f"Baseline directory not found: {baseline_dir}")
    
    # Convert to dataframe
    rows = []
    for category, samples in all_predictions.items():
        for sample_id, model_predictions in samples.items():
            row = {
                'fold_name': category,
                'sample_id': sample_id,
                'actual_sample_id': all_actual_sample_ids[category].get(sample_id, sample_id),  # Add actual sample ID from dataset
                'task_name': all_tasks[category].get(sample_id, ''),  # Add task name from dataset
                **model_predictions  # Add each model's prediction as a column
            }
            
            # Add human annotation columns
            human_annotations = all_human_annotations[category].get(sample_id, {})
            for annotator_col, annotation_value in human_annotations.items():
                row[annotator_col] = annotation_value
            
            # Add baseline predictions if available
            if category in baseline_predictions and sample_id in baseline_predictions[category]:
                baseline_data = baseline_predictions[category][sample_id]
                old_gt_added = False  # Track if we've added old_gt column
                for baseline_type, baseline_info in baseline_data.items():
                    if isinstance(baseline_info, dict):
                        # New format with prediction and old_gt
                        row[f'baseline_{baseline_type}'] = baseline_info['prediction']
                        # Only add old_gt once since all baselines should have the same original ground truth
                        if not old_gt_added:
                            row['old_gt'] = baseline_info['old_gt']
                            old_gt_added = True
                    else:
                        # Old format (just prediction)
                        row[f'baseline_{baseline_type}'] = baseline_info
                if len(baseline_data) > 0:
                    print(f"    Added {len(baseline_data)} baseline predictions for {sample_id} in {category}")
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data found!")
        return df
    
    # Filter out unanswerable questions and tools
    print(f"\nFiltering out unanswerable questions and tools...")
    original_count = len(df)
    df = df[~df['fold_name'].isin(['unanswerable', 'tools'])]
    filtered_count = len(df)
    print(f"Removed {original_count - filtered_count} unanswerable and tools samples ({original_count} -> {filtered_count})")
    
    # Fill missing human annotator columns with "NONE"
    human_annotator_cols = [col for col in df.columns if col.startswith('human_annotator_')]
    for col in human_annotator_cols:
        df[col] = df[col].fillna('NONE')
    
    print(f"Constructed dataframe with {len(df)} samples across {len(categories)} categories")
    print(f"Models in dataframe: {[col for col in df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'trace', 'majority_vote', 'majority_proportion'] and not col.startswith('human_annotator_')]}")
    print(f"Human annotators in dataframe: {human_annotator_cols}")
    print()
    
    return df

def calculate_inter_rater_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate inter-rater reliability metrics from the predictions dataframe."""
    print("Calculating inter-rater reliability metrics...")
    
    # Get model columns (exclude fold_name, sample_id, actual_sample_id, trace, majority_vote, majority_proportion, and human_annotator columns)
    model_cols = [col for col in df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'trace', 'majority_vote', 'majority_proportion'] and not col.startswith('human_annotator_')]
    
    if len(model_cols) < 2:
        print("Need at least 2 models for inter-rater reliability calculation")
        return pd.DataFrame()
    
    results = []
    
    # Calculate kappa for each category
    for category in df['fold_name'].unique():
        category_df = df[df['fold_name'] == category]
        print(f"Processing category: {category} ({len(category_df)} samples)")
        
        # Calculate kappa between all pairs of models
        for i, model1 in enumerate(model_cols):
            for j, model2 in enumerate(model_cols[i+1:], i+1):
                # Get predictions for this pair
                pred1 = category_df[model1].dropna()
                pred2 = category_df[model2].dropna()
                
                # Find common samples
                common_samples = list(set(pred1.index) & set(pred2.index))
                if len(common_samples) == 0:
                    print(f"    {model1} vs {model2}: No common samples")
                    continue
                
                # Align predictions
                aligned_pred1 = pred1.loc[common_samples]
                aligned_pred2 = pred2.loc[common_samples]
                
                # Calculate kappa
                try:
                    kappa = cohen_kappa_score(aligned_pred1, aligned_pred2)
                    results.append({
                        'category': category,
                        'model1': model1,
                        'model2': model2,
                        'kappa': kappa,
                        'n_samples': len(common_samples),
                        'n_total_samples': len(category_df)
                    })
                    print(f"    {model1} vs {model2}: κ = {kappa:.3f} (n={len(common_samples)})")
                except Exception as e:
                    print(f"    {model1} vs {model2}: Error calculating kappa: {e}")
                    results.append({
                        'category': category,
                        'model1': model1,
                        'model2': model2,
                        'kappa': np.nan,
                        'n_samples': len(common_samples),
                        'n_total_samples': len(category_df)
                    })
    
    return pd.DataFrame(results)

def print_summary_statistics(kappa_df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("="*80)
    print("COHEN'S KAPPA INTER-RATER RELIABILITY SUMMARY")
    print("="*80)
    
    if kappa_df.empty:
        print("No kappa scores calculated.")
        return
    
    # Overall statistics
    valid_kappas = kappa_df['kappa'].dropna()
    if len(valid_kappas) > 0:
        print(f"Overall Statistics:")
        print(f"  Number of model pairs: {len(valid_kappas)}")
        print(f"  Mean κ: {valid_kappas.mean():.3f}")
        print(f"  Median κ: {valid_kappas.median():.3f}")
        print(f"  Std κ: {valid_kappas.std():.3f}")
        print(f"  Min κ: {valid_kappas.min():.3f}")
        print(f"  Max κ: {valid_kappas.max():.3f}")
        print()
    
    # Per-category statistics
    print("Per-Category Statistics:")
    category_stats = kappa_df.groupby('category').agg({
        'kappa': ['count', 'mean', 'std', 'min', 'max'],
        'n_samples': ['mean', 'min', 'max']
    }).round(3)
    print(category_stats)
    print()
    
    # Per-model-pair statistics
    print("Per-Model-Pair Statistics:")
    pair_stats = kappa_df.groupby(['model1', 'model2']).agg({
        'kappa': ['count', 'mean', 'std', 'min', 'max'],
        'n_samples': ['mean', 'min', 'max']
    }).round(3)
    print(pair_stats)
    print()
    
    # Sample count analysis
    print("Sample Count Analysis:")
    sample_counts = kappa_df.groupby('category')['n_total_samples'].first().sort_values(ascending=False)
    print(sample_counts)
    print()
    
    # Interpretation guide
    print("Cohen's Kappa Interpretation:")
    print("  κ < 0.20: Poor agreement")
    print("  0.20 ≤ κ < 0.40: Fair agreement")
    print("  0.40 ≤ κ < 0.60: Moderate agreement")
    print("  0.60 ≤ κ < 0.80: Good agreement")
    print("  κ ≥ 0.80: Excellent agreement")
    print()

def analyze_model_category_performance(predictions_df, model_cols, summary_file):
    """
    Analyze which models perform best in specific categories and their contribution to majority vote.
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE BY CATEGORY ANALYSIS")
    print("="*80)
    
    with open(summary_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL PERFORMANCE BY CATEGORY ANALYSIS\n")
        f.write("="*80 + "\n\n")
    
    category_analysis = {}
    
    for category in sorted(predictions_df['fold_name'].unique()):
        cat_df = predictions_df[predictions_df['fold_name'] == category]
        if 'majority_vote' not in cat_df.columns:
            continue
            
        print(f"\nCategory: {category}")
        print("-" * 50)
        
        # Calculate accuracy for each model in this category
        model_accuracies = {}
        model_kappas = {}
        
        for model in model_cols:
            if model in cat_df.columns:
                # Calculate accuracy
                model_pred = cat_df[model].dropna()
                gt_aligned = cat_df.loc[model_pred.index, 'majority_vote'].dropna()
                common_idx = model_pred.index.intersection(gt_aligned.index)
                
                if len(common_idx) > 0:
                    model_pred_aligned = model_pred.loc[common_idx]
                    gt_aligned_common = gt_aligned.loc[common_idx]
                    
                    accuracy = (model_pred_aligned == gt_aligned_common).mean()
                    model_accuracies[model] = accuracy
                    
                    # Calculate Cohen's kappa
                    if len(model_pred_aligned) > 1 and len(set(model_pred_aligned)) > 1 and len(set(gt_aligned_common)) > 1:
                        kappa = cohen_kappa_score(gt_aligned_common, model_pred_aligned)
                    else:
                        kappa = 0.0
                    model_kappas[model] = kappa
                else:
                    model_accuracies[model] = 0.0
                    model_kappas[model] = 0.0
        
        # Sort models by accuracy
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        print("Model Performance (by accuracy):")
        for i, (model, accuracy) in enumerate(sorted_models):
            kappa = model_kappas[model]
            print(f"  {i+1}. {model}: {accuracy:.3f} accuracy, κ={kappa:.3f}")
        
        # Analyze majority vote composition
        if 'majority_vote' in cat_df.columns:
            majority_vote = cat_df['majority_vote'].dropna()
            majority_breakdown = {}
            
            for model in model_cols:
                if model in cat_df.columns:
                    # Count how many times this model's prediction matches the majority vote
                    model_pred = cat_df[model].dropna()
                    common_idx = majority_vote.index.intersection(model_pred.index)
                    
                    if len(common_idx) > 0:
                        model_pred_aligned = model_pred.loc[common_idx]
                        majority_aligned = majority_vote.loc[common_idx]
                        
                        matches = (model_pred_aligned == majority_aligned).sum()
                        total = len(common_idx)
                        percentage = (matches / total * 100) if total > 0 else 0
                        majority_breakdown[model] = {'matches': matches, 'total': total, 'percentage': percentage}
            
            # Sort by percentage contribution to majority vote
            majority_contributors = sorted(majority_breakdown.items(), key=lambda x: x[1]['percentage'], reverse=True)
            
            print(f"\nMajority Vote Composition:")
            for model, stats in majority_contributors:
                print(f"  {model}: {stats['matches']}/{stats['total']} ({stats['percentage']:.1f}%)")
            
            # Identify dominant models (contribute >50% to majority vote)
            dominant_models = [model for model, stats in majority_contributors if stats['percentage'] > 50]
            if dominant_models:
                print(f"\nDominant Models (>50% majority contribution): {', '.join(dominant_models)}")
            else:
                print(f"\nNo single model dominates majority vote (all <50%)")
            
            # Check if majority vote differs from best individual model
            best_model = sorted_models[0][0] if sorted_models else None
            if best_model and 'majority_vote' in cat_df.columns:
                best_model_pred = cat_df[best_model].dropna()
                majority_vote_clean = cat_df['majority_vote'].dropna()
                common_maj_idx = best_model_pred.index.intersection(majority_vote_clean.index)
                
                if len(common_maj_idx) > 0:
                    best_model_aligned = best_model_pred.loc[common_maj_idx]
                    majority_aligned = majority_vote_clean.loc[common_maj_idx]
                    
                    agreement = (best_model_aligned == majority_aligned).mean()
                    print(f"\nMajority vote vs best model ({best_model}) agreement: {agreement:.3f}")
                    
                    if agreement < 0.8:
                        print(f"  ⚠️  Majority vote differs significantly from best individual model!")
                    else:
                        print(f"  ✅ Majority vote aligns well with best individual model")
        
        # Store analysis for summary
        category_analysis[category] = {
            'best_model': sorted_models[0][0] if sorted_models else None,
            'best_accuracy': sorted_models[0][1] if sorted_models else 0.0,
            'dominant_models': dominant_models if 'dominant_models' in locals() else [],
            'model_accuracies': model_accuracies,
            'majority_breakdown': majority_breakdown if 'majority_breakdown' in locals() else {}
        }
        
        # Write to summary file
        with open(summary_file, 'a') as f:
            f.write(f"Category: {category}\n")
            f.write("-" * 50 + "\n")
            f.write("Model Performance (by accuracy):\n")
            for i, (model, accuracy) in enumerate(sorted_models):
                kappa = model_kappas[model]
                f.write(f"  {i+1}. {model}: {accuracy:.3f} accuracy, κ={kappa:.3f}\n")
            
            if 'majority_breakdown' in locals():
                f.write(f"\nMajority Vote Composition:\n")
                for model, stats in majority_contributors:
                    f.write(f"  {model}: {stats['matches']}/{stats['total']} ({stats['percentage']:.1f}%)\n")
                
                if dominant_models:
                    f.write(f"\nDominant Models (>50% majority contribution): {', '.join(dominant_models)}\n")
                else:
                    f.write(f"\nNo single model dominates majority vote (all <50%)\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    # Overall analysis
    print(f"\n" + "="*80)
    print("OVERALL MODEL DOMINANCE ANALYSIS")
    print("="*80)
    
    # Count how many categories each model is best in
    model_wins = {}
    model_dominance = {}
    
    for category, analysis in category_analysis.items():
        best_model = analysis['best_model']
        if best_model:
            model_wins[best_model] = model_wins.get(best_model, 0) + 1
        
        for dominant_model in analysis['dominant_models']:
            model_dominance[dominant_model] = model_dominance.get(dominant_model, 0) + 1
    
    print("Categories where each model is best:")
    for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {wins} categories")
    
    print(f"\nCategories where each model dominates majority vote (>50%):")
    for model, dominance in sorted(model_dominance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {dominance} categories")
    
    # Create model agreement with majority vote table
    print(f"\n" + "="*80)
    print("MODEL AGREEMENT WITH MAJORITY VOTE BY CATEGORY")
    print("="*80)
    
    # Create a table with categories as rows and models as columns
    agreement_table = {}
    for category in sorted(predictions_df['fold_name'].unique()):
        cat_df = predictions_df[predictions_df['fold_name'] == category]
        if 'majority_vote' in cat_df.columns:
            agreement_table[category] = {}
            for model in model_cols:
                if model in cat_df.columns:
                    model_pred = cat_df[model].dropna()
                    majority_vote = cat_df['majority_vote'].dropna()
                    common_idx = model_pred.index.intersection(majority_vote.index)
                    
                    if len(common_idx) > 0:
                        model_pred_aligned = model_pred.loc[common_idx]
                        majority_aligned = majority_vote.loc[common_idx]
                        agreement_rate = (model_pred_aligned == majority_aligned).mean() * 100
                        agreement_table[category][model] = agreement_rate
                    else:
                        agreement_table[category][model] = 0.0
                else:
                    agreement_table[category][model] = 0.0
    
    # Print the table
    print(f"\n{'Category':<20} ", end="")
    for model in model_cols:
        print(f"{model:<15} ", end="")
    print()
    print("-" * (20 + 15 * len(model_cols)))
    
    for category, model_agreements in agreement_table.items():
        print(f"{category:<20} ", end="")
        for model in model_cols:
            agreement = model_agreements.get(model, 0.0)
            print(f"{agreement:<15.1f} ", end="")
        print()
    
    # Write overall analysis to summary
    with open(summary_file, 'a') as f:
        f.write("OVERALL MODEL DOMINANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Categories where each model is best:\n")
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model}: {wins} categories\n")
        
        f.write(f"\nCategories where each model dominates majority vote (>50%):\n")
        for model, dominance in sorted(model_dominance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model}: {dominance} categories\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL AGREEMENT WITH MAJORITY VOTE BY CATEGORY\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Category':<20} ")
        for model in model_cols:
            f.write(f"{model:<15} ")
        f.write("\n")
        f.write("-" * (20 + 15 * len(model_cols)) + "\n")
        
        for category, model_agreements in agreement_table.items():
            f.write(f"{category:<20} ")
            for model in model_cols:
                agreement = model_agreements.get(model, 0.0)
                f.write(f"{agreement:<15.1f} ")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")

def main():
    """Main function to calculate inter-rater reliability metrics."""
    parser = argparse.ArgumentParser(description="Calculate Cohen's kappa inter-rater reliability scores for GT labelling results")
    parser.add_argument("--input_dir", "-i", type=str, default="results/gt_labelling_rowan_modified",
                       help="Input directory containing GT labelling results (default: results/gt_labelling_rowan_modified)")
    parser.add_argument("--baseline_dir", "-b", type=str, default=None,
                       help="Optional baseline directory containing baseline results")
    parser.add_argument("--output_prefix", "-o", type=str, default="gt_labelling",
                       help="Prefix for output files (default: gt_labelling)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.input_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print(f"Please provide a valid input directory using --input_dir or -i")
        return
    
    # Construct predictions dataframe
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    predictions_df = construct_predictions_dataframe(results_dir, baseline_dir)
    
    if predictions_df.empty:
        print("No predictions found!")
        return
    
    # Calculate inter-rater reliability
    kappa_df = calculate_inter_rater_reliability(predictions_df)
    
    if kappa_df.empty:
        print("No kappa scores calculated!")
        return
    
    # Print summary statistics
    print_summary_statistics(kappa_df)
    
    # Save detailed results
    output_file = f"{args.output_prefix}_kappa_results_v2.csv"
    kappa_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Note: Predictions dataframe will be saved after majority vote calculation
    
    # Save summary statistics
    summary_file = f"{args.output_prefix}_kappa_summary_v2.txt"
    with open(summary_file, 'w') as f:
        f.write("COHEN'S KAPPA INTER-RATER RELIABILITY SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        valid_kappas = kappa_df['kappa'].dropna()
        if len(valid_kappas) > 0:
            f.write(f"Overall Statistics:\n")
            f.write(f"  Number of model pairs: {len(valid_kappas)}\n")
            f.write(f"  Mean κ: {valid_kappas.mean():.3f}\n")
            f.write(f"  Median κ: {valid_kappas.median():.3f}\n")
            f.write(f"  Std κ: {valid_kappas.std():.3f}\n")
            f.write(f"  Min κ: {valid_kappas.min():.3f}\n")
            f.write(f"  Max κ: {valid_kappas.max():.3f}\n\n")
        
        f.write("Per-Category Statistics:\n")
        category_stats = kappa_df.groupby('category').agg({
            'kappa': ['count', 'mean', 'std', 'min', 'max'],
            'n_samples': ['mean', 'min', 'max']
        }).round(3)
        f.write(category_stats.to_string())
        f.write("\n\nPer-Model-Pair Statistics:\n")
        pair_stats = kappa_df.groupby(['model1', 'model2']).agg({
            'kappa': ['count', 'mean', 'std', 'min', 'max'],
            'n_samples': ['mean', 'min', 'max']
        }).round(3)
        f.write(pair_stats.to_string())
        f.write("\n\nCohen's Kappa Interpretation:\n")
        f.write("  κ < 0.20: Poor agreement\n")
        f.write("  0.20 ≤ κ < 0.40: Fair agreement\n")
        f.write("  0.40 ≤ κ < 0.60: Moderate agreement\n")
        f.write("  0.60 ≤ κ < 0.80: Good agreement\n")
        f.write("  κ ≥ 0.80: Excellent agreement\n")
    
    print(f"Summary statistics saved to: {summary_file}")
    
    # Save simple format: model1 model2 category cohen_kappa
    simple_file = f"{args.output_prefix}_kappa_simple.txt"
    with open(simple_file, 'w') as f:
        f.write("model1 model2 category cohen_kappa\n")
        # Sort by kappa value (highest to lowest), handling NaN values
        sorted_kappa_df = kappa_df.sort_values('kappa', ascending=False, na_position='last')
        for _, row in sorted_kappa_df.iterrows():
            kappa_val = f"{row['kappa']:.3f}" if not pd.isna(row['kappa']) else "N/A"
            f.write(f"{row['model1']} {row['model2']} {row['category']} {kappa_val}\n")
    
    print(f"Simple format (sorted by kappa) saved to: {simple_file}")
    
    # Additional analysis: All models wrong and majority vote vs ground truth
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Get judge model columns only (gpt-oss-120b, gpt-4o, claude-sonnet-4, gpt-5, claude-opus-4)
    judge_models = ['gpt-oss-120b', 'gpt-4o', 'claude-sonnet-4', 'gpt-5', 'claude-opus-4']
    judge_model_cols = [col for col in judge_models if col in predictions_df.columns]
    
    # Get all model columns for other analysis (exclude fold_name, sample_id, actual_sample_id, trace, on_policy_evidence, majority_vote, majority_proportion, and human_annotator columns)
    model_cols = [col for col in predictions_df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'trace', 'majority_vote', 'majority_proportion'] and not col.startswith('human_annotator_')]
    
    if len(judge_model_cols) > 0:
        # Calculate majority vote for each sample using only judge models
        predictions_df['majority_vote'] = predictions_df[judge_model_cols].mode(axis=1)[0]
        
        # Calculate proportion for majority vote
        def calculate_majority_proportion(row):
            model_predictions = row[judge_model_cols].values
            majority_vote = row['majority_vote']
            if pd.isna(majority_vote):
                return 0.0
            # Count how many judge models voted for the majority choice
            majority_count = (model_predictions == majority_vote).sum()
            return majority_count / len(judge_model_cols)
        
        predictions_df['majority_proportion'] = predictions_df.apply(calculate_majority_proportion, axis=1)
        
        # Save predictions dataframe with majority vote columns
        predictions_file = f"{args.output_prefix}_predictions_dataframe.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions dataframe with majority vote saved to: {predictions_file}")
        
        print("Additional analysis completed - majority vote calculated using judge models only")
        
        # Upload individual samples to S3
        upload_individual_samples_to_s3(predictions_df)

def upload_individual_samples_to_s3(predictions_df):
    """Upload individual samples from predictions dataframe to S3 using batch uploads."""
    
    # Define the judge models to upload
    judge_models = ['gpt-oss-120b', 'gpt-4o', 'claude-sonnet-4', 'gpt-5', 'claude-opus-4']
    available_models = [model for model in judge_models if model in predictions_df.columns]
    
    total_uploads = len(predictions_df) * len(available_models)
    print(f"\n🔄 Preparing {total_uploads} individual samples for batch upload to S3 ({len(predictions_df)} samples × {len(available_models)} models)...")
    
    try:
        s3_client = S3SampleClient(
            bucket="dipika-lie-detection-data",
            prefix="post-processed-data/"
        )
        
        # Prepare all samples for batch upload
        batch_samples = []
        
        for _, row in predictions_df.iterrows():
            for original_model in available_models:
                try:
                    # Create sample data for this specific model
                    sample_data = {
                        "fold_name": row["fold_name"],
                        "sample_id": row["sample_id"],
                        "actual_sample_id": row["actual_sample_id"],
                        "task_name": row["task_name"],
                        "model_prediction": row[original_model],
                        "all_model_predictions": {
                            model_name: row[model_name] for model_name in available_models
                        },
                        "majority_vote": row["majority_vote"],
                        "majority_proportion": row["majority_proportion"],
                        "model_name": original_model,
                        "upload_timestamp": datetime.now().isoformat()
                    }
                    
                    # Add provider prefix to model name
                    model_with_provider = original_model
                    if 'claude' in original_model:
                        model_with_provider = 'anthropic/' + original_model
                    elif 'gpt' in original_model:
                        model_with_provider = 'openai/' + original_model
                    elif 'gemma' in original_model:
                        model_with_provider = 'google/' + original_model
                    
                    # Add to batch: (model, task, sample_id, content)
                    batch_samples.append((
                        model_with_provider,
                        row["task_name"],
                        str(row["actual_sample_id"]),
                        sample_data
                    ))
                        
                except Exception as e:
                    print(f"❌ Error preparing sample {row.get('actual_sample_id', 'unknown')} for model {original_model}: {e}")
        
        print(f"📦 Prepared {len(batch_samples)} samples for batch upload")
        
        # Perform batch upload with optimal batch size for S3
        successful_uploads, failed_uploads = s3_client.put_samples_batch(
            samples=batch_samples,
            batch_size=25  # AWS S3 recommended concurrent request limit
        )
        
        print(f"📊 S3 Batch Upload Summary: {successful_uploads} successful, {failed_uploads} failed")
        print(f"📁 Files uploaded to: s3://dipika-lie-detection-data/post-processed-data/{'{provider}'}/{'{model}'}/{'{task_name}'}/{'{sample_id}'}.json")
        print(f"🎯 Models uploaded: {', '.join(available_models)}")
        
        return successful_uploads, failed_uploads
        
    except Exception as e:
        print(f"❌ Error setting up S3 client: {e}")
        return 0, total_uploads

def main2():
    """Main function to calculate inter-rater reliability metrics."""
    parser = argparse.ArgumentParser(description="Calculate Cohen's kappa inter-rater reliability scores for GT labelling results")
    parser.add_argument("--input_dir", "-i", type=str, default="results/gt_labelling_rowan_modified",
                       help="Input directory containing GT labelling results (default: results/gt_labelling_rowan_modified)")
    parser.add_argument("--baseline_dir", "-b", type=str, default=None,
                       help="Optional baseline directory containing baseline results")
    parser.add_argument("--output_prefix", "-o", type=str, default="gt_labelling",
                       help="Prefix for output files (default: gt_labelling)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.input_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print(f"Please provide a valid input directory using --input_dir or -i")
        return
    
    # Construct predictions dataframe
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    predictions_df = construct_predictions_dataframe(results_dir, baseline_dir)
    
    if predictions_df.empty:
        print("No predictions found!")
        return
    
    # Calculate inter-rater reliability
    kappa_df = calculate_inter_rater_reliability(predictions_df)
    
    if kappa_df.empty:
        print("No kappa scores calculated!")
        return
    
    # Print summary statistics
    print_summary_statistics(kappa_df)
    
    # Save detailed results
    output_file = f"{args.output_prefix}_kappa_results_v2.csv"
    kappa_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    # Note: Predictions dataframe will be saved after majority vote calculation
    
    # Save summary statistics
    summary_file = f"{args.output_prefix}_kappa_summary_v2.txt"
    with open(summary_file, 'w') as f:
        f.write("COHEN'S KAPPA INTER-RATER RELIABILITY SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        valid_kappas = kappa_df['kappa'].dropna()
        if len(valid_kappas) > 0:
            f.write(f"Overall Statistics:\n")
            f.write(f"  Number of model pairs: {len(valid_kappas)}\n")
            f.write(f"  Mean κ: {valid_kappas.mean():.3f}\n")
            f.write(f"  Median κ: {valid_kappas.median():.3f}\n")
            f.write(f"  Std κ: {valid_kappas.std():.3f}\n")
            f.write(f"  Min κ: {valid_kappas.min():.3f}\n")
            f.write(f"  Max κ: {valid_kappas.max():.3f}\n\n")
        
        f.write("Per-Category Statistics:\n")
        category_stats = kappa_df.groupby('category').agg({
            'kappa': ['count', 'mean', 'std', 'min', 'max'],
            'n_samples': ['mean', 'min', 'max']
        }).round(3)
        f.write(category_stats.to_string())
        f.write("\n\nPer-Model-Pair Statistics:\n")
        pair_stats = kappa_df.groupby(['model1', 'model2']).agg({
            'kappa': ['count', 'mean', 'std', 'min', 'max'],
            'n_samples': ['mean', 'min', 'max']
        }).round(3)
        f.write(pair_stats.to_string())
        f.write("\n\nCohen's Kappa Interpretation:\n")
        f.write("  κ < 0.20: Poor agreement\n")
        f.write("  0.20 ≤ κ < 0.40: Fair agreement\n")
        f.write("  0.40 ≤ κ < 0.60: Moderate agreement\n")
        f.write("  0.60 ≤ κ < 0.80: Good agreement\n")
        f.write("  κ ≥ 0.80: Excellent agreement\n")
    
    print(f"Summary statistics saved to: {summary_file}")
    
    # Save simple format: model1 model2 category cohen_kappa
    simple_file = f"{args.output_prefix}_kappa_simple.txt"
    with open(simple_file, 'w') as f:
        f.write("model1 model2 category cohen_kappa\n")
        # Sort by kappa value (highest to lowest), handling NaN values
        sorted_kappa_df = kappa_df.sort_values('kappa', ascending=False, na_position='last')
        for _, row in sorted_kappa_df.iterrows():
            kappa_val = f"{row['kappa']:.3f}" if not pd.isna(row['kappa']) else "N/A"
            f.write(f"{row['model1']} {row['model2']} {row['category']} {kappa_val}\n")
    
    print(f"Simple format (sorted by kappa) saved to: {simple_file}")
    
    # Additional analysis: All models wrong and majority vote vs ground truth
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Get judge model columns only (gpt-oss-120b, gpt-4o, claude-sonnet-4, gpt-5, claude-opus-4)
    judge_models = ['gpt-oss-120b', 'gpt-4o', 'claude-sonnet-4', 'gpt-5', 'claude-opus-4']
    judge_model_cols = [col for col in judge_models if col in predictions_df.columns]
    
    # Get all model columns for other analysis (exclude fold_name, sample_id, actual_sample_id, trace, on_policy_evidence, majority_vote, majority_proportion, and human_annotator columns)
    model_cols = [col for col in predictions_df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'trace', 'majority_vote', 'majority_proportion'] and not col.startswith('human_annotator_')]
    
    if len(judge_model_cols) > 0:
        # Calculate majority vote for each sample using only judge models
        predictions_df['majority_vote'] = predictions_df[judge_model_cols].mode(axis=1)[0]
        
        # Calculate proportion for majority vote
        def calculate_majority_proportion(row):
            model_predictions = row[judge_model_cols].values
            majority_vote = row['majority_vote']
            if pd.isna(majority_vote):
                return 0.0
            # Count how many judge models voted for the majority choice
            majority_count = (model_predictions == majority_vote).sum()
            return majority_count / len(judge_model_cols)
        
        predictions_df['majority_proportion'] = predictions_df.apply(calculate_majority_proportion, axis=1)
        
        # Save predictions dataframe with majority vote columns
        predictions_file = f"{args.output_prefix}_predictions_dataframe.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions dataframe with majority vote saved to: {predictions_file}")
        
        print("Additional analysis completed - majority vote calculated using judge models only")
        
        # Write detailed table to summary file
        with open(summary_file, 'a') as f:
            f.write("\n\n" + "="*80 + "\n")
            f.write("SAMPLES WHERE ALL MODELS PREDICTED WRONG BY CATEGORY\n")
            f.write("="*80 + "\n\n")
            
            for category in sorted(predictions_df['fold_name'].unique()):
                cat_df = predictions_df[predictions_df['fold_name'] == category]
                cat_all_wrong_mask = (cat_df[model_cols] != cat_df['majority_vote'].values[:, None]).all(axis=1)
                cat_all_wrong_samples = cat_df[cat_all_wrong_mask]
                
                f.write(f"Category: {category}\n")
                f.write(f"Total samples: {len(cat_df)}\n")
                f.write(f"All models wrong: {len(cat_all_wrong_samples)} ({len(cat_all_wrong_samples)/len(cat_df)*100:.1f}%)\n")
                
                if len(cat_all_wrong_samples) > 0:
                    f.write("\nSample details (all models wrong):\n")
                    f.write("Sample ID | Ground Truth | Model Predictions | Majority Vote | Majority Prop\n")
                    f.write("-" * 80 + "\n")
                    
                    for idx, row in cat_all_wrong_samples.iterrows():
                        sample_id = row.get('sample_id', f'sample_{idx}')
                        gt = row['majority_vote']
                        model_preds = [str(row[col]) for col in model_cols]
                        majority_vote = row.get('majority_vote', 'N/A')
                        majority_prop = row.get('majority_proportion', 0.0)
                        
                        f.write(f"{sample_id} | {gt} | {', '.join(model_preds)} | {majority_vote} | {majority_prop:.2f}\n")
                else:
                    f.write("No samples where all models predicted wrong.\n")
                
                f.write("\n" + "-"*80 + "\n\n")
        
        # Write majority vote vs ground truth kappa table to summary file
        with open(summary_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("MAJORITY VOTE vs GROUND TRUTH COHEN'S KAPPA BY CATEGORY\n")
            f.write("="*80 + "\n\n")
            f.write("Category | κ(maj,gt) | κ(maj,gpt-oss) | κ(maj,gpt-4o) | κ(maj,claude-sonnet) | κ(maj,gpt-5) | κ(maj,claude-opus) | Agreement Rate | All Models Wrong\n")
            f.write("-" * 120 + "\n")
        
        # Calculate Cohen's kappa between majority vote and ground truth
        try:
            majority_vote_clean = predictions_df['majority_vote'].dropna()
            ground_truth_clean = predictions_df.loc[majority_vote_clean.index, 'majority_vote'].dropna()
            
            # Align the series
            common_idx = majority_vote_clean.index.intersection(ground_truth_clean.index)
            if len(common_idx) > 0:
                majority_vote_aligned = majority_vote_clean.loc[common_idx]
                ground_truth_aligned = ground_truth_clean.loc[common_idx]
                
                majority_vs_gt_kappa = cohen_kappa_score(ground_truth_aligned, majority_vote_aligned)
                print(f"\nCohen's kappa between majority vote and ground truth: {majority_vs_gt_kappa:.3f}")
                
                # Calculate accuracy of majority vote
                majority_accuracy = (majority_vote_aligned == ground_truth_aligned).mean()
                print(f"Majority vote accuracy: {majority_accuracy:.3f}")
                
                # Per-category analysis
                print(f"\nPer-category analysis:")
                for category in sorted(predictions_df['fold_name'].unique()):
                    cat_df = predictions_df[predictions_df['fold_name'] == category]
                    if 'majority_vote' in cat_df.columns:
                        cat_majority = cat_df['majority_vote'].dropna()
                        cat_gt = cat_df.loc[cat_majority.index, 'majority_vote'].dropna()
                        
                        common_cat_idx = cat_majority.index.intersection(cat_gt.index)
                        if len(common_cat_idx) > 0:
                            cat_majority_aligned = cat_majority.loc[common_cat_idx]
                            cat_gt_aligned = cat_gt.loc[common_cat_idx]
                            
                            # Calculate majority vote vs ground truth kappa
                            cat_kappa_gt = cohen_kappa_score(cat_gt_aligned, cat_majority_aligned)
                            cat_accuracy = (cat_majority_aligned == cat_gt_aligned).mean()
                            
                            # Calculate majority vote vs each model kappa
                            kappa_values = []
                            for model in model_cols:
                                if model in cat_df.columns:
                                    cat_model = cat_df.loc[common_cat_idx, model].dropna()
                                    common_model_idx = cat_majority_aligned.index.intersection(cat_model.index)
                                    if len(common_model_idx) > 0:
                                        cat_maj_model = cat_majority_aligned.loc[common_model_idx]
                                        cat_model_aligned = cat_model.loc[common_model_idx]
                                        
                                        if len(cat_maj_model) > 1 and len(set(cat_maj_model)) > 1 and len(set(cat_model_aligned)) > 1:
                                            model_kappa = cohen_kappa_score(cat_model_aligned, cat_maj_model)
                                        else:
                                            model_kappa = 0.0
                                        kappa_values.append(f"{model_kappa:.3f}")
                                    else:
                                        kappa_values.append("N/A")
                                else:
                                    kappa_values.append("N/A")
                            
                            cat_all_wrong = ((cat_df[model_cols] != cat_df['majority_vote'].values[:, None]).all(axis=1)).sum()
                            
                            # Calculate average agreement rate for majority vote
                            # This is the percentage of samples where majority vote matches ground truth
                            cat_agreement_rate = cat_accuracy * 100
                            
                            print(f"  {category}: κ(maj,gt)={cat_kappa_gt:.3f}, accuracy={cat_accuracy:.3f}, agreement_rate={cat_agreement_rate:.1f}%, all_wrong={cat_all_wrong}")
                            
                            # Write to summary file
                            with open(summary_file, 'a') as f:
                                f.write(f"{category:<12} | {cat_kappa_gt:>8.3f} | {kappa_values[0]:>12} | {kappa_values[1]:>12} | {kappa_values[2]:>15} | {kappa_values[3]:>12} | {kappa_values[4]:>15} | {cat_agreement_rate:>12.1f}% | {cat_all_wrong:>13}\n")
            else:
                print("No common samples found for majority vote vs ground truth comparison")
        except Exception as e:
            print(f"Error calculating majority vote analysis: {e}")
    else:
        print("Cannot perform additional analysis - missing required columns")
    
    # Perform model category performance analysis
    if len(model_cols) > 0 and 'majority_vote' in predictions_df.columns:
        try:
            analyze_model_category_performance(predictions_df, model_cols, summary_file)
        except Exception as e:
            print(f"Error in model category performance analysis: {e}")
    
    # Add human annotator analysis section
    human_annotator_cols = [col for col in predictions_df.columns if col.startswith('human_annotator_')]
    
    print("\n" + "="*80)
    print("HUMAN ANNOTATOR ANALYSIS")
    print("="*80)
    
    with open(summary_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("HUMAN ANNOTATOR ANALYSIS\n")
        f.write("="*80 + "\n\n")
    
    if human_annotator_cols:
        
        for annotator_col in human_annotator_cols:
            annotator_name = annotator_col.replace('human_annotator_', '')
            print(f"\nAnalysis for Human Annotator: {annotator_name}")
            print("-" * 50)
            
            with open(summary_file, 'a') as f:
                f.write(f"Analysis for Human Annotator: {annotator_name}\n")
                f.write("-" * 50 + "\n\n")
                
                # Majority vote vs Human annotator Cohen's kappa by category
                f.write("MAJORITY VOTE vs HUMAN ANNOTATOR COHEN'S KAPPA BY CATEGORY\n")
                f.write("-" * 60 + "\n")
                f.write("Category | κ(maj,human) | Agreement Rate | N Samples\n")
                f.write("-" * 60 + "\n")
                
                # Model accuracy vs Human annotator by category
                f.write(f"\nMODEL ACCURACY vs HUMAN ANNOTATOR BY CATEGORY\n")
                f.write("-" * 60 + "\n")
                f.write("Category | ")
                for model in model_cols:
                    f.write(f"{model:<12} | ")
                f.write("Majority Vote\n")
                f.write("-" * (60 + 15 * len(model_cols)) + "\n")
            
            print("MAJORITY VOTE vs HUMAN ANNOTATOR COHEN'S KAPPA BY CATEGORY")
            print("-" * 60)
            print("Category | κ(maj,human) | Agreement Rate | N Samples")
            print("-" * 60)
            
            print(f"\nMODEL ACCURACY vs HUMAN ANNOTATOR BY CATEGORY")
            print("-" * 60)
            print("Category | ", end="")
            for model in model_cols:
                print(f"{model:<12} | ", end="")
            print("Majority Vote")
            print("-" * (60 + 15 * len(model_cols)))
            
            # Only process categories that have human annotations
            categories_with_human_annotations = []
            for category in sorted(predictions_df['fold_name'].unique()):
                cat_df = predictions_df[predictions_df['fold_name'] == category]
                valid_human_mask = cat_df[annotator_col] != 'NONE'
                valid_cat_df = cat_df[valid_human_mask]
                if len(valid_cat_df) > 0:
                    categories_with_human_annotations.append(category)
            
            if not categories_with_human_annotations:
                print("No categories have human annotations for this annotator.")
                with open(summary_file, 'a') as f:
                    f.write("No categories have human annotations for this annotator.\n")
                continue
            
            print(f"Categories with human annotations: {len(categories_with_human_annotations)}")
            with open(summary_file, 'a') as f:
                f.write(f"Categories with human annotations: {len(categories_with_human_annotations)}\n\n")
            
            for category in categories_with_human_annotations:
                cat_df = predictions_df[predictions_df['fold_name'] == category]
                
                # Filter out samples where human annotation is 'NONE'
                valid_human_mask = cat_df[annotator_col] != 'NONE'
                valid_cat_df = cat_df[valid_human_mask]
                
                # Majority vote vs Human annotator
                if 'majority_vote' in valid_cat_df.columns:
                    majority_vote = valid_cat_df['majority_vote'].dropna()
                    human_annotations = valid_cat_df[annotator_col].dropna()
                    
                    common_idx = majority_vote.index.intersection(human_annotations.index)
                    if len(common_idx) > 0:
                        majority_aligned = majority_vote.loc[common_idx]
                        human_aligned = human_annotations.loc[common_idx]
                        
                        # Calculate Cohen's kappa
                        if len(majority_aligned) > 1 and len(set(majority_aligned)) > 1 and len(set(human_aligned)) > 1:
                            kappa_maj_human = cohen_kappa_score(human_aligned, majority_aligned)
                        else:
                            kappa_maj_human = 0.0
                        
                        # Calculate agreement rate
                        agreement_rate = (majority_aligned == human_aligned).mean() * 100
                        
                        # Store kappa info for later printing
                        kappa_info = f"{kappa_maj_human:>10.3f} | {agreement_rate:>12.1f}% | {len(common_idx):>9}"
                    else:
                        kappa_info = "No common samples"
                else:
                    kappa_info = "No majority vote data"
                
                # Model accuracy vs Human annotator
                model_accuracies = []
                
                for model in model_cols:
                    if model in valid_cat_df.columns:
                        model_pred = valid_cat_df[model].dropna()
                        human_annotations = valid_cat_df[annotator_col].dropna()
                        
                        common_idx = model_pred.index.intersection(human_annotations.index)
                        if len(common_idx) > 0:
                            model_aligned = model_pred.loc[common_idx]
                            human_aligned = human_annotations.loc[common_idx]
                            
                            accuracy = (model_aligned == human_aligned).mean() * 100
                            print(f"{accuracy:>10.1f}% | ", end="")
                            with open(summary_file, 'a') as f:
                                f.write(f"{accuracy:>10.1f}% | ")
                        else:
                            print(f"{'N/A':>10} | ", end="")
                            with open(summary_file, 'a') as f:
                                f.write(f"{'N/A':>10} | ")
                    else:
                        print(f"{'N/A':>10} | ", end="")
                        with open(summary_file, 'a') as f:
                            f.write(f"{'N/A':>10} | ")
                
                # Majority vote accuracy vs Human annotator
                if 'majority_vote' in valid_cat_df.columns:
                    majority_vote = valid_cat_df['majority_vote'].dropna()
                    human_annotations = valid_cat_df[annotator_col].dropna()
                    
                    common_idx = majority_vote.index.intersection(human_annotations.index)
                    if len(common_idx) > 0:
                        majority_aligned = majority_vote.loc[common_idx]
                        human_aligned = human_annotations.loc[common_idx]
                        
                        accuracy = (majority_aligned == human_aligned).mean() * 100
                        print(f"{accuracy:>10.1f}%")
                        with open(summary_file, 'a') as f:
                            f.write(f"{accuracy:>10.1f}%\n")
                    else:
                        print(f"{'N/A':>10}")
                        with open(summary_file, 'a') as f:
                            f.write(f"{'N/A':>10}\n")
                else:
                    print(f"{'N/A':>10}")
                    with open(summary_file, 'a') as f:
                        f.write(f"{'N/A':>10}\n")
            
            with open(summary_file, 'a') as f:
                f.write("\n" + "="*80 + "\n\n")
        
        print("\n" + "="*80)
    else:
        print("No human annotations found in the data.")
        with open(summary_file, 'a') as f:
            f.write("No human annotations found in the data.\n")

    predictions_file = f"{args.output_prefix}_predictions_dataframe.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions dataframe with majority vote saved to: {predictions_file}")

    # Upload individual samples to S3
    upload_individual_samples_to_s3(predictions_df)

def process_gt_labelling_results(results_dir: str) -> Dict[str, Any]:
    """
    Process GT labelling results from all models and calculate majority vote.
    
    Args:
        results_dir: Path to the results directory containing GT labelling results
        
    Returns:
        Dictionary mapping sample identifiers to majority vote results
    """
    results_path = Path(results_dir)
    gt_results_dir = results_path / "gt_labelling_rowan_modified"
    
    if not gt_results_dir.exists():
        print(f"❌ GT labelling results directory not found: {gt_results_dir}")
        return {}
    
    # Dictionary to store all predictions for each sample
    sample_predictions = defaultdict(list)
    
    # Process each model directory
    for model_dir in gt_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"📊 Processing model: {model_name}")
        
        # Process each category directory
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"  📁 Processing category: {category}")
            
            # Find the latest result file for this model-category combination
            result_files = list(category_dir.glob(f"{category}_*_results_*.json"))
            if not result_files:
                print(f"    ⚠️ No result files found for {model_name}/{category}")
                continue
                
            # Get the latest file
            latest_file = max(result_files, key=lambda x: extract_datetime_from_filename(x.name))
            print(f"    📄 Using file: {latest_file.name}")
            
            # Load the results
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'samples' not in data:
                    print(f"    ⚠️ No samples found in {latest_file}")
                    continue
                
                # Process each sample
                for sample in data['samples']:
                    # Create a unique identifier for this sample
                    # Try different identifier fields based on the format
                    sample_id = sample.get('sample_id', '')
                    original_dataset_sample_id = sample.get('original_dataset_sample_id', '')
                    task_name = sample.get('task_name', '')
                    
                    # If using old format, extract from metadata
                    if not sample_id and 'metadata' in sample:
                        meta = sample['metadata']
                        if 'meta' in meta and 'sample_id' in meta['meta']:
                            sample_id = str(meta['meta']['sample_id'])
                        if 'meta' in meta and 'task' in meta['meta']:
                            task_name = meta['meta']['task']
                        if 's3_metadata' in meta and 'task_id' in meta['s3_metadata']:
                            original_dataset_sample_id = meta['s3_metadata']['task_id']
                    
                    # Use a combination of identifiers for uniqueness
                    unique_id = f"{sample_id}_{original_dataset_sample_id}_{task_name}"
                    
                    # Extract the prediction (A or B) - try both old and new formats
                    old_gt = sample.get('old_gt', sample.get('target', ''))
                    baseline_prediction = sample.get('baseline_prediction', sample.get('output', ''))
                    
                    # Try to extract A/B from the model's output
                    prediction = None
                    if baseline_prediction:
                        # Look for A or B in the output text
                        # First try to find "Label: lie" or "Label: not lie"
                        if 'Label: lie' in baseline_prediction:
                            prediction = 'A'
                        elif 'Label: not lie' in baseline_prediction:
                            prediction = 'B'
                        # Fallback: look for A or B in the text
                        elif 'A' in baseline_prediction and 'B' not in baseline_prediction:
                            prediction = 'A'
                        elif 'B' in baseline_prediction and 'A' not in baseline_prediction:
                            prediction = 'B'
                        elif baseline_prediction.strip() in ['A', 'B']:
                            prediction = baseline_prediction.strip()
                    
                    if prediction:
                        sample_predictions[unique_id].append({
                            'model': model_name,
                            'prediction': prediction,
                            'old_gt': old_gt,
                            'category': category,
                            'sample_data': sample
                        })
                    else:
                        print(f"    ⚠️ Could not extract prediction from: {baseline_prediction[:100]}...")
                
                print(f"    ✅ Processed {len(data['samples'])} samples")
                
            except Exception as e:
                print(f"    ❌ Error processing {latest_file}: {e}")
                continue
    
    # Calculate majority vote for each sample
    majority_vote_results = {}
    
    print(f"\n📊 Calculating majority vote for {len(sample_predictions)} unique samples...")
    
    for unique_id, predictions in sample_predictions.items():
        if len(predictions) < 2:
            print(f"⚠️ Sample {unique_id} has only {len(predictions)} predictions, skipping")
            continue
        
        # Count predictions
        prediction_counts = defaultdict(int)
        for pred in predictions:
            prediction_counts[pred['prediction']] += 1
        
        # Find majority vote
        majority_prediction = max(prediction_counts.items(), key=lambda x: x[1])
        majority_label = majority_prediction[0]
        majority_count = majority_prediction[1]
        total_predictions = len(predictions)
        
        # Check for ties
        tied_predictions = [k for k, v in prediction_counts.items() if v == majority_count]
        
        if len(tied_predictions) > 1:
            print(f"⚠️ Sample {unique_id} has a tie: {prediction_counts}")
            # For ties, use the first prediction alphabetically
            majority_label = sorted(tied_predictions)[0]
        
        # Store the result
        majority_vote_results[unique_id] = {
            'new_gt': majority_label,
            'prediction_counts': dict(prediction_counts),
            'total_predictions': total_predictions,
            'majority_count': majority_count,
            'is_tie': len(tied_predictions) > 1,
            'predictions': predictions
        }
    
    print(f"✅ Calculated majority vote for {len(majority_vote_results)} samples")
    
    # Print summary statistics
    tie_count = sum(1 for result in majority_vote_results.values() if result['is_tie'])
    print(f"📈 Summary:")
    print(f"  - Total samples: {len(majority_vote_results)}")
    print(f"  - Ties: {tie_count}")
    print(f"  - Clear majority: {len(majority_vote_results) - tie_count}")
    
    return majority_vote_results

def update_samples_with_new_gt(results_dir: str, majority_vote_results: Dict[str, Any]) -> None:
    """
    Update all result files with the new_gt field based on majority vote.
    
    Args:
        results_dir: Path to the results directory
        majority_vote_results: Dictionary containing majority vote results
    """
    results_path = Path(results_dir)
    gt_results_dir = results_path / "gt_labelling_rowan_modified"
    
    print(f"\n🔄 Updating result files with new_gt field...")
    
    # Process each model directory
    for model_dir in gt_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"📊 Updating model: {model_name}")
        
        # Process each category directory
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"  📁 Updating category: {category}")
            
            # Find the latest result file for this model-category combination
            result_files = list(category_dir.glob(f"{category}_*_results_*.json"))
            if not result_files:
                continue
                
            latest_file = max(result_files, key=lambda x: extract_datetime_from_filename(x.name))
            
            # Load the results
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'samples' not in data:
                    continue
                
                # Update each sample with new_gt
                updated_samples = []
                for sample in data['samples']:
                    # Try different identifier fields based on the format
                    sample_id = sample.get('sample_id', '')
                    original_dataset_sample_id = sample.get('original_dataset_sample_id', '')
                    task_name = sample.get('task_name', '')
                    
                    # If using old format, extract from metadata
                    if not sample_id and 'metadata' in sample:
                        meta = sample['metadata']
                        if 'meta' in meta and 'sample_id' in meta['meta']:
                            sample_id = str(meta['meta']['sample_id'])
                        if 'meta' in meta and 'task' in meta['meta']:
                            task_name = meta['meta']['task']
                        if 's3_metadata' in meta and 'task_id' in meta['s3_metadata']:
                            original_dataset_sample_id = meta['s3_metadata']['task_id']
                    
                    unique_id = f"{sample_id}_{original_dataset_sample_id}_{task_name}"
                    
                    # Add new_gt field
                    if unique_id in majority_vote_results:
                        sample['new_gt'] = majority_vote_results[unique_id]['new_gt']
                    else:
                        sample['new_gt'] = sample.get('old_gt', sample.get('target', ''))  # Fallback to old_gt or target
                    
                    updated_samples.append(sample)
                
                # Save the updated results
                data['samples'] = updated_samples
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"    ✅ Updated {len(updated_samples)} samples in {latest_file.name}")
                
            except Exception as e:
                print(f"    ❌ Error updating {latest_file}: {e}")
                continue

def main_with_gt_processing():
    """Main function that includes GT labelling processing."""
    parser = argparse.ArgumentParser(description="Calculate GT metrics with majority vote processing")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory path")
    parser.add_argument("--process_gt", action="store_true", help="Process GT labelling results and calculate majority vote")
    
    args = parser.parse_args()
    
    if args.process_gt:
        print("🚀 Processing GT labelling results and calculating majority vote...")
        
        # Process GT labelling results
        majority_vote_results = process_gt_labelling_results(args.results_dir)
        
        if majority_vote_results:
            # Update all result files with new_gt
            update_samples_with_new_gt(args.results_dir, majority_vote_results)
            
            print("\n✅ GT labelling processing completed!")
            print(f"📊 Processed {len(majority_vote_results)} samples with majority vote")
        else:
            print("❌ No GT labelling results found to process")
    
    # Run the original main function
    main()

if __name__ == "__main__":
    main_with_gt_processing()
