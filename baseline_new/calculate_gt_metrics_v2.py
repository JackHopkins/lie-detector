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
from sklearn.metrics import cohen_kappa_score
import numpy as np

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
        
        # Load actual sample IDs from the original dataset
        actual_sample_ids = load_actual_sample_ids_from_dataset(category)
        
        predictions = {}
        for i, sample in enumerate(samples):
            # Use index-based ID for consistency
            sample_id = f"sample_{i}"
            
            # Get actual sample ID from dataset
            actual_sample_id = actual_sample_ids.get(i, sample_id)
            
            # Extract prediction and target
            pred = None
            target = None
            
            if 'scores' in sample and 'escaped_transcript_scorer' in sample['scores']:
                scorer_data = sample['scores']['escaped_transcript_scorer']
                if 'metadata' in scorer_data:
                    pred = scorer_data['metadata'].get('predicted_binary')
                    target = scorer_data['metadata'].get('target_binary')
                else:
                    print(f"    Sample {sample_id}: No metadata in scorer")
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
                predictions[sample_id] = {
                    'prediction': pred,
                    'target': target,
                    'actual_sample_id': actual_sample_id,
                    **human_annotations  # Add human annotation columns
                }
        
        print(f"    Loaded {len(predictions)} predictions from {file_path.name}")
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def construct_predictions_dataframe(results_dir: Path) -> pd.DataFrame:
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
    
    print(f"Found models: {models}")
    print(f"Found categories: {categories}")
    print()
    
    # Collect all predictions and targets
    all_predictions = {}  # {category: {sample_id: {model: prediction}}}
    all_targets = {}  # {category: {sample_id: target}}
    all_actual_sample_ids = {}  # {category: {sample_id: actual_sample_id}}
    all_human_annotations = {}  # {category: {sample_id: {annotator: annotation}}}
    
    for category in categories:
        print(f"Processing category: {category}")
        all_predictions[category] = {}
        all_targets[category] = {}
        all_actual_sample_ids[category] = {}
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
    
    # Convert to dataframe
    rows = []
    for category, samples in all_predictions.items():
        for sample_id, model_predictions in samples.items():
            row = {
                'fold_name': category,
                'sample_id': sample_id,
                'actual_sample_id': all_actual_sample_ids[category].get(sample_id, sample_id),  # Add actual sample ID from dataset
                'on_policy_evidence': all_targets[category].get(sample_id, None),  # Add target as on_policy_evidence
                **model_predictions  # Add each model's prediction as a column
            }
            
            # Add human annotation columns
            human_annotations = all_human_annotations[category].get(sample_id, {})
            for annotator_col, annotation_value in human_annotations.items():
                row[annotator_col] = annotation_value
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data found!")
        return df
    
    # Fill missing human annotator columns with "NONE"
    human_annotator_cols = [col for col in df.columns if col.startswith('human_annotator_')]
    for col in human_annotator_cols:
        df[col] = df[col].fillna('NONE')
    
    print(f"Constructed dataframe with {len(df)} samples across {len(categories)} categories")
    print(f"Models in dataframe: {[col for col in df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'on_policy_evidence'] and not col.startswith('human_annotator_')]}")
    print(f"Human annotators in dataframe: {human_annotator_cols}")
    print()
    
    return df

def calculate_inter_rater_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate inter-rater reliability metrics from the predictions dataframe."""
    print("Calculating inter-rater reliability metrics...")
    
    # Get model columns (exclude fold_name, sample_id, actual_sample_id, on_policy_evidence, and human_annotator columns)
    model_cols = [col for col in df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'on_policy_evidence'] and not col.startswith('human_annotator_')]
    
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
        if 'on_policy_evidence' not in cat_df.columns:
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
                gt_aligned = cat_df.loc[model_pred.index, 'on_policy_evidence'].dropna()
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
    parser.add_argument("--output_prefix", "-o", type=str, default="gt_labelling",
                       help="Prefix for output files (default: gt_labelling)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.input_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print(f"Please provide a valid input directory using --input_dir or -i")
        return
    
    # Construct predictions dataframe
    predictions_df = construct_predictions_dataframe(results_dir)
    
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
    
    # Get model columns (exclude fold_name, sample_id, actual_sample_id, on_policy_evidence, and human_annotator columns)
    model_cols = [col for col in predictions_df.columns if col not in ['fold_name', 'sample_id', 'actual_sample_id', 'on_policy_evidence'] and not col.startswith('human_annotator_')]
    
    if len(model_cols) > 0 and 'on_policy_evidence' in predictions_df.columns:
        # Calculate majority vote for each sample
        predictions_df['majority_vote'] = predictions_df[model_cols].mode(axis=1)[0]
        
        # Calculate proportion for majority vote
        def calculate_majority_proportion(row):
            model_predictions = row[model_cols].values
            majority_vote = row['majority_vote']
            if pd.isna(majority_vote):
                return 0.0
            # Count how many models voted for the majority choice
            majority_count = (model_predictions == majority_vote).sum()
            return majority_count / len(model_cols)
        
        predictions_df['majority_proportion'] = predictions_df.apply(calculate_majority_proportion, axis=1)
        
        # Save predictions dataframe with majority vote columns
        predictions_file = f"{args.output_prefix}_predictions_dataframe.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions dataframe with majority vote saved to: {predictions_file}")
        
        # Find samples where all models predict wrong - overall
        all_wrong_mask = (predictions_df[model_cols] != predictions_df['on_policy_evidence'].values[:, None]).all(axis=1)
        all_wrong_samples = predictions_df[all_wrong_mask]
        
        print(f"Samples where ALL models predict wrong: {len(all_wrong_samples)}")
        print(f"Total samples: {len(predictions_df)}")
        print(f"Percentage where all models wrong: {len(all_wrong_samples)/len(predictions_df)*100:.1f}%")
        
        # Breakdown by category
        print(f"\nBreakdown by category (samples where ALL models predict wrong):")
        for category in predictions_df['fold_name'].unique():
            cat_df = predictions_df[predictions_df['fold_name'] == category]
            cat_all_wrong_mask = (cat_df[model_cols] != cat_df['on_policy_evidence'].values[:, None]).all(axis=1)
            cat_all_wrong_count = cat_all_wrong_mask.sum()
            cat_total = len(cat_df)
            cat_percentage = (cat_all_wrong_count / cat_total * 100) if cat_total > 0 else 0
            print(f"  {category}: {cat_all_wrong_count}/{cat_total} ({cat_percentage:.1f}%)")
        
        # Write detailed table to summary file
        with open(summary_file, 'a') as f:
            f.write("\n\n" + "="*80 + "\n")
            f.write("SAMPLES WHERE ALL MODELS PREDICTED WRONG BY CATEGORY\n")
            f.write("="*80 + "\n\n")
            
            for category in sorted(predictions_df['fold_name'].unique()):
                cat_df = predictions_df[predictions_df['fold_name'] == category]
                cat_all_wrong_mask = (cat_df[model_cols] != cat_df['on_policy_evidence'].values[:, None]).all(axis=1)
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
                        gt = row['on_policy_evidence']
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
            ground_truth_clean = predictions_df.loc[majority_vote_clean.index, 'on_policy_evidence'].dropna()
            
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
                    if 'majority_vote' in cat_df.columns and 'on_policy_evidence' in cat_df.columns:
                        cat_majority = cat_df['majority_vote'].dropna()
                        cat_gt = cat_df.loc[cat_majority.index, 'on_policy_evidence'].dropna()
                        
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
                            
                            cat_all_wrong = ((cat_df[model_cols] != cat_df['on_policy_evidence'].values[:, None]).all(axis=1)).sum()
                            
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
    if len(model_cols) > 0 and 'on_policy_evidence' in predictions_df.columns:
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

if __name__ == "__main__":
    main()
