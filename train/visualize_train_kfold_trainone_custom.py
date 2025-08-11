#!/usr/bin/env python3
"""
Visualize k-fold training results for all models in a timestamp directory.

This script reads the training results from all model subdirectories in a timestamp
directory and creates heatmap visualizations (AUC, Accuracy, F1 scores) for each model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
import argparse
import glob
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from PIL import ImageDraw, ImageFont

# Import functions from preprocess_training_data.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the taxonomy loading functions
from preprocess_training_data import load_taxonomy_mappings, normalize_task_name

# Import complete task mapping functions
from build_complete_task_mapping import (
    build_complete_task_mapping,
    create_complete_task_name_mapping,
    find_task_category,
    process_single_jsonl_file
)

def load_training_results(results_file):
    """Load training results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def load_training_summary(input_path):
    """Load training summary to get correct training set sizes."""
    # Extract the model directory from the input path
    # input_path should be something like: /workspace/lie-detector/organized_balanced_training_cleaned_20250722_135859/openrouter_google_gemma-3-12b-it/folds_colors_chat_format
    # We need to go up one level to get to the model directory
    model_dir = Path(input_path).parent
    summary_file = model_dir / "training_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        return summary
    else:
        print(f"Warning: Training summary not found at {summary_file}")
        return None

def load_baseline_results(baseline_dir, model_name=None):
    """Load baseline results from the baseline directory."""
    baseline_dir = Path(baseline_dir) if baseline_dir else Path("/workspace/lie-detector/outputs/baseline_results_250")
    
    if not baseline_dir.exists():
        print(f"Warning: Baseline directory {baseline_dir} does not exist")
        return None
    
    baseline_results = {}
    
    # Look for results in the baseline directory
    for result_file in baseline_dir.rglob("results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract model size from filename or metadata
            model = data.get('metadata', {}).get('model', '')
            if '4b' in model.lower():
                model_size = '4b'
            elif '12b' in model.lower():
                model_size = '12b'
            elif '27b' in model.lower():
                model_size = '27b'
            else:
                continue
            
            # Extract fold name from path
            fold_name = None
            for part in result_file.parts:
                if part.endswith('_train'):
                    fold_name = part.replace('_train', '')
                    break
            
            if fold_name:
                if fold_name not in baseline_results:
                    baseline_results[fold_name] = {}
                
                # Extract metrics
                overall_metrics = data.get('overall_metrics', {})
                baseline_results[fold_name][model_size] = {
                    'accuracy': overall_metrics.get('accuracy', 0.0),
                    'f1': overall_metrics.get('f1_score', 0.0),
                    'precision': overall_metrics.get('precision', 0.0),
                    'recall': overall_metrics.get('recall', 0.0)
                }
                
        except Exception as e:
            print(f"Error loading baseline result {result_file}: {e}")
    
    return baseline_results

def get_baseline_score_for_category(baseline_results, category, metric, model_size):
    """Get baseline score for a specific category and metric."""
    if not baseline_results:
        return None
    
    # Map categories to baseline fold names
    category_to_fold = {
        'sandbagging': 'sandbagging',
        'off-policy_dolus_chat': 'off-policy_dolus_chat',
        'off-policy_truth_is_universal': 'off-policy_truth_is_universal',
        'ascii': 'sandbagging',  # ascii tasks are part of sandbagging
        'cot': 'off-policy_truth_is_universal',  # cot tasks are part of truth_is_universal
        'games': 'off-policy_dolus_chat',  # games tasks are part of dolus_chat
        'mask1': 'off-policy_dolus_chat',  # mask1 tasks are part of dolus_chat
        'mask2': 'off-policy_truth_is_universal',  # mask2 tasks are part of truth_is_universal
        'sycophancy': 'off-policy_truth_is_universal',  # sycophancy tasks are part of truth_is_universal
        'self_sycophancy': 'off-policy_dolus_chat'  # self_sycophancy tasks are part of dolus_chat
    }
    
    fold_name = category_to_fold.get(category)
    if not fold_name or fold_name not in baseline_results:
        return None
    
    if model_size not in baseline_results[fold_name]:
        return None
    
    return baseline_results[fold_name][model_size].get(metric, None)

def get_fold_groups_from_subdirectories(model_results_dir):
    """
    Get fold groups by looking at subdirectories within the model directory.
    
    Args:
        model_results_dir: Path to the model's results directory
        
    Returns:
        group1_folds: List of folds in group 1
        group2_folds: List of folds in group 2
        fold_mapping: Dictionary mapping "fold1", "fold2" to actual fold names
    """
    # Get all subdirectories in the model directory
    subdirs = [d for d in Path(model_results_dir).iterdir() if d.is_dir()]
    
    # Extract fold names from subdirectory names
    fold_names = []
    for subdir in subdirs:
        # Extract fold name from subdirectory name
        # Assuming subdirectory names contain fold information
        fold_name = subdir.name
        if fold_name not in fold_names:
            fold_names.append(fold_name)
    
    # Sort fold names alphabetically
    fold_names = sorted(fold_names)
    
    if len(fold_names) < 2:
        print(f"Warning: Only found {len(fold_names)} fold(s) in {model_results_dir}")
        return [], [], {}
    
    # Assign first fold to group 1, second fold to group 2
    group1_folds = [fold_names[0]]
    group2_folds = [fold_names[1]] if len(fold_names) > 1 else []
    
    # Create mapping
    fold_mapping = {
        "fold1": fold_names[0],
        "fold2": fold_names[1] if len(fold_names) > 1 else ""
    }
    
    print(f"Found folds: {fold_names}")
    print(f"Group 1: {group1_folds}")
    print(f"Group 2: {group2_folds}")
    print(f"Fold mapping: {fold_mapping}")
    
    return group1_folds, group2_folds, fold_mapping

def extract_auc_scores(all_training_results, training_summary=None, group1_folds=None, group2_folds=None):
    """
    Extract AUC scores from training results.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        training_summary: Training summary containing correct training set sizes
        group1_folds: List of folds in group 1
        group2_folds: List of folds in group 2
        
    Returns:
        auc_matrix: 2D numpy array of AUC scores
        train_labels: List of training fold names
        test_labels: List of test fold names
    """
    from sklearn.metrics import roc_auc_score
    
    # Get all fold names and sort alphabetically
    train_folds = sorted(list(all_training_results.keys()))
    test_folds = sorted(list(all_training_results[train_folds[0]].keys()))
    
    # Create AUC matrix
    auc_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract ROC data and calculate AUC
                results = all_training_results[train_fold][test_fold]
                if 'roc_data' in results and 'y_true' in results['roc_data'] and 'y_prob' in results['roc_data']:
                    y_true = results['roc_data']['y_true']
                    y_prob = results['roc_data']['y_prob']
                    try:
                        # The y_prob values are probabilities of class B (truthful)
                        # For lie detection, we want A (lie) as positive class
                        # So we need to flip both the labels and probabilities
                        y_true_flipped = [1 - label for label in y_true]  # Flip: A=1, B=0
                        y_prob_a = [1 - p for p in y_prob]  # Probability of A (lie)
                        auc_score = roc_auc_score(y_true_flipped, y_prob_a)
                        auc_matrix[i, j] = auc_score
                    except Exception as e:
                        print(f"Error calculating AUC for {train_fold} -> {test_fold}: {e}")
                        auc_matrix[i, j] = np.nan
                else:
                    auc_matrix[i, j] = np.nan
            else:
                auc_matrix[i, j] = np.nan
    
    # Create simplified labels based on the provided groups
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        if fold in group1_folds:
            train_labels.append("Group 1")
        else:
            train_labels.append("Group 2")
    
    for fold in test_folds:
        if fold in group1_folds:
            test_labels.append("Group 1")
        else:
            test_labels.append("Group 2")
    
    return auc_matrix, train_labels, test_labels

def create_auc_heatmap(auc_matrix, train_labels, test_labels, output_path, group1_folds, group2_folds, fold_mapping, title="AUC Scores: Train vs Test Sets"):
    """
    Create and save AUC heatmap.
    
    Args:
        auc_matrix: 2D numpy array of AUC scores
        train_labels: List of training fold names
        test_labels: List of test fold names
        output_path: Path to save the plot
        group1_folds: List of folds in group 1
        group2_folds: List of folds in group 2
        fold_mapping: Dictionary mapping fold names
        title: Plot title
    """
    # Create DataFrame for seaborn
    df = pd.DataFrame(auc_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Blues", 
                cbar_kws={'label': 'AUC'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # No legend for individual plots - only in the combined grid
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved AUC heatmap to: {output_path}")

def create_accuracy_heatmap(all_training_results, output_path, training_summary=None, group1_folds=None, group2_folds=None, fold_mapping=None, title="Accuracy Scores: Train vs Test Sets"):
    """
    Create and save accuracy heatmap as an alternative to AUC.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        output_path: Path to save the plot
        training_summary: Training summary containing correct training set sizes
        group1_folds: List of folds in group 1
        group2_folds: List of folds in group 2
        fold_mapping: Dictionary mapping fold names
        title: Plot title
    """
    # Get all fold names and sort alphabetically
    train_folds = sorted(list(all_training_results.keys()))
    test_folds = sorted(list(all_training_results[train_folds[0]].keys()))
    
    # Create accuracy matrix
    acc_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract accuracy score
                results = all_training_results[train_fold][test_fold]
                acc_score = results.get('accuracy', 0.0)
                acc_matrix[i, j] = acc_score
            else:
                acc_matrix[i, j] = np.nan
    
    # Create simplified labels based on the provided groups
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        if fold in group1_folds:
            train_labels.append("Group 1")
        else:
            train_labels.append("Group 2")
    
    for fold in test_folds:
        if fold in group1_folds:
            test_labels.append("Group 1")
        else:
            test_labels.append("Group 2")
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(acc_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Greens", 
                cbar_kws={'label': 'Accuracy'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy heatmap to: {output_path}")

def create_f1_heatmap(all_training_results, output_path, training_summary=None, group1_folds=None, group2_folds=None, fold_mapping=None, title="F1 Scores: Train vs Test Sets"):
    """
    Create and save F1 score heatmap.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        output_path: Path to save the plot
        training_summary: Training summary containing correct training set sizes
        group1_folds: List of folds in group 1
        group2_folds: List of folds in group 2
        fold_mapping: Dictionary mapping fold names
        title: Plot title
    """
    # Get all fold names and sort alphabetically
    train_folds = sorted(list(all_training_results.keys()))
    test_folds = sorted(list(all_training_results[train_folds[0]].keys()))
    
    # Create F1 matrix
    f1_matrix = np.zeros((len(train_folds), len(test_folds)))
    
    for i, train_fold in enumerate(train_folds):
        for j, test_fold in enumerate(test_folds):
            if test_fold in all_training_results[train_fold]:
                # Extract F1 score
                results = all_training_results[train_fold][test_fold]
                f1_score = results.get('f1_weighted', 0.0)
                f1_matrix[i, j] = f1_score
            else:
                f1_matrix[i, j] = np.nan
    
    # Create simplified labels based on the provided groups
    train_labels = []
    test_labels = []
    
    for fold in train_folds:
        if fold in group1_folds:
            train_labels.append("Group 1")
        else:
            train_labels.append("Group 2")
    
    for fold in test_folds:
        if fold in group1_folds:
            test_labels.append("Group 1")
        else:
            test_labels.append("Group 2")
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(f1_matrix, index=train_labels, columns=test_labels)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap="Reds", 
                cbar_kws={'label': 'F1 Score'},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # No legend for individual plots - only in the combined grid
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved F1 heatmap to: {output_path}")

def create_scaling_grid_plot(timestamp_dir, output_dir):
    """
    Create a grid plot combining all PNG files from different models.
    
    Args:
        timestamp_dir: Path to the timestamp directory containing model results
        output_dir: Path to save the combined grid plot
    """
    # Define the models in order of increasing size - only use available ones
    all_model_sizes = ['4b', '12b', '27b']
    metrics = ['auc', 'accuracy', 'f1']
    
    # Find all PNG files in the timestamp directory
    png_files = []
    available_model_sizes = []
    
    for model_size in all_model_sizes:
        model_pattern = f"*google_gemma-3-{model_size}*"
        model_dirs = glob.glob(str(Path(timestamp_dir) / model_pattern))
        
        for model_dir in model_dirs:
            # Find PNG files in this model directory
            model_png_files = glob.glob(str(Path(model_dir) / "*.png"))
            if model_png_files:  # Only include models that have files
                available_model_sizes.append(model_size)
                for png_file in model_png_files:
                    png_files.append((model_size, png_file))
    
    if not png_files:
        print("No PNG files found to combine")
        return
    
    # Sort available model sizes in the desired order: 4b, 12b, 27b
    desired_order = ['4b', '12b', '27b']
    available_model_sizes = [size for size in desired_order if size in available_model_sizes]
    print(f"Available model sizes: {available_model_sizes}")
    
    # Group files by metric and model size
    metric_files = {}
    for model_size, png_file in png_files:
        filename = Path(png_file).name.lower()
        for metric in metrics:
            if metric in filename:
                if metric not in metric_files:
                    metric_files[metric] = {}
                metric_files[metric][model_size] = png_file
                break
    
    # Check if we have all required files for available models
    required_files = []
    missing_files = []
    for metric in metrics:
        for model_size in available_model_sizes:
            if metric in metric_files and model_size in metric_files[metric]:
                required_files.append(metric_files[metric][model_size])
            else:
                missing_files.append(f"{metric} for {model_size}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return
    
    print(f"Found {len(required_files)} files to combine")
    
    # Load and resize images
    images = []
    for metric in metrics:
        row_images = []
        for model_size in available_model_sizes:
            img_path = metric_files[metric][model_size]
            img = Image.open(img_path)
            # Resize to a standard size (e.g., 800x600)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            row_images.append(img)
        images.append(row_images)
    
    # Calculate grid dimensions
    grid_width = len(available_model_sizes)  # Number of available models
    grid_height = len(metrics)     # 3 rows (auc, accuracy, f1)
    
    # Create the combined image
    img_width, img_height = images[0][0].size
    combined_width = grid_width * img_width
    combined_height = grid_height * img_height
    
    # Add extra space at the bottom for legend
    legend_height = 100  # pixels for legend
    combined_img = Image.new('RGB', (combined_width, combined_height + legend_height), 'white')
    
    # Paste images into the grid
    for row_idx, row_images in enumerate(images):
        for col_idx, img in enumerate(row_images):
            x = col_idx * img_width
            y = row_idx * img_height
            combined_img.paste(img, (x, y))
    
    # Add legend at the bottom
    from PIL import ImageDraw, ImageFont
    
    # Create a drawing object
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Use the known fold names
    group1_name = "train on cot ascii mask2 sycophancy off policy truth is universal"
    group2_name = "train on games mask1 off policy dolus chat sandbagging self sycophancy"
    
    legend_text = f"Group 1: {group1_name}\nGroup 2: {group2_name}"
    
    # Calculate text position (centered at bottom)
    bbox = draw.textbbox((0, 0), legend_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (combined_width - text_width) // 2
    text_y = combined_height + 30  # 30 pixels from bottom of grid
    
    # Draw the legend text
    draw.text((text_x, text_y), legend_text, fill='black', font=font)
    
    # Save the combined image
    output_path = Path(output_dir) / "scaling_grid_plot.png"
    combined_img.save(output_path, 'PNG', dpi=(300, 300))
    
    print(f"Saved scaling grid plot to: {output_path}")
    print(f"Grid layout: {grid_height} rows x {grid_width} columns")
    print(f"Rows (top to bottom): {', '.join(metrics)}")
    print(f"Columns (left to right): {', '.join(available_model_sizes)}")

def visualize_model_results(model_results_dir, output_dir, model_name, title_name):
    """
    Create visualizations for a single model.
    
    Args:
        model_results_dir: Path to the model's results directory
        output_dir: Path to save visualizations
        model_name: Name of the model for titles
    """
    # Find results file
    results_file = Path(model_results_dir) / "train_one_eval_all_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return False
    
    # Load results
    print(f"Loading results from: {results_file}")
    results = load_training_results(results_file)
    
    # Extract training results
    all_training_results = results.get('all_training_results', {})
    
    if not all_training_results:
        print("No training results found in the file")
        return False
    
    # Get input_path from results to load training summary
    input_path = results.get('input_path', str(model_results_dir))
    training_summary = load_training_summary(input_path)
    
    # Get fold groups from subdirectories
    group1_folds, group2_folds, fold_mapping = get_fold_groups_from_subdirectories(model_results_dir)
    
    if not group1_folds or not group2_folds:
        print("Could not determine fold groups from subdirectories")
        return False
    
    # Create visualizations
    print(f"Creating visualizations for {model_name}...")
    
    # Sanitize model name for filenames (replace / with _)
    model_name_safe = model_name.replace('/', '_')
    
    # 1. AUC Heatmap (if AUC scores are available)
    try:
        auc_matrix, train_labels, test_labels = extract_auc_scores(all_training_results, training_summary, group1_folds, group2_folds)
        if not np.all(np.isnan(auc_matrix)):
            auc_output = output_dir / f"auc_heatmap_{model_name_safe}.png"
            create_auc_heatmap(auc_matrix, train_labels, test_labels, auc_output, 
                             group1_folds, group2_folds, fold_mapping, f"AUC Scores: {title_name}")
        else:
            print("No AUC scores found in results")
    except Exception as e:
        print(f"Error creating AUC heatmap: {e}")
    
    # 2. Accuracy Heatmap
    try:
        acc_output = output_dir / f"accuracy_heatmap_{model_name_safe}.png"
        create_accuracy_heatmap(all_training_results, acc_output, training_summary, 
                               group1_folds, group2_folds, fold_mapping, f"Accuracy Scores: {title_name}")
    except Exception as e:
        print(f"Error creating accuracy heatmap: {e}")
    
    # 3. F1 Score Heatmap
    try:
        f1_output = output_dir / f"f1_heatmap_{model_name_safe}.png"
        create_f1_heatmap(all_training_results, f1_output, training_summary, 
                          group1_folds, group2_folds, fold_mapping, f"F1 Scores: {title_name}")
    except Exception as e:
        print(f"Error creating F1 heatmap: {e}")
    
    return True

def create_fine_grained_grid_plot(timestamp_dir, output_dir, taxonomy_path):
    """
    Create fine-grained grid plots that group test tasks by taxonomy categories.
    
    Args:
        timestamp_dir: Path to the timestamp directory containing model results
        output_dir: Path to save the combined grid plot
        taxonomy_path: Path to the taxonomy CSV file
    """
    # Load taxonomy mappings
    taxonomy_mappings = load_taxonomy_mappings(taxonomy_path)
    if not taxonomy_mappings:
        print("Warning: No taxonomy mappings loaded")
        return
    
    # Use the 'colors' mapping if available
    if 'colors' in taxonomy_mappings:
        task_to_category = {}
        for category, tasks in taxonomy_mappings['colors'].items():
            for task in tasks:
                task_to_category[task] = category
    else:
        print("No 'colors' mapping found in taxonomy")
        return
    
    print(f"Loaded taxonomy with {len(task_to_category)} task mappings")
    
    # Define the models in order of increasing size
    all_model_sizes = ['4b', '12b', '27b']
    metrics = ['auc', 'accuracy', 'f1']
    
    # Find all model directories
    available_model_sizes = []
    model_results = {}
    
    for model_size in all_model_sizes:
        model_pattern = f"*google_gemma-3-{model_size}*"
        model_dirs = glob.glob(str(Path(timestamp_dir) / model_pattern))
        
        for model_dir in model_dirs:
            results_file = Path(model_dir) / "train_one_eval_all_results.json"
            if results_file.exists():
                available_model_sizes.append(model_size)
                model_results[model_size] = model_dir
                break
    
    if not model_results:
        print("No model results found")
        return
    
    # Sort available model sizes in the desired order: 4b, 12b, 27b
    desired_order = ['4b', '12b', '27b']
    available_model_sizes = [size for size in desired_order if size in available_model_sizes]
    print(f"Available model sizes: {available_model_sizes}")
    
    # Process each model and create fine-grained plots
    for model_size in available_model_sizes:
        model_dir = model_results[model_size]
        print(f"\nProcessing fine-grained plots for {model_size} model...")
        
        # Load results for this model
        results_file = Path(model_dir) / "train_one_eval_all_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        all_training_results = results.get('all_training_results', {})
        if not all_training_results:
            print(f"No training results found for {model_size}")
            continue
        
        # Load baseline data for this model
        model_name = f"openrouter_google_gemma-3-{model_size}-it"
        baseline_data = load_baseline_results(None, model_name)
        
        # Create fine-grained plots for each metric
        for metric in metrics:
            create_fine_grained_metric_plot(
                timestamp_dir,
                output_dir, 
                model_size, 
                metric, 
                taxonomy_path, 
                baseline_data
            )

def create_fine_grained_metric_plot(timestamp_dir, output_dir, model_size, metric, taxonomy_path, baseline_data=None):
    """Create fine-grained metric plot for a specific model size and metric."""
    print(f"Creating fine-grained {metric} plot for {model_size} model...")
    
    # Load taxonomy
    taxonomy_df = pd.read_csv(taxonomy_path)
    task_to_category = dict(zip(taxonomy_df['Task'], taxonomy_df['colors']))
    
    # Find model directory
    model_pattern = f"*{model_size}*"
    model_dirs = list(Path(timestamp_dir).glob(model_pattern))
    
    if not model_dirs:
        print(f"No model directories found for pattern: {model_pattern}")
        return
    
    model_dir = model_dirs[0]
    print(f"Processing model directory: {model_dir}")
    
    # Find detailed prediction files
    detailed_pred_files = list(model_dir.glob("*detailed_predictions_with_task.json"))
    print(f"Found {len(detailed_pred_files)} detailed prediction files for {model_size}")
    
    for file_path in detailed_pred_files:
        print(f"  {file_path.name}")
    
    # Collect all predictions and group by category
    all_predictions = {}
    categories = set()
    
    for file_path in detailed_pred_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"Processing file: {file_path.name}")
            print(f"  Total predictions: {len(data.get('predictions', []))}")
            
            for prediction in data.get('predictions', []):
                task = prediction.get('task', 'unknown')
                normalized_task = normalize_task_name(task)
                category = task_to_category.get(normalized_task, 'unknown')
                categories.add(category)
                
                if category not in all_predictions:
                    all_predictions[category] = []
                
                all_predictions[category].append({
                    'true_label': prediction.get('true_label', 0),
                    'predicted_label': prediction.get('predicted_label', 0),
                    'probabilities': prediction.get('probabilities', [0, 0])
                })
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Categories for {metric}: {sorted(categories)}")
    for category in sorted(categories):
        print(f"    {category}: {len(all_predictions.get(category, []))} tasks")
    

    
    # Create matrix for heatmap
    categories_list = sorted(categories)
    matrix = np.zeros((2, len(categories_list)))  # 2 rows: Group 1, Group 2
    
    # Separate predictions by training fold
    group1_predictions = {}
    group2_predictions = {}
    
    for file_path in detailed_pred_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Determine which group this file belongs to based on filename
            filename = file_path.name
            if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
                group = "group1"
            elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
                group = "group2"
            else:
                continue
            
            # Process predictions for this group
            for prediction in data.get('predictions', []):
                task = prediction.get('task', 'unknown')
                normalized_task = normalize_task_name(task)
                category = task_to_category.get(normalized_task, 'unknown')
                
                if group == "group1":
                    if category not in group1_predictions:
                        group1_predictions[category] = []
                    group1_predictions[category].append({
                        'true_label': prediction.get('true_label', 0),
                        'predicted_label': prediction.get('predicted_label', 0),
                        'probabilities': prediction.get('probabilities', {})
                    })
                else:
                    if category not in group2_predictions:
                        group2_predictions[category] = []
                    group2_predictions[category].append({
                        'true_label': prediction.get('true_label', 0),
                        'predicted_label': prediction.get('predicted_label', 0),
                        'probabilities': prediction.get('probabilities', {})
                    })
                    
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Calculate metrics for each group and category
    for i, category in enumerate(categories_list):
        # Group 1 metrics
        group1_preds = group1_predictions.get(category, [])
        if group1_preds:
            true_labels = [p['true_label'] for p in group1_preds]
            predicted_labels = [p['predicted_label'] for p in group1_preds]
            
            if metric == 'auc':
                try:
                    probabilities = [p['probabilities'].get('B_prob', 0) for p in group1_preds]
                    score = roc_auc_score(true_labels, probabilities)
                except ValueError:
                    score = np.nan
            elif metric == 'accuracy':
                score = accuracy_score(true_labels, predicted_labels)
            elif metric == 'f1':
                numeric_true_labels = [1 if label == "B" else 0 for label in true_labels]
                numeric_predictions = [1 if pred == "B" else 0 for pred in predicted_labels]
                score = f1_score(numeric_true_labels, numeric_predictions, average='binary')
            else:
                score = np.nan
            matrix[0, i] = score
        else:
            matrix[0, i] = np.nan
        
        # Group 2 metrics
        group2_preds = group2_predictions.get(category, [])
        if group2_preds:
            true_labels = [p['true_label'] for p in group2_preds]
            predicted_labels = [p['predicted_label'] for p in group2_preds]
            
            if metric == 'auc':
                try:
                    probabilities = [p['probabilities'].get('B_prob', 0) for p in group2_preds]
                    score = roc_auc_score(true_labels, probabilities)
                except ValueError:
                    score = np.nan
            elif metric == 'accuracy':
                score = accuracy_score(true_labels, predicted_labels)
            elif metric == 'f1':
                numeric_true_labels = [1 if label == "B" else 0 for label in true_labels]
                numeric_predictions = [1 if pred == "B" else 0 for pred in predicted_labels]
                score = f1_score(numeric_true_labels, numeric_predictions, average='binary')
            else:
                score = np.nan
            matrix[1, i] = score
        else:
            matrix[1, i] = np.nan
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix data:")
    for i, row in enumerate(matrix):
        print(f"  Row {i}: {row}")
    
    # Add baseline row for accuracy plots
    if metric == 'accuracy' and baseline_data and 'llama_chat' in baseline_data:
        print("Adding baseline row...")
        # Create baseline row
        baseline_row = []
        for category in categories_list:
            # Find baseline value for this category
            category_baseline = None
            for fold_name, baseline_acc in baseline_data['llama_chat'].items():
                # More precise matching to avoid conflicts
                if fold_name == category:
                    category_baseline = baseline_acc
                    break
            
            if category_baseline is not None:
                baseline_row.append(category_baseline)
                print(f"  {category}: {category_baseline}")
            else:
                baseline_row.append(np.nan)
                print(f"  {category}: no baseline found")
        
        # Add baseline row to matrix
        baseline_matrix = np.vstack([matrix, np.array(baseline_row).reshape(1, -1)])
        train_labels_with_baseline = ['Group 1', 'Group 2', 'Baseline (Llama Chat)']
        print(f"Baseline row: {baseline_row}")
    else:
        baseline_matrix = matrix
        train_labels_with_baseline = ['Group 1', 'Group 2']
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(baseline_matrix, index=train_labels_with_baseline, columns=categories_list)
    print(f"Final DataFrame:")
    print(df)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5, 
                cbar_kws={'label': f'{metric.upper()} Score'})
    plt.title(f'Fine-grained {metric.upper()} - {model_size.upper()} Model')
    plt.xlabel('Test Categories')
    plt.ylabel('Training Groups')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'fine_grained_{metric}_{model_size}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved fine-grained {metric} plot for {model_size} to: {output_path}")

def create_fine_grained_scaling_grid_plot(timestamp_dir, output_dir, taxonomy_path):
    """
    Create a scaling grid plot combining fine-grained plots from different models.
    
    Args:
        timestamp_dir: Path to the timestamp directory containing model results
        output_dir: Path to save the combined grid plot
        taxonomy_path: Path to the taxonomy CSV file
    """
    # Load taxonomy mappings
    taxonomy_mappings = load_taxonomy_mappings(taxonomy_path)
    if not taxonomy_mappings:
        print("Warning: No taxonomy mappings loaded")
        return
    
    # Use the 'colors' mapping if available
    if 'colors' in taxonomy_mappings:
        task_to_category = {}
        for category, tasks in taxonomy_mappings['colors'].items():
            for task in tasks:
                task_to_category[task] = category
    else:
        print("No 'colors' mapping found in taxonomy")
        return
    
    # Define the models in order of increasing size
    all_model_sizes = ['4b', '12b', '27b']
    metrics = ['auc', 'accuracy', 'f1']
    
    # Find all PNG files for fine-grained plots
    png_files = []
    available_model_sizes = []
    
    for model_size in all_model_sizes:
        model_pattern = f"*google_gemma-3-{model_size}*"
        model_dirs = glob.glob(str(Path(timestamp_dir) / model_pattern))
        
        for model_dir in model_dirs:
            # Look for fine-grained PNG files in the main output directory
            fine_grained_pattern = f"fine_grained_*_{model_size}.png"
            # Check in the main output directory (not model subdirectories)
            main_png_files = glob.glob(str(Path(output_dir) / fine_grained_pattern))
            if main_png_files:  # Only include models that have files
                available_model_sizes.append(model_size)
                for png_file in main_png_files:
                    png_files.append((model_size, png_file))
    
    if not png_files:
        print("No fine-grained PNG files found to combine")
        return
    
    # Sort available model sizes in the desired order: 4b, 12b, 27b
    desired_order = ['4b', '12b', '27b']
    available_model_sizes = [size for size in desired_order if size in available_model_sizes]
    print(f"Available model sizes: {available_model_sizes}")
    
    # Group files by metric and model size
    metric_files = {}
    for model_size, png_file in png_files:
        filename = Path(png_file).name.lower()
        for metric in metrics:
            if f"fine_grained_{metric}" in filename:
                if metric not in metric_files:
                    metric_files[metric] = {}
                metric_files[metric][model_size] = png_file
                break
    
    # Check if we have all required files for available models
    required_files = []
    missing_files = []
    for metric in metrics:
        for model_size in available_model_sizes:
            if metric in metric_files and model_size in metric_files[metric]:
                required_files.append(metric_files[metric][model_size])
            else:
                missing_files.append(f"fine_grained_{metric} for {model_size}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return
    
    print(f"Found {len(required_files)} fine-grained files to combine")
    
    # Load and resize images
    images = []
    for metric in metrics:
        row_images = []
        for model_size in available_model_sizes:
            img_path = metric_files[metric][model_size]
            img = Image.open(img_path)
            # Resize to a standard size (e.g., 800x600)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            row_images.append(img)
        images.append(row_images)
    
    # Calculate grid dimensions
    grid_width = len(available_model_sizes)  # Number of available models
    grid_height = len(metrics)     # 3 rows (auc, accuracy, f1)
    
    # Create the combined image
    img_width, img_height = images[0][0].size
    combined_width = grid_width * img_width
    combined_height = grid_height * img_height
    
    # Add extra space at the bottom for legend
    legend_height = 100  # pixels for legend
    combined_img = Image.new('RGB', (combined_width, combined_height + legend_height), 'white')
    
    # Paste images into the grid
    for row_idx, row_images in enumerate(images):
        for col_idx, img in enumerate(row_images):
            x = col_idx * img_width
            y = row_idx * img_height
            combined_img.paste(img, (x, y))
    
    # Add legend at the bottom
    from PIL import ImageDraw, ImageFont
    
    # Create a drawing object
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Create legend text with taxonomy categories
    legend_text = "Test Categories by Taxonomy:\n"
    for category, tasks in taxonomy_mappings['colors'].items():
        legend_text += f"{category}: {len(tasks)} tasks\n"
    
    # Calculate text position (centered at bottom)
    bbox = draw.textbbox((0, 0), legend_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (combined_width - text_width) // 2
    text_y = combined_height + 30  # 30 pixels from bottom of grid
    
    # Draw the legend text
    draw.text((text_x, text_y), legend_text, fill='black', font=font)
    
    # Save the combined image
    output_path = Path(output_dir) / "fine_grained_scaling_grid_plot.png"
    combined_img.save(output_path, 'PNG', dpi=(300, 300))
    
    print(f"Saved fine-grained scaling grid plot to: {output_path}")
    print(f"Grid layout: {grid_height} rows x {grid_width} columns")
    print(f"Rows (top to bottom): {', '.join(metrics)}")
    print(f"Columns (left to right): {', '.join(available_model_sizes)}")

def create_fine_grained_plots_for_scaling_grid(timestamp_dir, output_dir, taxonomy_path):
    """
    Create fine-grained plots and save them in model directories for scaling grid plot to find.
    
    Args:
        timestamp_dir: Path to the timestamp directory containing model results
        output_dir: Path to save the combined grid plot
        taxonomy_path: Path to the taxonomy CSV file
    """
    # Load taxonomy mappings
    taxonomy_mappings = load_taxonomy_mappings(taxonomy_path)
    if not taxonomy_mappings:
        print("Warning: No taxonomy mappings loaded")
        return
    
    # Use the 'colors' mapping if available
    if 'colors' in taxonomy_mappings:
        task_to_category = {}
        for category, tasks in taxonomy_mappings['colors'].items():
            for task in tasks:
                task_to_category[task] = category
    else:
        print("No 'colors' mapping found in taxonomy")
        return
    
    print(f"Loaded taxonomy with {len(task_to_category)} task mappings")
    
    # Define the models in order of increasing size
    all_model_sizes = ['4b', '12b', '27b']
    metrics = ['auc', 'accuracy', 'f1']
    
    # Find all model directories
    available_model_sizes = []
    model_results = {}
    
    for model_size in all_model_sizes:
        model_pattern = f"*google_gemma-3-{model_size}*"
        model_dirs = glob.glob(str(Path(timestamp_dir) / model_pattern))
        
        for model_dir in model_dirs:
            results_file = Path(model_dir) / "train_one_eval_all_results.json"
            if results_file.exists():
                available_model_sizes.append(model_size)
                model_results[model_size] = model_dir
                break
    
    if not model_results:
        print("No model results found")
        return
    
    # Sort available model sizes in the desired order: 4b, 12b, 27b
    desired_order = ['4b', '12b', '27b']
    available_model_sizes = [size for size in desired_order if size in available_model_sizes]
    print(f"Available model sizes: {available_model_sizes}")
    
    # Process each model and create fine-grained plots
    for model_size in available_model_sizes:
        model_dir = model_results[model_size]
        print(f"\nProcessing fine-grained plots for {model_size} model...")
        
        # Load results for this model
        results_file = Path(model_dir) / "train_one_eval_all_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        all_training_results = results.get('all_training_results', {})
        if not all_training_results:
            print(f"No training results found for {model_size}")
            continue
        
        # Create fine-grained plots for each metric
        for metric in metrics:
            create_fine_grained_metric_plot_for_scaling_grid(
                all_training_results, 
                model_size, 
                metric, 
                task_to_category, 
                model_dir  # Save in model directory
            )

def create_fine_grained_metric_plot_for_scaling_grid(all_training_results, model_size, metric, task_to_category, model_dir):
    """
    Create a fine-grained plot for a specific metric, grouping test tasks by taxonomy categories.
    Uses task keys from detailed predictions instead of parsing composite fold names.
    
    Args:
        all_training_results: Dictionary containing results for each training fold
        model_size: Model size (e.g., '4b', '27b')
        metric: Metric name ('auc', 'accuracy', 'f1')
        task_to_category: Dictionary mapping task names to taxonomy categories
        model_dir: Model directory to save the plot
    """
    # Get all fold names and sort alphabetically
    train_folds = sorted(list(all_training_results.keys()))
    
    # Load detailed predictions to get actual task information
    detailed_pred_files = list(Path(model_dir).glob("*detailed_predictions_with_task.json"))
    
    if not detailed_pred_files:
        print(f"  No detailed predictions with task found for {model_size}")
        return
    
    # Group tasks by category using actual task keys from detailed predictions
    category_to_tasks = defaultdict(set)
    task_to_category_mapping = {}
    
    # Process each detailed predictions file
    for pred_file in detailed_pred_files:
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        
        # Extract fold name from filename to match with all_training_results
        filename = pred_file.name
        if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal_eval_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
            train_fold = "train_on_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal"
        elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy_eval_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
            train_fold = "train_on_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
        else:
            continue
        
        # Process each prediction to get task categories
        for prediction in pred_data['predictions']:
            task = prediction.get('task', 'unknown')
            
            # Map task to category using taxonomy
            category = 'unknown'
            if task in task_to_category:
                category = task_to_category[task]
            
            # Store the mapping
            task_to_category_mapping[task] = category
            category_to_tasks[category].add(task)
    
    # Convert sets to sorted lists
    categories = sorted(category_to_tasks.keys())
    for category in categories:
        category_to_tasks[category] = sorted(list(category_to_tasks[category]))
    
    print(f"  Categories for {metric}: {categories}")
    for category in categories:
        print(f"    {category}: {len(category_to_tasks[category])} tasks")
    
    # Create the fine-grained matrix
    # Rows: Train folds (Group 1, Group 2)
    # Columns: Test categories (tools, ascii, self_sycophancy, etc.)
    train_labels = []
    test_labels = []
    
    # Determine train fold groups (same logic as before)
    group1_folds = []
    group2_folds = []
    
    if len(train_folds) >= 2:
        group1_folds = [train_folds[0]]
        group2_folds = [train_folds[1]]
    
    # Create train labels
    for fold in train_folds:
        if fold in group1_folds:
            train_labels.append("Group 1")
        else:
            train_labels.append("Group 2")
    
    # Create test labels (category names)
    test_labels = categories
    
    # Create the matrix
    matrix = np.zeros((len(train_folds), len(categories)))
    
    # Fill the matrix with aggregated scores
    for i, train_fold in enumerate(train_folds):
        for j, category in enumerate(categories):
            category_tasks = category_to_tasks[category]
            scores = []
            
            # For each task in this category, calculate individual task scores
            for task in category_tasks:
                # Find predictions for this specific task
                task_predictions = []
                task_true_labels = []
                
                # Load detailed predictions and filter by task
                for pred_file in detailed_pred_files:
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                    
                    # Extract train fold from filename
                    filename = pred_file.name
                    if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal_eval_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
                        file_train_fold = "train_on_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal"
                    elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy_eval_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
                        file_train_fold = "train_on_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
                    else:
                        continue
                    
                    # Only process if this file matches our train fold
                    if file_train_fold == train_fold:
                        for prediction in pred_data['predictions']:
                            pred_task = prediction.get('task', 'unknown')
                            if pred_task == task:
                                task_predictions.append(prediction.get('predicted_label', 0))
                                task_true_labels.append(prediction.get('true_label', 0))
                
                # Calculate metric for this specific task
                if task_predictions and task_true_labels:
                    # Convert string labels to numeric for calculations
                    # "A" = 0 (lie), "B" = 1 (truth)
                    numeric_true_labels = [1 if label == "B" else 0 for label in task_true_labels]
                    numeric_predictions = [1 if pred == "B" else 0 for pred in task_predictions]
                    
                    if metric == 'auc':
                        try:
                            from sklearn.metrics import roc_auc_score
                            # Flip labels for AUC calculation (1 = lie, 0 = truth)
                            y_true_flipped = [1 - label for label in numeric_true_labels]
                            y_pred_flipped = [1 - p for p in numeric_predictions]
                            auc_score = roc_auc_score(y_true_flipped, y_pred_flipped)
                            scores.append(auc_score)
                        except Exception as e:
                            print(f"Error calculating AUC for {train_fold} -> {task}: {e}")
                    elif metric == 'accuracy':
                        from sklearn.metrics import accuracy_score
                        acc_score = accuracy_score(numeric_true_labels, numeric_predictions)
                        scores.append(acc_score)
                    elif metric == 'f1':
                        from sklearn.metrics import f1_score
                        f1_score_val = f1_score(numeric_true_labels, numeric_predictions, average='weighted')
                        scores.append(f1_score_val)
            
            # Calculate mean score for this category
            if scores:
                matrix[i, j] = np.mean(scores)
            else:
                matrix[i, j] = np.nan
    
    # Set up the plot
    plt.figure(figsize=(max(8, len(categories) * 0.8), 6))
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(matrix, index=train_labels, columns=categories)
    
    # Choose colormap based on metric
    if metric == 'auc':
        cmap = "Blues"
        title = f"AUC Scores: {model_size.upper()} Model"
    elif metric == 'accuracy':
        cmap = "Greens"
        title = f"Accuracy Scores: {model_size.upper()} Model"
    elif metric == 'f1':
        cmap = "Reds"
        title = f"F1 Scores: {model_size.upper()} Model"
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap=cmap, 
                cbar_kws={'label': metric.upper()},
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True,
                mask=np.isnan(df))  # Mask NaN values to show them as white
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Categories", fontsize=12)
    plt.ylabel("Train Groups", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add legend for Group 1 and Group 2
    group1_name = "train on cot ascii mask2 sycophancy off policy truth is universal"
    group2_name = "train on games mask1 off policy dolus chat sandbagging self sycophancy"
    legend_text = f"Group 1: {group1_name}\nGroup 2: {group2_name}"
    
    # Add legend at the bottom
    plt.figtext(0.5, 0.02, legend_text, ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot in the model directory
    output_path = Path(model_dir) / f"fine_grained_{metric}_{model_size}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved fine-grained {metric} plot for {model_size} to: {output_path}")

def map_predictions_to_tasks(timestamp_dir, subdir):
    """
    Map predictions to task names for each model.
    
    Args:
        timestamp_dir: Timestamp directory name
        subdir: Subdirectory name
    """
    import json
    from pathlib import Path
    
    # Define model directories
    model_dirs = [
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-4b-it_train_one_eval_all_{timestamp_dir}",
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-12b-it_train_one_eval_all_{timestamp_dir}",
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-27b-it_train_one_eval_all_{timestamp_dir}"
    ]
    
    for model_dir in model_dirs:
        if not Path(model_dir).exists():
            print(f"Model directory not found: {model_dir}")
            continue
            
        # Extract model size from directory name
        if "4b-it" in model_dir:
            model_size = "4b-it"
        elif "12b-it" in model_dir:
            model_size = "12b-it"
        elif "27b-it" in model_dir:
            model_size = "27b-it"
        else:
            print(f"Could not determine model size from: {model_dir}")
            continue
            
        print(f"\nProcessing model: {model_size}")
        print(f"Model directory: {model_dir}")
        
        # Training data paths for each fold (model-specific)
        training_data_paths = {
            "cot_ascii_mask2_sycophancy_off-policy_truth_is_universal": f"/workspace/lie-detector/organized_balanced_training_{timestamp_dir}_cleaned_50_50/openrouter_google_gemma-3-{model_size}/folds_colors_chat_format/cot_ascii_mask2_sycophancy_off-policy_truth_is_universal",
            "games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy": f"/workspace/lie-detector/organized_balanced_training_{timestamp_dir}_cleaned_50_50/openrouter_google_gemma-3-{model_size}/folds_colors_chat_format/games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
        }
        
        # Find all detailed predictions files
        detailed_pred_files = list(Path(model_dir).glob("*detailed_predictions.json"))
        
        for pred_file in detailed_pred_files:
            print(f"\nProcessing: {pred_file.name}")
            
            # Extract fold name from filename
            filename = pred_file.name
            if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal_eval_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
                fold_name = "cot_ascii_mask2_sycophancy_off-policy_truth_is_universal"
            elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy_eval_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
                fold_name = "games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
            else:
                print(f"Could not determine fold name from filename: {filename}")
                continue
            
            # Load training data to get task mappings
            training_data_path = training_data_paths[fold_name]
            train_file = Path(training_data_path) / "train.jsonl"
            test_file = Path(training_data_path) / "test.jsonl"
            
            if not train_file.exists() or not test_file.exists():
                print(f"Training data files not found for fold: {fold_name}")
                continue
            
            # Load training data and create message to task mapping
            message_to_task = {}
            
            # Load train.jsonl
            print(f"Loading train data from: {train_file}")
            with open(train_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    messages = data.get('messages', [])
                    task = data.get('task', 'unknown')
                    
                    # Create a consistent string representation of messages
                    messages_str = json.dumps(messages, sort_keys=True)
                    message_to_task[messages_str] = task
            
            # Load test.jsonl
            print(f"Loading test data from: {test_file}")
            with open(test_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    messages = data.get('messages', [])
                    task = data.get('task', 'unknown')
                    
                    # Create a consistent string representation of messages
                    messages_str = json.dumps(messages, sort_keys=True)
                    message_to_task[messages_str] = task
            
            print(f"Loaded {len(message_to_task)} message-to-task mappings for {fold_name}")
            
            # Load detailed predictions
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            # Add task key to each prediction
            matched_count = 0
            total_count = len(pred_data['predictions'])
            
            for prediction in pred_data['predictions']:
                messages = prediction['messages']
                messages_str = json.dumps(messages, sort_keys=True)
                
                if messages_str in message_to_task:
                    prediction['task'] = message_to_task[messages_str]
                    matched_count += 1
                else:
                    prediction['task'] = 'unknown'
            
            print(f"Matched {matched_count}/{total_count} predictions with tasks")
            
            # Save updated predictions
            output_file = pred_file.parent / f"{pred_file.stem}_with_task.json"
            with open(output_file, 'w') as f:
                json.dump(pred_data, f, indent=2)
            
            print(f"Saved updated predictions to: {output_file}")
            
            # Print some statistics about the tasks found
            task_counts = {}
            for prediction in pred_data['predictions']:
                task = prediction.get('task', 'unknown')
                task_counts[task] = task_counts.get(task, 0) + 1
            
            print("Task distribution:")
            for task, count in sorted(task_counts.items()):
                print(f"  {task}: {count} predictions")

def add_task_keys_to_detailed_predictions(base_output_dir, timestamp_dir, subdir):
    """
    Add task keys to detailed predictions files by matching messages with training data.
    
    Args:
        base_output_dir: Base output directory
        timestamp_dir: Timestamp directory name
        subdir: Subdirectory name
    """
    import json
    from pathlib import Path
    
    # Define model directories
    model_dirs = [
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-4b-it_train_one_eval_all_{timestamp_dir}",
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-12b-it_train_one_eval_all_{timestamp_dir}",
        f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}/openrouter_google_gemma-3-27b-it_train_one_eval_all_{timestamp_dir}"
    ]
    
    for model_dir in model_dirs:
        if not Path(model_dir).exists():
            print(f"Model directory not found: {model_dir}")
            continue
            
        # Extract model size from directory name
        if "4b-it" in model_dir:
            model_size = "4b-it"
        elif "12b-it" in model_dir:
            model_size = "12b-it"
        elif "27b-it" in model_dir:
            model_size = "27b-it"
        else:
            print(f"Could not determine model size from: {model_dir}")
            continue
            
        # Training data paths for each fold (model-specific)
        training_data_paths = {
            "cot_ascii_mask2_sycophancy_off-policy_truth_is_universal": f"/workspace/lie-detector/organized_balanced_training_{timestamp_dir}_cleaned_50_50/openrouter_google_gemma-3-{model_size}/folds_colors_chat_format/cot_ascii_mask2_sycophancy_off-policy_truth_is_universal",
            "games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy": f"/workspace/lie-detector/organized_balanced_training_{timestamp_dir}_cleaned_50_50/openrouter_google_gemma-3-{model_size}/folds_colors_chat_format/games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
        }
        
        print(f"\nProcessing model directory: {model_dir}")
        print(f"Using training data for model: {model_size}")
        
        # Find all detailed predictions files
        detailed_pred_files = list(Path(model_dir).glob("*detailed_predictions.json"))
        
        for pred_file in detailed_pred_files:
            print(f"Processing: {pred_file.name}")
            
            # Extract fold name from filename
            filename = pred_file.name
            if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal_eval_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
                fold_name = "cot_ascii_mask2_sycophancy_off-policy_truth_is_universal"
            elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy_eval_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
                fold_name = "games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy"
            else:
                print(f"Could not determine fold name from filename: {filename}")
                continue
            
            # Load training data to get task mappings
            training_data_path = training_data_paths[fold_name]
            train_file = Path(training_data_path) / "train.jsonl"
            test_file = Path(training_data_path) / "test.jsonl"
            
            if not train_file.exists() or not test_file.exists():
                print(f"Training data files not found for fold: {fold_name}")
                continue
            
            # Load training data and create message to task mapping
            message_to_task = {}
            
            # Load train data
            with open(train_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    messages_str = json.dumps(data['messages'], sort_keys=True)
                    message_to_task[messages_str] = data.get('task', 'unknown')
            
            # Load test data
            with open(test_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    messages_str = json.dumps(data['messages'], sort_keys=True)
                    message_to_task[messages_str] = data.get('task', 'unknown')
            
            print(f"Loaded {len(message_to_task)} message-to-task mappings for {fold_name}")
            
            # Load detailed predictions
            with open(pred_file, 'r') as f:
                predictions_data = json.load(f)
            
            # Add task keys to predictions
            updated_predictions = []
            matched_count = 0
            
            for prediction in predictions_data['predictions']:
                messages = prediction['messages']
                messages_str = json.dumps(messages, sort_keys=True)
                
                # Find matching task
                task = message_to_task.get(messages_str, 'unknown')
                
                # Add task to prediction
                prediction['task'] = task
                updated_predictions.append(prediction)
                
                if task != 'unknown':
                    matched_count += 1
            
            # Update the data
            predictions_data['predictions'] = updated_predictions
            
            # Save to new file with "with_task" suffix
            output_dir = Path(model_dir)
            output_filename = pred_file.name.replace('.json', '_with_task.json')
            output_path = output_dir / output_filename
            
            with open(output_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            print(f"Saved updated predictions to: {output_path}")
            print(f"Matched {matched_count}/{len(updated_predictions)} predictions with tasks")

def add_color_fold_keys_to_detailed_predictions(timestamp_dir, subdir):
    """
    Add color_fold key to each prediction entry by mapping tasks to their taxonomy categories.
    
    Args:
        timestamp_dir: Timestamp directory (e.g., "20250728_005156")
        subdir: Subdirectory (e.g., "lr_0.0002_complete")
    """
    print("="*80)
    print("Adding color_fold keys to detailed predictions...")
    print("="*80)
    
    # Load taxonomy mappings
    taxonomy_path = Path("/workspace/lie-detector/common/simple_lie_taxonomy.csv")
    if not taxonomy_path.exists():
        print(f"Taxonomy file not found: {taxonomy_path}")
        return
    
    # Load taxonomy mappings
    task_to_category = {}
    with open(taxonomy_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row['Task']
            category = row['colors']
            task_to_category[task] = category
    
    print(f"Loaded {len(task_to_category)} task-to-category mappings from taxonomy")
    
    # Helper function to normalize task names (handle hyphen vs underscore differences)
    def normalize_task_name(task_name):
        """Normalize task name to match taxonomy format."""
        # Keep certain tasks with hyphens as-is since they exist in taxonomy with hyphens
        hyphen_tasks = [
            'alibi-multi_workplace_investigation',
            'alibi-multi_theft_investigation', 
            'alibi-multi_fraud_investigation',
            'alibi-multi_assault_investigation',
            'alibi-multi_breaking_entering_investigation',
            'off-policy_dolus_chat',
            'off-policy_truth_is_universal'
        ]
        
        if task_name in hyphen_tasks:
            return task_name
        
        # For other tasks, replace hyphens with underscores
        normalized = task_name.replace('-', '_')
        return normalized
    
    # Base paths
    base_output_dir = Path(f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}")
    
    # Find model directories
    model_dirs = []
    for model_dir in base_output_dir.glob("*google_gemma-3-*"):
        if model_dir.is_dir():
            model_dirs.append(model_dir)
    
    if not model_dirs:
        print("No model directories found")
        return
    
    print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        print(f"\nProcessing model directory: {model_dir}")
        
        # Process detailed prediction files
        detailed_prediction_files = list(model_dir.glob("*detailed_predictions_with_task.json"))
        
        if not detailed_prediction_files:
            print("No detailed prediction files with task keys found")
            continue
        
        for pred_file in detailed_prediction_files:
            print(f"Processing: {pred_file.name}")
            
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                
                # Add color_fold key to each prediction
                updated_count = 0
                unknown_tasks = set()
                for prediction in data.get('predictions', []):
                    if 'task' in prediction:
                        task = prediction['task']
                        normalized_task = normalize_task_name(task)
                        
                        if normalized_task in task_to_category:
                            prediction['color_fold'] = task_to_category[normalized_task]
                            updated_count += 1
                        else:
                            print(f"  Warning: Task '{task}' (normalized: '{normalized_task}') not found in taxonomy")
                            prediction['color_fold'] = "unknown"
                            unknown_tasks.add(task)
                
                # Save updated file
                output_file = pred_file.parent / f"{pred_file.stem}_with_color_fold.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"  Updated {updated_count}/{len(data.get('predictions', []))} predictions with color_fold")
                if unknown_tasks:
                    print(f"  Unknown tasks: {list(unknown_tasks)}")
                print(f"  Saved to: {output_file}")
                
            except Exception as e:
                print(f"  Error processing {pred_file}: {e}")
    
    print("\n" + "="*80)
    print("Color fold key addition completed!")
    print("="*80)

def create_simple_fine_grained_plot(model_dir, metric, output_dir):
    """
    Create a simple fine-grained grid plot using color_fold key from detailed predictions.
    
    Args:
        model_dir: Path to model directory
        metric: Metric to plot ('auc', 'accuracy', 'f1')
        output_dir: Directory to save the plot
    """
    # Load detailed predictions - prefer processed files with unified prefix
    detailed_prediction_files = list(model_dir.glob("*detailed_predictions_with_task_and_color_fold.json"))
    
    # If no processed files found, look for original files
    if not detailed_prediction_files:
        detailed_prediction_files = list(model_dir.glob("*detailed_predictions.json"))
        # Filter out any files that already have partial suffixes
        detailed_prediction_files = [f for f in detailed_prediction_files 
                                   if not any(suffix in f.name for suffix in 
                                             ['_with_task.json', '_with_color_fold.json', '_with_task_with_color_fold.json'])]
    
    print(f"Looking for files in: {model_dir}")
    print(f"Found {len(detailed_prediction_files)} files")
    for f in detailed_prediction_files:
        print(f"  {f.name}")
    
    if not detailed_prediction_files:
        print(f"No detailed prediction files with color_fold found in {model_dir}")
        return None
    
    # Group predictions by train fold and color_fold category
    train_fold_predictions = {}
    
    for pred_file in detailed_prediction_files:
        # Determine train fold from filename
        filename = pred_file.name
        if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in filename:
            train_fold = "Group 1"
            print(f"    -> Group 1")
        elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in filename:
            train_fold = "Group 2"
            print(f"    -> Group 2")
        else:
            print(f"    -> Unknown train fold, skipping")
            continue
        
        try:
            with open(pred_file, 'r') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
            print(f"  Read {len(predictions)} predictions from {pred_file.name}")
            print(f"  About to process predictions...")
            
            # Add task and color_fold keys if they don't exist
            if predictions and 'task' not in predictions[0]:
                print(f"  Adding task and color_fold keys to {pred_file.name}")
                # Build complete task mapping
                complete_task_mappings, _ = build_complete_task_mapping()
                task_to_category_mapping = create_complete_task_name_mapping()
                
                updated_count = 0
                for prediction in predictions:
                    prediction_messages = prediction.get('messages', [])
                    prediction_messages_str = json.dumps(prediction_messages, sort_keys=True)
                    
                    # Find matching task using complete mapping
                    matched_task = complete_task_mappings.get(prediction_messages_str, "unknown")
                    
                    if matched_task != "unknown":
                        prediction['task'] = matched_task
                        # Find category using the mapping
                        category = find_task_category(matched_task, task_to_category_mapping)
                        prediction['color_fold'] = category
                        updated_count += 1
                    else:
                        prediction['task'] = "unknown"
                        prediction['color_fold'] = "unknown"
                
                print(f"  Updated {updated_count}/{len(predictions)} predictions with task and color_fold")
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Error reading {pred_file}: {e}")
            print(f"  Skipping this file...")
            continue
        
        if train_fold not in train_fold_predictions:
            train_fold_predictions[train_fold] = {}
        
        # Group predictions by color_fold category
        try:
            print(f"    Starting to process {len(predictions)} predictions...")
            valid_preds = 0
            color_fold_values = []
            for pred in predictions:
                if 'color_fold' in pred and pred['color_fold'] != 'unknown':
                    color_fold = pred['color_fold']
                    if color_fold not in train_fold_predictions[train_fold]:
                        train_fold_predictions[train_fold][color_fold] = []
                    train_fold_predictions[train_fold][color_fold].append(pred)
                    valid_preds += 1
                    color_fold_values.append(color_fold)
            print(f"    Added {valid_preds} valid predictions to {train_fold}")
            if valid_preds > 0:
                unique_color_folds = list(set(color_fold_values))
                print(f"    Color folds: {unique_color_folds}")
            else:
                # Check what color_fold values exist
                all_color_folds = [p.get('color_fold', 'MISSING') for p in predictions[:10]]
                print(f"    Sample color_fold values: {all_color_folds}")
        except Exception as e:
            print(f"    Error processing predictions: {e}")
            continue
    
    if not train_fold_predictions:
        print(f"No valid predictions found in {model_dir}")
        return None
    
    # Get all unique categories
    all_categories = set()
    for train_fold, categories in train_fold_predictions.items():
        all_categories.update(categories.keys())
    
    if not all_categories:
        print(f"No valid categories found in {model_dir}")
        return None
    
    # Sort categories for consistent ordering
    categories = sorted(list(all_categories))
    train_folds = sorted(list(train_fold_predictions.keys()))
    
    # Create the matrix
    matrix = np.zeros((len(train_folds), len(categories)))
    
    # Fill the matrix with scores
    for i, train_fold in enumerate(train_folds):
        for j, category in enumerate(categories):
            predictions = train_fold_predictions[train_fold].get(category, [])
            
            if len(predictions) < 2:  # Need at least 2 predictions for meaningful metrics
                matrix[i, j] = np.nan
                continue
            
            # Convert labels to numeric
            true_labels = [1 if p['true_label'] == 'B' else 0 for p in predictions]
            pred_labels = [1 if p['predicted_label'] == 'B' else 0 for p in predictions]
            
            # Calculate metric
            if metric == 'auc':
                try:
                    score = roc_auc_score(true_labels, pred_labels)
                except ValueError:
                    score = np.nan
            elif metric == 'accuracy':
                score = accuracy_score(true_labels, pred_labels)
            elif metric == 'f1':
                score = f1_score(true_labels, pred_labels)
            else:
                score = np.nan
            
            matrix[i, j] = score
    
    # Create the heatmap
    plt.figure(figsize=(max(12, len(categories) * 1.2), 10))
    
    # Add baseline rows if available (skip for AUC since baseline doesn't have AUC scores)
    baseline_results = process_baseline_results(None)
    # Extract model size from the full model name
    model_name = model_dir.name.split('_')[2].lower()  # e.g., 'gemma-3-12b-it'
    if '4b' in model_name:
        model_size = '4b'
    elif '12b' in model_name:
        model_size = '12b'
    elif '27b' in model_name:
        model_size = '27b'
    else:
        model_size = 'unknown'
    
    if baseline_results and metric != 'auc':  # Skip baseline rows for AUC plots
        # Add each baseline type as a separate row
        baseline_rows = []
        baseline_row_names = []
        
        for baseline_type in ['llama_chat', 'llama_chat_reasoning', 'rowans_escaped_transcript']:
            if baseline_type in baseline_results:
                baseline_scores = []
                for category in categories:
                    baseline_score = get_baseline_score_for_category_and_type(baseline_results, category, metric, model_size, baseline_type)
                    if baseline_score is not None:
                        baseline_scores.append(baseline_score)
                    else:
                        baseline_scores.append(np.nan)  # Use NaN if no baseline available
                
                baseline_rows.append(baseline_scores)
                baseline_row_names.append(f'Baseline ({baseline_type})')
        
        if baseline_rows:
            # Add baseline rows to matrix
            baseline_matrix = np.vstack([matrix, np.array(baseline_rows)])
            train_folds_with_baseline = train_folds + baseline_row_names
            df = pd.DataFrame(baseline_matrix, index=train_folds_with_baseline, columns=categories)
            print(f"Added {len(baseline_rows)} baseline rows: {baseline_row_names}")
        else:
            df = pd.DataFrame(matrix, index=train_folds, columns=categories)
            print("No baseline rows added")
    else:
        if metric == 'auc':
            print("Skipping baseline rows for AUC plots (baseline results don't contain AUC scores)")
        df = pd.DataFrame(matrix, index=train_folds, columns=categories)
    
    # Use green gradient for all metrics
    cmap = "Greens"
    title = f"{metric.title()} Scores: {model_dir.name.split('_')[2]} Model"
    
    # Create heatmap without colorbar
    sns.heatmap(df, 
                annot=True, 
                fmt=".3f", 
                cmap=cmap, 
                cbar=False,  # Remove colorbar
                vmin=0.0, 
                vmax=1.0,
                center=0.5,
                square=True,
                mask=np.isnan(df))  # Mask NaN values to show them as white
    
    # Add asterisk annotations for accuracy plots to show which categories were present in training
    if metric == 'accuracy':
        # Define which categories belong to each group
        group1_categories = ['cot', 'ascii', 'mask2', 'sycophancy', 'off-policy_truth_is_universal']
        group2_categories = ['games', 'mask1', 'off-policy_dolus_chat', 'sandbagging', 'self_sycophancy']
        
        # Use train_folds_with_baseline if it exists, otherwise use train_folds
        folds_for_asterisks = train_folds_with_baseline if 'train_folds_with_baseline' in locals() else train_folds
        
        # Add asterisks to show which categories were present in each training fold
        for i, fold in enumerate(folds_for_asterisks):
            if fold.startswith('Baseline'):
                continue  # Skip baseline rows
                
            # Determine which group this fold belongs to
            fold_lower = fold.lower().replace(' ', '').replace('\n', '')
            if any(group1_cat in fold_lower for group1_cat in group1_categories):
                # This is a Group 1 fold, add asterisks to Group 1 categories
                for j, category in enumerate(categories):
                    if category in group1_categories:
                        plt.text(j + 0.5, i + 0.8, '*', ha='center', va='center', 
                                fontsize=10, fontweight='bold', color='white')
            elif any(group2_cat in fold_lower for group2_cat in group2_categories):
                # This is a Group 2 fold, add asterisks to Group 2 categories
                for j, category in enumerate(categories):
                    if category in group2_categories:
                        plt.text(j + 0.5, i + 0.8, '*', ha='center', va='center', 
                                fontsize=10, fontweight='bold', color='white')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Test Sets", fontsize=12)
    plt.ylabel("Train Sets", fontsize=12)
    
    # Format x-axis labels: replace _/- with spaces and add line breaks every two words
    x_labels = []
    for cat in categories:
        # Replace underscores and hyphens with spaces
        formatted = cat.replace('_', ' ').replace('-', ' ')
        # Split into words and add line breaks every two words
        words = formatted.split()
        if len(words) <= 2:
            x_labels.append(formatted)
        else:
            formatted_parts = []
            for i in range(0, len(words), 2):
                formatted_parts.append(' '.join(words[i:i+2]))
            x_labels.append('\n'.join(formatted_parts))
    
    # Format y-axis labels: replace _/- with spaces and add line breaks every two words
    y_labels = []
    # Use train_folds_with_baseline if it exists, otherwise use train_folds
    folds_for_labels = train_folds_with_baseline if 'train_folds_with_baseline' in locals() else train_folds
    for fold in folds_for_labels:
        if fold.startswith('Baseline'):
            # Add newline after "Baseline" for baseline labels
            baseline_type = fold.replace('Baseline (', '').replace(')', '')
            y_labels.append(f'Baseline\n({baseline_type})')
        else:
            # Replace underscores and hyphens with spaces
            formatted = fold.replace('_', ' ').replace('-', ' ')
            # Split into words and add line breaks every two words
            words = formatted.split()
            if len(words) <= 2:
                y_labels.append(formatted)
            else:
                formatted_parts = []
                for i in range(0, len(words), 2):
                    formatted_parts.append(' '.join(words[i:i+2]))
                y_labels.append('\n'.join(formatted_parts))
    
    # Calculate row and column averages
    row_averages = df.mean(axis=1, skipna=True)
    col_averages = df.mean(axis=0, skipna=True)
    
    # Set the formatted labels with proper alignment and smaller font
    # Position ticks at the center of each box
    plt.xticks(np.arange(len(categories)) + 0.5, x_labels, rotation=0, ha='center', va='top', fontsize=6)
    plt.yticks(np.arange(len(folds_for_labels)) + 0.5, y_labels, rotation=0, ha='right', va='center', fontsize=6)
    
    # Add row averages on the right side
    for i, (row_name, avg) in enumerate(row_averages.items()):
        if not np.isnan(avg):
            plt.text(len(categories) + 0.5, i + 0.5, f'{avg:.3f}', 
                    ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Add "Avg" label for row averages
    plt.text(len(categories) + 0.5, -0.3, 'Avg', ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Function to map groups to their color folds
    def get_group_color_folds():
        group1_folds = ['cot', 'ascii', 'mask2', 'sycophancy', 'off-policy_truth_is_universal']
        group2_folds = ['games', 'mask1', 'off-policy_dolus_chat', 'sandbagging', 'self_sycophancy']
        
        # Replace _ and - with spaces in the fold names
        def format_fold_name(fold):
            return fold.replace('_', ' ').replace('-', ' ')
        
        return {
            'Group 1': ', '.join([format_fold_name(fold) for fold in group1_folds]),
            'Group 2': ', '.join([format_fold_name(fold) for fold in group2_folds])
        }
    
    # Get group color folds
    group_folds = get_group_color_folds()
    group1_name = group_folds['Group 1']
    group2_name = group_folds['Group 2']
    legend_text = f"Group 1: {group1_name}\nGroup 2: {group2_name}"
    
    # Add legend at the bottom with centered box but left-aligned text
    # First, get the text width to calculate the offset for centering
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    
    # Create a temporary text object to measure its width
    temp_text = plt.figtext(0.5, 0.02, legend_text, ha='left', va='bottom', fontsize=10, 
                           bbox=bbox_props, transform=fig.transFigure)
    bbox = temp_text.get_window_extent(renderer)
    text_width = bbox.width / fig.get_window_extent().width
    
    # Remove the temporary text
    temp_text.remove()
    
    # Calculate the x position to center the box
    x_pos = 0.5 - (text_width / 2)
    
    # Add the text with the calculated position to center the box
    plt.figtext(x_pos, 0.12, legend_text, ha='left', va='bottom', fontsize=10, 
                bbox=bbox_props, transform=fig.transFigure)
    
    plt.tight_layout()
    
    # Save the plot
    model_size_lower = model_dir.name.split("_")[2].lower()
    output_path = Path(output_dir) / f"simple_fine_grained_{metric}_{model_size_lower}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved simple fine-grained {metric} grid plot for {model_size_lower} to: {output_path}")
    print(f"Categories for {metric}: {list(categories)}")
    for train_fold in train_folds:
        for category in categories:
            score = matrix[train_folds.index(train_fold), categories.index(category)]
            if not np.isnan(score):
                print(f"  {train_fold} -> {category}: {score:.3f}")
    
    return output_path

def create_simple_fine_grained_scaling_grid_plot(timestamp_dir, output_dir):
    """
    Create a combined scaling grid plot for simplified fine-grained plots.
    
    Args:
        timestamp_dir: Timestamp directory (e.g., "20250728_005156")
        output_dir: Directory to save the combined plot
    """
    print("="*80)
    print("Creating simple fine-grained scaling grid plot...")
    print("="*80)
    
    # Base paths
    base_output_dir = Path(f"/workspace/lie-detector/visualizations/{timestamp_dir}")
    
    # Find all simple fine-grained plot files
    plot_files = list(base_output_dir.rglob("simple_fine_grained_*.png"))
    
    if not plot_files:
        print("No simple fine-grained plot files found")
        return
    
    print(f"Found {len(plot_files)} simple fine-grained plot files")
    
    # Group files by metric and model size
    metric_files = {}
    for plot_file in plot_files:
        filename = plot_file.name.lower()
        
        # Extract metric and model size from filename
        if "simple_fine_grained_auc_" in filename:
            metric = "auc"
        elif "simple_fine_grained_accuracy_" in filename:
            metric = "accuracy"
        elif "simple_fine_grained_f1_" in filename:
            metric = "f1"
        else:
            continue
        
        # Extract model size
        if "4b-it" in filename:
            model_size = "4b"
        elif "12b-it" in filename:
            model_size = "12b"
        elif "27b-it" in filename:
            model_size = "27b"
        else:
            continue
        
        if metric not in metric_files:
            metric_files[metric] = {}
        metric_files[metric][model_size] = plot_file
    
    # Define the order we want
    metrics = ['auc', 'accuracy', 'f1']
    model_sizes = ['4b', '12b', '27b']
    
    # Check which models we have
    available_model_sizes = []
    for model_size in model_sizes:
        if any(model_size in metric_files.get(metric, {}) for metric in metrics):
            available_model_sizes.append(model_size)
    
    if not available_model_sizes:
        print("No valid model sizes found")
        return
    
    print(f"Available model sizes: {available_model_sizes}")
    
    # Load and resize images
    images = []
    for metric in metrics:
        row_images = []
        for model_size in available_model_sizes:
            if metric in metric_files and model_size in metric_files[metric]:
                img_path = metric_files[metric][model_size]
                img = Image.open(img_path)
                # Resize to a standard size
                img = img.resize((800, 600), Image.Resampling.LANCZOS)
                row_images.append(img)
            else:
                # Create a blank image if file doesn't exist
                blank_img = Image.new('RGB', (800, 600), 'white')
                row_images.append(blank_img)
        images.append(row_images)
    
    # Calculate grid dimensions
    grid_width = len(available_model_sizes)
    grid_height = len(metrics)
    
    # Create the combined image
    img_width, img_height = images[0][0].size
    combined_width = grid_width * img_width
    combined_height = grid_height * img_height
    
    # Add extra height for legend
    legend_height = 100
    combined_height += legend_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
    
    # Paste images into the grid
    for row_idx, row_images in enumerate(images):
        for col_idx, img in enumerate(row_images):
            x = col_idx * img_width
            y = row_idx * img_height
            combined_img.paste(img, (x, y))
    
    # Add legend at the bottom
    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    legend_text = f"Fine-Grained Performance by Taxonomy Category\nRows: {', '.join(metrics)} | Columns: {', '.join(available_model_sizes)}"
    text_bbox = draw.textbbox((0, 0), legend_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (combined_width - text_width) // 2
    text_y = combined_height - legend_height + 20
    
    draw.text((text_x, text_y), legend_text, fill='black', font=font)
    
    # Save the combined image
    output_path = Path(output_dir) / "simple_fine_grained_scaling_grid_plot.png"
    combined_img.save(output_path, 'PNG', dpi=(300, 300))
    
    print(f"Saved simple fine-grained scaling grid plot to: {output_path}")
    print(f"Grid layout: {grid_height} rows x {grid_width} columns")
    print(f"Rows (top to bottom): {', '.join(metrics)}")
    print(f"Columns (left to right): {', '.join(available_model_sizes)}")
    
    return output_path

def add_task_and_color_fold_to_all_predictions(timestamp_dir, subdir):
    """
    Add task and color_fold keys to ALL detailed prediction JSON files using complete task mapping.
    
    Args:
        timestamp_dir: Timestamp directory (e.g., "20250728_005156")
        subdir: Subdirectory (e.g., "lr_0.0002_complete")
    """
    print("="*80)
    print("Adding task and color_fold keys to ALL detailed predictions using complete task mapping...")
    print("="*80)
    
    # Build complete task mapping from ALL JSONL files
    complete_task_mappings, all_task_counts = build_complete_task_mapping()
    
    # Load taxonomy mappings
    task_to_category_mapping = create_complete_task_name_mapping()
    
    # Base paths
    base_output_dir = Path(f"/workspace/lie-detector/outputs/{timestamp_dir}/{subdir}")
    
    # Find model directories
    model_dirs = []
    for model_dir in base_output_dir.glob("*google_gemma-3-*"):
        if model_dir.is_dir():
            model_dirs.append(model_dir)
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Process each model directory
    for model_dir in model_dirs:
        print(f"\nProcessing model directory: {model_dir}")
        
        # Find detailed prediction files that need updating
        detailed_prediction_files = []
        for pred_file in model_dir.glob("*detailed_predictions.json"):
            if "_with_task_and_color_fold.json" in pred_file.name:
                continue
            detailed_prediction_files.append(pred_file)
        
        if not detailed_prediction_files:
            print(f"No detailed prediction files found in {model_dir} that need updating")
            continue
        
        print(f"Found {len(detailed_prediction_files)} detailed prediction files to update")
        
        for pred_file in detailed_prediction_files:
            print(f"\nProcessing: {pred_file.name}")
            
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error reading {pred_file}: {e}")
                continue
            
            try:
                # Add task and color_fold keys to each prediction
                updated_count = 0
                task_counts = Counter()
                
                for prediction in data.get('predictions', []):
                    prediction_messages = prediction.get('messages', [])
                    prediction_messages_str = json.dumps(prediction_messages, sort_keys=True)
                    
                    # Find matching task using complete mapping
                    matched_task = complete_task_mappings.get(prediction_messages_str, "unknown")
                    
                    if matched_task != "unknown":
                        prediction['task'] = matched_task
                        # Find category using the mapping
                        category = find_task_category(matched_task, task_to_category_mapping)
                        prediction['color_fold'] = category
                        updated_count += 1
                        task_counts[matched_task] += 1
                    else:
                        prediction['task'] = "unknown"
                        prediction['color_fold'] = "unknown"
                
                # Save updated file
                output_file = pred_file.parent / f"{pred_file.stem}_with_task_and_color_fold.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"  Updated {updated_count}/{len(data.get('predictions', []))} predictions with task and color_fold")
                print(f"  Task distribution:")
                for task, count in task_counts.most_common(5):
                    print(f"    {task}: {count}")
                print(f"  Saved to: {output_file}")
                
            except Exception as e:
                print(f"  Error processing {pred_file}: {e}")
                continue

def process_baseline_results(baseline_dir):
    """
    Process baseline results from the baseline directory and organize by baseline type and fold.
    
    Args:
        baseline_dir: Path to baseline results directory
        
    Returns:
        Dict with structure: {baseline_type: {fold: {model_size: {metric: score}}}}
    """
    baseline_dir = Path(baseline_dir) if baseline_dir else Path("/workspace/lie-detector/outputs/baseline_results_250")
    
    if not baseline_dir.exists():
        print(f"Warning: Baseline directory {baseline_dir} does not exist")
        return {}
    
    baseline_results = {}
    
    # Look for results in the baseline directory
    for result_file in baseline_dir.rglob("results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract model size from filename or metadata
            model = data.get('metadata', {}).get('model', '')
            if '4b' in model.lower():
                model_size = '4b'
            elif '12b' in model.lower():
                model_size = '12b'
            elif '27b' in model.lower():
                model_size = '27b'
            else:
                continue
            
            # Extract baseline type from path (e.g., llama_chat, llama_chat_reasoning, rowans_escaped_transcript)
            baseline_type = None
            for part in result_file.parts:
                if part in ['llama_chat', 'llama_chat_reasoning', 'rowans_escaped_transcript']:
                    baseline_type = part
                    break
            
            if not baseline_type:
                continue
            
            # Extract fold name from path
            fold_name = None
            for part in result_file.parts:
                if part.endswith('_train'):
                    fold_name = part.replace('_train', '')
                    break
            
            if fold_name:
                # Initialize baseline type if not exists
                if baseline_type not in baseline_results:
                    baseline_results[baseline_type] = {}
                
                # Initialize fold if not exists
                if fold_name not in baseline_results[baseline_type]:
                    baseline_results[baseline_type][fold_name] = {}
                
                # Extract metrics
                overall_metrics = data.get('overall_metrics', {})
                baseline_results[baseline_type][fold_name][model_size] = {
                    'accuracy': overall_metrics.get('accuracy', 0.0),
                    'f1': overall_metrics.get('f1_score', 0.0),
                    'precision': overall_metrics.get('precision', 0.0),
                    'recall': overall_metrics.get('recall', 0.0)
                }
                
        except Exception as e:
            print(f"Error loading baseline result {result_file}: {e}")
    
    return baseline_results

def get_baseline_score_for_category_and_type(baseline_results, category, metric, model_size, baseline_type):
    """Get baseline score for a specific category, metric, model size, and baseline type."""
    if not baseline_results or baseline_type not in baseline_results:
        return None
    
    # Map categories to baseline fold names - use direct mapping since baseline folds match plot categories
    category_to_fold = {
        'sandbagging': 'sandbagging',
        'off-policy_dolus_chat': 'off-policy_dolus_chat',
        'off-policy_truth_is_universal': 'off-policy_truth_is_universal',
        'ascii': 'ascii',
        'cot': 'cot',
        'games': 'games',
        'mask1': 'mask1',
        'mask2': 'mask2',
        'sycophancy': 'sycophancy',
        'self_sycophancy': 'self_sycophancy'
    }
    
    fold_name = category_to_fold.get(category)
    if not fold_name or fold_name not in baseline_results[baseline_type]:
        return None
    
    if model_size not in baseline_results[baseline_type][fold_name]:
        return None
    
    return baseline_results[baseline_type][fold_name][model_size].get(metric, None)

def create_unseen_violin_plot(model_dir: Path, output_dir: Path) -> Optional[Path]:
    """Create a single combined violin plot of deception score (A_prob_normalized) on unseen categories.

    For files trained on Group 1, include only Group 2 categories; for files trained on Group 2, include only
    Group 1 categories. Both are plotted together with a hue indicating the training group.
    """
    # Identify processed detailed prediction files (with task and color_fold)
    pred_files: List[Path] = []
    pred_files.extend(model_dir.glob("*detailed_predictions_with_task_and_color_fold.json"))
    pred_files.extend(model_dir.glob("*detailed_predictions_with_task_with_color_fold.json"))
    if not pred_files:
        print(f"No detailed prediction files with task and color_fold found in {model_dir}")
        return None

    group1_categories = ['cot', 'ascii', 'mask2', 'sycophancy', 'off-policy_truth_is_universal']
    group2_categories = ['games', 'mask1', 'off-policy_dolus_chat', 'sandbagging', 'self_sycophancy']

    # Collect combined records across both training folds
    combined_records: List[Dict[str, Any]] = []

    def format_label(label: str) -> str:
        cleaned = label.replace('_', ' ').replace('-', ' ')
        words = cleaned.split()
        if len(words) <= 2:
            return cleaned
        return '\n'.join(' '.join(words[i:i+2]) for i in range(0, len(words), 2))

    for pred_path in pred_files:
        fname = pred_path.name
        if "train_cot_ascii_mask2_sycophancy_off-policy_truth_is_universal" in fname:
            group_label = 'Trained on Group 1'
            allowed_categories = set(group2_categories)
        elif "train_games_mask1_off-policy_dolus_chat_sandbagging_self_sycophancy" in fname:
            group_label = 'Trained on Group 2'
            allowed_categories = set(group1_categories)
        else:
            continue

        # Load predictions
        try:
            with open(pred_path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  Error reading {pred_path}: {exc}")
            continue

        # Collect per-sample deception scores for unseen categories
        for pred in data.get('predictions', []):
            category = pred.get('color_fold', 'unknown')
            if not category or category == 'unknown' or category not in allowed_categories:
                continue
            probs = pred.get('probabilities') or {}
            if 'A_prob_normalized' in probs:
                deception = probs['A_prob_normalized']
            elif 'A_prob' in probs:
                deception = probs['A_prob']
            else:
                continue
            if deception is None:
                continue
            truth_label = pred.get('true_label')
            if truth_label not in ('A', 'B'):
                continue
            truth_name = 'Lie (A)' if truth_label == 'A' else 'Truth (B)'
            combined_records.append({'category': category, 'deception': float(deception), 'group': group_label, 'truth': truth_name})

    if not combined_records:
        print(f"No unseen-category deception scores found for {model_dir}")
        return None

    df_all = pd.DataFrame.from_records(combined_records)
    desired_order = [c for c in (group1_categories + group2_categories) if c in df_all['category'].unique()]
    if not desired_order:
        desired_order = sorted(df_all['category'].unique())

    # Single plot with two violins per category (Lie A vs Truth B), across both training groups
    import matplotlib.pyplot as plt
    plt.figure(figsize=(max(14, len(desired_order) * 1.6), 6))
    palette_truth = {'Lie (A)': '#E15759', 'Truth (B)': '#4E79A7'}
    sns.violinplot(
        data=df_all,
        x='category',
        y='deception',
        hue='truth',
        order=desired_order,
        dodge=True,
        cut=0,
        inner='box',
        density_norm='width',
        palette=palette_truth
    )
    plt.ylim(0.0, 1.0)
    x_tick_labels = [format_label(cat) for cat in desired_order]
    plt.xticks(np.arange(len(desired_order)), x_tick_labels, rotation=0, ha='center', va='top', fontsize=8)
    plt.yticks(fontsize=9)
    plt.legend(title='Ground truth', loc='upper right')

    try:
        model_id = model_dir.name.split('_')[2]
    except Exception:
        model_id = model_dir.name
    plt.title(f"Deception score on unseen categories: {model_id}", fontsize=14, fontweight='bold')
    plt.xlabel('Test Sets', fontsize=12)
    plt.ylabel('Deception score (A_prob_normalized)', fontsize=12)
    plt.tight_layout()

    model_size_lower = (model_id or 'model').lower()
    out_name = f"unseen_violin_deception_{model_size_lower}.png"
    out_path = Path(output_dir) / out_name
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined unseen-fold violin plot to: {out_path}")

    # Also copy into the outputs model directory for convenience
    try:
        import shutil
        shutil.copyfile(out_path, model_dir / out_name)
    except Exception as exc:
        print(f"Warning: could not copy violin to outputs dir: {exc}")

    return out_path

def main():
    parser = argparse.ArgumentParser(description="Visualize k-fold training results for all models")
    parser.add_argument("--timestamp_dir", type=str, required=True,
                        help="Timestamp directory containing model results (e.g., 20250728_005156)")
    parser.add_argument("--subdir", type=str, default="lr_0.0002_complete",
                        help="Subdirectory containing model results (e.g., lr_0.0002_complete)")
    parser.add_argument("--base_output_dir", type=str, default="./visualizations",
                        help="Base output directory for visualizations")
    parser.add_argument("--taxonomy_path", type=str, default="/workspace/lie-detector/common/simple_lie_taxonomy.csv",
                        help="Path to the taxonomy CSV file for fine-grained plots")
    
    args = parser.parse_args()
    
    # Construct full paths - using the custom outputs directory with subdirectory
    timestamp_dir = Path("/workspace/lie-detector/outputs") / args.timestamp_dir / args.subdir
    output_dir = Path(args.base_output_dir) / args.timestamp_dir / args.subdir
    
    if not timestamp_dir.exists():
        print(f"Timestamp directory not found: {timestamp_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing timestamp directory: {timestamp_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all model subdirectories
    model_dirs = [d for d in timestamp_dir.iterdir() if d.is_dir() and d.name.startswith('openrouter_google_gemma-3-')]
    
    if not model_dirs:
        print(f"No model directories found in {timestamp_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Process each model
    successful_models = 0
    for model_dir in model_dirs:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_dir.name}")
        print(f"{'='*80}")
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_dir.name
        model_output_dir.mkdir(exist_ok=True)
        
        # Extract model name for titles
        model_name = model_dir.name.replace('openrouter_google_gemma-3-', 'google/gemma-3-')
        
        # Clean up the title: replace _ and - with spaces, remove everything after _train_one
        title_name = model_name
        if '_train_one' in title_name:
            title_name = title_name.split('_train_one')[0]
        title_name = title_name.replace('_', ' ').replace('-', ' ')
        
        # Create visualizations for this model
        if visualize_model_results(model_dir, model_output_dir, model_name, title_name):
            successful_models += 1
            print(f"Successfully created visualizations for {model_dir.name}")
        else:
            print(f"Failed to create visualizations for {model_dir.name}")
    
    # Create scaling grid plot
    print(f"\n{'='*80}")
    print("Creating scaling grid plot...")
    print(f"{'='*80}")
    create_scaling_grid_plot(output_dir, output_dir)
    
    # Create fine-grained plots (removed complex versions - using simple versions only)
    
    # Add task keys to detailed predictions
    print(f"\n{'='*80}")
    print("Adding task keys to detailed predictions...")
    print(f"{'='*80}")
    add_task_keys_to_detailed_predictions(output_dir, args.timestamp_dir, args.subdir)
    
    # Add color_fold keys to detailed predictions
    print(f"\n{'='*80}")
    print("Adding color_fold keys to detailed predictions...")
    print(f"{'='*80}")
    add_color_fold_keys_to_detailed_predictions(args.timestamp_dir, args.subdir)
    
    # Add task and color_fold keys to ALL detailed predictions
    print(f"\n{'='*80}")
    print("Adding task and color_fold keys to ALL detailed predictions...")
    print(f"{'='*80}")
    add_task_and_color_fold_to_all_predictions(args.timestamp_dir, args.subdir)
    
    # Create simplified fine-grained plots
    print(f"\n{'='*80}")
    print("Creating simplified fine-grained plots...")
    print(f"{'='*80}")
    
    # Find model directories
    base_output_dir = Path(f"/workspace/lie-detector/outputs/{args.timestamp_dir}/{args.subdir}")
    model_dirs = []
    for model_dir in base_output_dir.glob("*google_gemma-3-*"):
        if model_dir.is_dir():
            model_dirs.append(model_dir)
    
    # Create simplified fine-grained plots for each model
    metrics = ['auc', 'accuracy', 'f1']
    for model_dir in model_dirs:
        print(f"\nProcessing model: {model_dir.name}")
        for metric in metrics:
            create_simple_fine_grained_plot(model_dir, metric, output_dir)

    # Create unseen-fold violin plots (one per model)
    for model_dir in model_dirs:
        print(f"\nCreating unseen-fold violin plot for model: {model_dir.name}")
        create_unseen_violin_plot(model_dir, output_dir)
    
    # Create combined scaling grid plot
    print(f"\n{'='*80}")
    print("Creating simplified fine-grained scaling grid plot...")
    print(f"{'='*80}")
    create_simple_fine_grained_scaling_grid_plot(args.timestamp_dir, output_dir)
    
    print(f"\n{'='*80}")
    
    print(f"\n{'='*80}")
    print(f"Summary: Successfully processed {successful_models}/{len(model_dirs)} models")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 