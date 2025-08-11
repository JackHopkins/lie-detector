#!/usr/bin/env python3
"""
Learning Rate Experiment Analysis and Plotting Script

This script analyzes the learning rate experiments by:
1. Loading training logs from checkpoints for each learning rate and fold
2. Creating plots showing training curves for each fold with all learning rates
3. Identifying the best learning rate for each fold
4. Generating summary statistics

Usage:
    python plot_lr_experiments.py
"""

import json
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LRExperimentAnalyzer:
    def __init__(self, outputs_dir: str = "/workspace/lie-detector/outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.lr_values = []
        self.fold_names = []
        self.results = {}
        
    def discover_experiments(self) -> Tuple[List[str], List[str]]:
        """Discover available learning rates and fold names from the outputs directory."""
        # Find the experiment directory (should be the only one)
        exp_dirs = [d for d in self.outputs_dir.iterdir() if d.is_dir()]
        if not exp_dirs:
            raise ValueError(f"No experiment directories found in {self.outputs_dir}")
        
        exp_dir = exp_dirs[0]  # Use the first (and likely only) experiment directory
        print(f"Found experiment directory: {exp_dir}")
        
        # Extract learning rates from directory names
        lr_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('lr_')]
        lr_values = []
        for lr_dir in lr_dirs:
            lr_str = lr_dir.name.replace('lr_', '')
            try:
                # Convert scientific notation to float
                lr_val = float(lr_str)
                lr_values.append(lr_val)
            except ValueError:
                print(f"Warning: Could not parse learning rate from {lr_dir.name}")
                continue
        
        lr_values.sort()
        print(f"Found learning rates: {lr_values}")
        
        # Find fold names from the first learning rate directory
        if lr_dirs:
            first_lr_dir = lr_dirs[0]
            model_dirs = [d for d in first_lr_dir.iterdir() if d.is_dir()]
            if model_dirs:
                # Look for fold directories within the model directory
                fold_dirs = []
                for model_dir in model_dirs:
                    potential_fold_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('train_on_')]
                    fold_dirs.extend(potential_fold_dirs)
                
                if fold_dirs:
                    fold_names = [d.name.replace('train_on_', '') for d in fold_dirs]
                    fold_names = list(set(fold_names))  # Remove duplicates
                    fold_names.sort()
                else:
                    # Fallback: look for checkpoint directories
                    checkpoint_dirs = []
                    for model_dir in model_dirs:
                        potential_checkpoint_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
                        checkpoint_dirs.extend(potential_checkpoint_dirs)
                    
                    if checkpoint_dirs:
                        # Extract fold names from the parent directory structure
                        fold_names = []
                        for checkpoint_dir in checkpoint_dirs:
                            # Navigate up to find the fold name
                            parent_dir = checkpoint_dir.parent
                            if parent_dir.name.startswith('train_on_'):
                                fold_name = parent_dir.name.replace('train_on_', '')
                                fold_names.append(fold_name)
                        
                        fold_names = list(set(fold_names))
                        fold_names.sort()
                    else:
                        # Default fold names based on the codebase
                        fold_names = ['cot', 'mask1', 'mask2', 'sandbagging', 'self_sycophancy', 'sycophancy']
        else:
            # Default fold names
            fold_names = ['cot', 'mask1', 'mask2', 'sandbagging', 'self_sycophancy', 'sycophancy']
        
        print(f"Found fold names: {fold_names}")
        
        self.lr_values = lr_values
        self.fold_names = fold_names
        return lr_values, fold_names
    
    def load_training_logs(self, lr: float, fold: str) -> Optional[Dict]:
        """Load training logs from checkpoint for a specific learning rate and fold."""
        exp_dir = next(self.outputs_dir.iterdir())
        lr_dir = exp_dir / f"lr_{lr}"
        
        # Find the model directory
        model_dirs = [d for d in lr_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print(f"Warning: No model directories found for lr={lr}")
            return None
        
        # Look for the specific fold training directory
        fold_dir = None
        for model_dir in model_dirs:
            potential_fold_dir = model_dir / f"train_on_{fold}"
            if potential_fold_dir.exists():
                fold_dir = potential_fold_dir
                break
        
        if not fold_dir:
            print(f"Warning: No training directory found for lr={lr}, fold={fold}")
            return None
        
        # Find the latest checkpoint
        checkpoint_dirs = [d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        if not checkpoint_dirs:
            print(f"Warning: No checkpoints found for lr={lr}, fold={fold}")
            return None
        
        # Get the latest checkpoint (highest number)
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.replace('checkpoint-', '')))
        
        # Load trainer_state.json
        trainer_state_file = latest_checkpoint / "trainer_state.json"
        if not trainer_state_file.exists():
            print(f"Warning: No trainer_state.json found in {latest_checkpoint}")
            return None
        
        try:
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            # Extract training history
            log_history = trainer_state.get('log_history', [])
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(log_history)
            
            # Add metadata
            df['lr'] = lr
            df['fold'] = fold
            
            return {
                'data': df,
                'best_metric': trainer_state.get('best_metric'),
                'best_global_step': trainer_state.get('best_global_step'),
                'global_step': trainer_state.get('global_step'),
                'epoch': trainer_state.get('epoch')
            }
            
        except Exception as e:
            print(f"Error loading training logs for lr={lr}, fold={fold}: {e}")
            return None
    
    def analyze_all_experiments(self):
        """Load and analyze all learning rate experiments."""
        print("Loading training logs for all experiments...")
        
        for lr in self.lr_values:
            for fold in self.fold_names:
                print(f"Loading lr={lr}, fold={fold}...")
                result = self.load_training_logs(lr, fold)
                if result:
                    key = (lr, fold)
                    self.results[key] = result
        
        print(f"Loaded {len(self.results)} experiment results")
    
    def create_lr_comparison_plots(self, output_dir: str = "lr_analysis_plots"):
        """Create learning rate comparison plots for each fold."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create plots for each fold
        for fold in self.fold_names:
            print(f"Creating plot for fold: {fold}")
            
            # Collect data for this fold
            fold_data = []
            for (lr, fold_name), result in self.results.items():
                if fold_name == fold:
                    df = result['data'].copy()
                    fold_data.append(df)
            
            if not fold_data:
                print(f"No data found for fold {fold}")
                continue
            
            # Combine all data for this fold
            combined_df = pd.concat(fold_data, ignore_index=True)
            
            # Create the plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Learning Rate Comparison - Fold: {fold}', fontsize=16, fontweight='bold')
            
            # Plot 1: Training Loss
            ax1 = axes[0, 0]
            for lr in sorted(combined_df['lr'].unique()):
                lr_data = combined_df[combined_df['lr'] == lr]
                if 'loss' in lr_data.columns:
                    ax1.plot(lr_data['step'], lr_data['loss'], 
                            marker='o', markersize=3, linewidth=2, 
                            label=f'LR={lr:.0e}')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Training Loss vs Steps')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Evaluation Loss
            ax2 = axes[0, 1]
            for lr in sorted(combined_df['lr'].unique()):
                lr_data = combined_df[combined_df['lr'] == lr]
                if 'eval_loss' in lr_data.columns:
                    eval_data = lr_data.dropna(subset=['eval_loss'])
                    if not eval_data.empty:
                        ax2.plot(eval_data['step'], eval_data['eval_loss'], 
                                marker='s', markersize=4, linewidth=2, 
                                label=f'LR={lr:.0e}')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Evaluation Loss')
            ax2.set_title('Evaluation Loss vs Steps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Learning Rate Schedule
            ax3 = axes[1, 0]
            for lr in sorted(combined_df['lr'].unique()):
                lr_data = combined_df[combined_df['lr'] == lr]
                if 'learning_rate' in lr_data.columns:
                    ax3.plot(lr_data['step'], lr_data['learning_rate'], 
                            marker='^', markersize=3, linewidth=2, 
                            label=f'LR={lr:.0e}')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # Plot 4: Gradient Norm
            ax4 = axes[1, 1]
            for lr in sorted(combined_df['lr'].unique()):
                lr_data = combined_df[combined_df['lr'] == lr]
                if 'grad_norm' in lr_data.columns:
                    ax4.plot(lr_data['step'], lr_data['grad_norm'], 
                            marker='d', markersize=3, linewidth=2, 
                            label=f'LR={lr:.0e}')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Gradient Norm')
            ax4.set_title('Gradient Norm vs Steps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(output_path / f'lr_comparison_{fold}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot for {fold} to {output_path / f'lr_comparison_{fold}.png'}")
    
    def create_summary_table(self, output_dir: str = "lr_analysis_plots"):
        """Create a summary table of the best learning rates for each fold."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        summary_data = []
        
        for fold in self.fold_names:
            fold_results = []
            for (lr, fold_name), result in self.results.items():
                if fold_name == fold:
                    fold_results.append({
                        'lr': lr,
                        'best_metric': result['best_metric'],
                        'best_step': result['best_global_step'],
                        'final_step': result['global_step'],
                        'final_epoch': result['epoch']
                    })
            
            if fold_results:
                # Find the best learning rate based on best_metric (lower is better for loss)
                best_result = min(fold_results, key=lambda x: x['best_metric'] if x['best_metric'] is not None else float('inf'))
                
                summary_data.append({
                    'Fold': fold,
                    'Best_LR': best_result['lr'],
                    'Best_Metric': best_result['best_metric'],
                    'Best_Step': best_result['best_step'],
                    'Final_Step': best_result['final_step'],
                    'Final_Epoch': best_result['final_epoch']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path / 'lr_summary.csv', index=False)
            print(f"Saved summary table to {output_path / 'lr_summary.csv'}")
            
            # Print summary
            print("\n" + "="*80)
            print("LEARNING RATE EXPERIMENT SUMMARY")
            print("="*80)
            print(summary_df.to_string(index=False))
            print("="*80)
    
    def create_final_loss_comparison(self, output_dir: str = "lr_analysis_plots"):
        """Create a bar plot comparing final losses across learning rates for each fold."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect final losses for each fold and learning rate
        final_losses = {}
        
        for fold in self.fold_names:
            final_losses[fold] = {}
            for (lr, fold_name), result in self.results.items():
                if fold_name == fold:
                    df = result['data']
                    if 'eval_loss' in df.columns:
                        # Get the last evaluation loss
                        eval_data = df.dropna(subset=['eval_loss'])
                        if not eval_data.empty:
                            final_loss = eval_data['eval_loss'].iloc[-1]
                            final_losses[fold][lr] = final_loss
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Final Evaluation Loss by Learning Rate for Each Fold', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, fold in enumerate(self.fold_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            fold_data = final_losses[fold]
            
            if fold_data:
                lrs = list(fold_data.keys())
                losses = list(fold_data.values())
                
                # Sort by learning rate
                sorted_data = sorted(zip(lrs, losses))
                lrs, losses = zip(*sorted_data)
                
                bars = ax.bar(range(len(lrs)), losses, alpha=0.7)
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Final Evaluation Loss')
                ax.set_title(f'Fold: {fold}')
                ax.set_xticks(range(len(lrs)))
                ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Highlight the best learning rate
                best_idx = np.argmin(losses)
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(0.8)
                
                # Add value labels on bars
                for j, (bar, loss) in enumerate(zip(bars, losses)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{loss:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.fold_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'final_loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved final loss comparison to {output_path / 'final_loss_comparison.png'}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Learning Rate Experiment Analysis")
        print("="*50)
        
        # Discover experiments
        self.discover_experiments()
        
        # Load all data
        self.analyze_all_experiments()
        
        # Create plots
        self.create_lr_comparison_plots()
        self.create_final_loss_comparison()
        self.create_summary_table()
        
        print("\nAnalysis complete! Check the 'lr_analysis_plots' directory for results.")

def main():
    parser = argparse.ArgumentParser(description='Analyze learning rate experiments')
    parser.add_argument('--outputs-dir', type=str, 
                       default='/workspace/lie-detector/outputs',
                       help='Path to outputs directory')
    parser.add_argument('--output-dir', type=str, 
                       default='lr_analysis_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = LRExperimentAnalyzer(args.outputs_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 