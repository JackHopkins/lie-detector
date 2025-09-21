#!/usr/bin/env python3
"""
Run State Manager for Evaluation Resumability

This module manages the state of evaluation runs, allowing for resuming
incomplete evaluations and avoiding recomputation of already completed
epoch/fold combinations.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class EvaluationStatus:
    """Status of a single evaluation (epoch/fold combination)."""
    status: str  # "pending", "running", "completed", "failed"
    eval_folds: List[str]  # Successfully completed evaluation folds
    failed_folds: List[str]  # Failed evaluation folds
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    error_message: Optional[str] = None


@dataclass 
class RunState:
    """Complete state of an evaluation run."""
    run_id: str
    created_time: str
    trained_fold: str
    model_name: str
    base_path: str
    log_base_dir: str
    total_epochs: int
    eval_folds: List[str]
    include_baseline: bool
    completed_evaluations: Dict[str, EvaluationStatus]
    run_parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert EvaluationStatus objects to dictionaries
        result['completed_evaluations'] = {
            key: asdict(value) for key, value in self.completed_evaluations.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunState':
        """Create RunState from dictionary."""
        # Convert evaluation status dictionaries back to objects
        completed_evaluations = {
            key: EvaluationStatus(**value) 
            for key, value in data['completed_evaluations'].items()
        }
        data['completed_evaluations'] = completed_evaluations
        return cls(**data)


class RunStateManager:
    """Manages evaluation run state for resumability."""
    
    def __init__(self, log_base_dir: str = "../../logs"):
        """
        Initialize run state manager.
        
        Args:
            log_base_dir: Base directory for log storage
        """
        self.log_base_dir = Path(log_base_dir)
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"run_{timestamp}"
    
    def get_run_state_path(self, run_id: str) -> Path:
        """Get path to run state file."""
        return self.log_base_dir / run_id / "run_state.json"
    
    def get_log_directory(self, run_id: str, trained_fold: str, epoch_key: str, eval_fold: Optional[str] = None) -> Path:
        """
        Get structured log directory path.
        
        Args:
            run_id: Unique run identifier
            trained_fold: Training fold name
            epoch_key: Epoch identifier (e.g., "epoch_0", "baseline")
            eval_fold: Optional evaluation fold name for more specific path
            
        Returns:
            Path to log directory
        """
        log_path = self.log_base_dir / run_id / trained_fold / epoch_key
        if eval_fold:
            log_path = log_path / eval_fold
        return log_path
    
    def create_run_state(
        self,
        run_id: str,
        trained_fold: str,
        model_name: str,
        base_path: str,
        eval_folds: List[str],
        total_epochs: int,
        include_baseline: bool = False,
        run_parameters: Optional[Dict[str, Any]] = None
    ) -> RunState:
        """
        Create a new run state.
        
        Args:
            run_id: Unique run identifier
            trained_fold: Training fold name
            model_name: Model name
            base_path: Base project path
            eval_folds: List of evaluation fold names
            total_epochs: Number of training epochs
            include_baseline: Whether baseline evaluation is included
            run_parameters: Additional parameters for this run
            
        Returns:
            New RunState object
        """
        current_time = datetime.now().isoformat()
        
        # Initialize empty completion status
        completed_evaluations = {}
        
        # Add baseline if requested
        if include_baseline:
            completed_evaluations["baseline"] = EvaluationStatus(
                status="pending",
                eval_folds=[],
                failed_folds=[]
            )
        
        # Add all epochs
        for epoch in range(total_epochs):
            completed_evaluations[f"epoch_{epoch}"] = EvaluationStatus(
                status="pending", 
                eval_folds=[],
                failed_folds=[]
            )
        
        return RunState(
            run_id=run_id,
            created_time=current_time,
            trained_fold=trained_fold,
            model_name=model_name,
            base_path=base_path,
            log_base_dir=str(self.log_base_dir),
            total_epochs=total_epochs,
            eval_folds=eval_folds,
            include_baseline=include_baseline,
            completed_evaluations=completed_evaluations,
            run_parameters=run_parameters or {}
        )
    
    def save_run_state(self, run_state: RunState) -> None:
        """Save run state to JSON file."""
        state_path = self.get_run_state_path(run_state.run_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_path, 'w') as f:
            json.dump(run_state.to_dict(), f, indent=2)
    
    def load_run_state(self, run_id: str) -> Optional[RunState]:
        """
        Load run state from JSON file.
        
        Args:
            run_id: Run identifier to load
            
        Returns:
            RunState object or None if not found
        """
        state_path = self.get_run_state_path(run_id)
        
        if not state_path.exists():
            return None
        
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            return RunState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load run state from {state_path}: {e}")
            return None
    
    def list_runs(self) -> List[Tuple[str, RunState]]:
        """
        List all available runs with their states.
        
        Returns:
            List of (run_id, run_state) tuples
        """
        runs = []
        
        for run_dir in self.log_base_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                run_state = self.load_run_state(run_dir.name)
                if run_state:
                    runs.append((run_dir.name, run_state))
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x[1].created_time, reverse=True)
        return runs
    
    def find_latest_incomplete_run(self, trained_fold: str, model_name: str) -> Optional[str]:
        """
        Find the most recent incomplete run for a given fold/model.
        
        Args:
            trained_fold: Training fold name
            model_name: Model name
            
        Returns:
            Run ID of latest incomplete run, or None if not found
        """
        runs = self.list_runs()
        
        for run_id, run_state in runs:
            if (run_state.trained_fold == trained_fold and 
                run_state.model_name == model_name and
                not self.is_run_complete(run_state)):
                return run_id
        
        return None
    
    def is_run_complete(self, run_state: RunState) -> bool:
        """Check if a run is completely finished."""
        for eval_status in run_state.completed_evaluations.values():
            if eval_status.status != "completed":
                return False
            # Check if all eval folds were completed
            if set(eval_status.eval_folds) != set(run_state.eval_folds):
                return False
        return True
    
    def get_pending_evaluations(self, run_state: RunState) -> Dict[str, List[str]]:
        """
        Get list of pending evaluations (epoch -> list of eval_folds).
        
        Args:
            run_state: Current run state
            
        Returns:
            Dictionary mapping epoch keys to lists of pending evaluation folds
        """
        pending = {}
        
        for epoch_key, eval_status in run_state.completed_evaluations.items():
            if eval_status.status == "completed":
                # Skip fully completed epochs
                continue
                
            # Find which eval folds are still pending
            completed_folds = set(eval_status.eval_folds)
            all_folds = set(run_state.eval_folds)
            pending_folds = list(all_folds - completed_folds)
            
            if pending_folds:
                pending[epoch_key] = pending_folds
        
        return pending
    
    def mark_evaluation_started(self, run_state: RunState, epoch_key: str) -> None:
        """Mark an evaluation as started."""
        if epoch_key in run_state.completed_evaluations:
            run_state.completed_evaluations[epoch_key].status = "running"
            run_state.completed_evaluations[epoch_key].start_time = datetime.now().isoformat()
        self.save_run_state(run_state)
    
    def mark_evaluation_completed(
        self, 
        run_state: RunState, 
        epoch_key: str, 
        completed_folds: List[str],
        failed_folds: Optional[List[str]] = None
    ) -> None:
        """
        Mark an evaluation as completed for specific folds.
        
        Args:
            run_state: Current run state
            epoch_key: Epoch being evaluated
            completed_folds: List of successfully completed eval folds
            failed_folds: List of failed eval folds
        """
        if epoch_key not in run_state.completed_evaluations:
            return
            
        eval_status = run_state.completed_evaluations[epoch_key]
        
        # Add newly completed folds (avoid duplicates)
        current_completed = set(eval_status.eval_folds)
        current_completed.update(completed_folds)
        eval_status.eval_folds = list(current_completed)
        
        # Update failed folds
        if failed_folds:
            current_failed = set(eval_status.failed_folds)
            current_failed.update(failed_folds)
            eval_status.failed_folds = list(current_failed)
        
        # Check if all eval folds are now completed
        all_folds = set(run_state.eval_folds)
        if current_completed == all_folds:
            eval_status.status = "completed"
            eval_status.completion_time = datetime.now().isoformat()
        elif eval_status.status == "pending":
            eval_status.status = "partial"
        
        self.save_run_state(run_state)
    
    def mark_evaluation_failed(
        self, 
        run_state: RunState, 
        epoch_key: str, 
        error_message: str,
        failed_folds: Optional[List[str]] = None
    ) -> None:
        """Mark an evaluation as failed."""
        if epoch_key in run_state.completed_evaluations:
            eval_status = run_state.completed_evaluations[epoch_key]
            eval_status.status = "failed"
            eval_status.error_message = error_message
            
            if failed_folds:
                current_failed = set(eval_status.failed_folds)
                current_failed.update(failed_folds)
                eval_status.failed_folds = list(current_failed)
        
        self.save_run_state(run_state)
    
    def get_run_summary(self, run_state: RunState) -> Dict[str, Any]:
        """Get a summary of run progress."""
        total_evaluations = len(run_state.completed_evaluations)
        completed_evaluations = sum(
            1 for status in run_state.completed_evaluations.values() 
            if status.status == "completed"
        )
        failed_evaluations = sum(
            1 for status in run_state.completed_evaluations.values()
            if status.status == "failed"
        )
        
        # Count total eval fold combinations
        total_combinations = total_evaluations * len(run_state.eval_folds)
        completed_combinations = sum(
            len(status.eval_folds) for status in run_state.completed_evaluations.values()
        )
        
        return {
            "run_id": run_state.run_id,
            "trained_fold": run_state.trained_fold,
            "model_name": run_state.model_name,
            "created_time": run_state.created_time,
            "total_epochs": total_evaluations,
            "completed_epochs": completed_evaluations,
            "failed_epochs": failed_evaluations,
            "total_eval_combinations": total_combinations,
            "completed_combinations": completed_combinations,
            "completion_percentage": (completed_combinations / total_combinations * 100) if total_combinations > 0 else 0,
            "is_complete": self.is_run_complete(run_state),
            "eval_folds": run_state.eval_folds
        }
    
    def cleanup_old_runs(self, keep_recent: int = 10) -> None:
        """
        Clean up old run directories, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent runs to keep
        """
        runs = self.list_runs()
        
        if len(runs) <= keep_recent:
            return
        
        # Remove old runs
        runs_to_remove = runs[keep_recent:]
        
        for run_id, _ in runs_to_remove:
            run_dir = self.log_base_dir / run_id
            if run_dir.exists():
                import shutil
                shutil.rmtree(run_dir)
                print(f"Removed old run: {run_id}")
        
        print(f"Cleaned up {len(runs_to_remove)} old runs")


def print_run_summary(run_state: RunState, manager: RunStateManager) -> None:
    """Print a formatted summary of run state."""
    summary = manager.get_run_summary(run_state)
    
    print(f"\n{'=' * 60}")
    print(f"RUN SUMMARY: {summary['run_id']}")
    print(f"{'=' * 60}")
    print(f"Trained Fold: {summary['trained_fold']}")
    print(f"Model: {summary['model_name']}")
    print(f"Created: {summary['created_time']}")
    print(f"Progress: {summary['completed_combinations']}/{summary['total_eval_combinations']} "
          f"({summary['completion_percentage']:.1f}%)")
    print(f"Epochs: {summary['completed_epochs']}/{summary['total_epochs']} completed")
    if summary['failed_epochs'] > 0:
        print(f"Failed Epochs: {summary['failed_epochs']}")
    print(f"Status: {'Complete' if summary['is_complete'] else 'Incomplete'}")
    
    # Show detailed epoch status
    print(f"\nEpoch Details:")
    for epoch_key, eval_status in run_state.completed_evaluations.items():
        completed_folds = len(eval_status.eval_folds)
        total_folds = len(run_state.eval_folds)
        status_icon = "✅" if eval_status.status == "completed" else "❌" if eval_status.status == "failed" else "⏳"
        print(f"  {status_icon} {epoch_key:12} - {completed_folds}/{total_folds} folds "
              f"({eval_status.status})")
        
        if eval_status.failed_folds:
            print(f"    Failed folds: {', '.join(eval_status.failed_folds)}")