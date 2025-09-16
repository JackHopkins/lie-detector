#!/usr/bin/env python3
"""
Training State Manager for Epoch-by-Epoch TogetherAI Training

This module manages persistent state for incremental training runs,
tracking models, epochs, and file IDs across training sessions.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class EpochInfo:
    """Information about a specific training epoch."""
    model_id: Optional[str] = None
    job_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    learning_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpochInfo':
        return cls(**data)


@dataclass
class FileInfo:
    """Information about uploaded training files."""
    file_id: str
    file_hash: str
    upload_time: float
    file_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileInfo':
        return cls(**data)


class TrainingState:
    """Manages persistent training state for a specific fold."""
    
    def __init__(self, fold_path: str, base_model: str = "openai/gpt-oss-120b"):
        self.fold_path = Path(fold_path).resolve()
        self.base_model = base_model
        self.state_file = self.fold_path / "training.json"
        
        # Initialize state
        self.epochs: Dict[int, EpochInfo] = {}
        self.files: Dict[str, FileInfo] = {}  # filename -> FileInfo
        self.created_time = time.time()
        self.updated_time = time.time()
        
        # Load existing state if available
        self.load_state()
    
    def load_state(self) -> None:
        """Load training state from disk if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Load basic fields
                self.base_model = data.get('base_model', self.base_model)
                self.created_time = data.get('created_time', self.created_time)
                self.updated_time = data.get('updated_time', self.updated_time)
                
                # Load epochs
                epochs_data = data.get('epochs', {})
                self.epochs = {}
                for epoch_str, epoch_data in epochs_data.items():
                    epoch_num = int(epoch_str)
                    self.epochs[epoch_num] = EpochInfo.from_dict(epoch_data)
                
                # Load files
                files_data = data.get('files', {})
                self.files = {}
                for filename, file_data in files_data.items():
                    self.files[filename] = FileInfo.from_dict(file_data)
                    
            except Exception as e:
                print(f"Warning: Could not load training state: {e}")
                print("Starting with fresh state...")
    
    def save_state(self) -> None:
        """Save training state to disk."""
        self.updated_time = time.time()
        
        # Prepare data for serialization
        epochs_data = {}
        for epoch_num, epoch_info in self.epochs.items():
            epochs_data[str(epoch_num)] = epoch_info.to_dict()
        
        files_data = {}
        for filename, file_info in self.files.items():
            files_data[filename] = file_info.to_dict()
        
        data = {
            'fold_path': str(self.fold_path),
            'base_model': self.base_model,
            'created_time': self.created_time,
            'updated_time': self.updated_time,
            'epochs': epochs_data,
            'files': files_data
        }
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with atomic write
        temp_file = self.state_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def get_current_epoch(self) -> int:
        """Get the next epoch to train (highest completed + 1)."""
        if not self.epochs:
            return 0
        
        completed_epochs = [
            epoch for epoch, info in self.epochs.items() 
            if info.status == "completed"
        ]
        
        if not completed_epochs:
            return 0
        
        return max(completed_epochs) + 1
    
    def get_base_model_for_epoch(self, epoch: int) -> str:
        """Get the base model to use for training a specific epoch."""
        if epoch == 0:
            return self.base_model
        
        # Find the most recent completed epoch before this one
        prev_epoch = epoch - 1
        while prev_epoch >= 0:
            if (prev_epoch in self.epochs and 
                self.epochs[prev_epoch].status == "completed" and 
                self.epochs[prev_epoch].model_id):
                return self.epochs[prev_epoch].model_id
            prev_epoch -= 1
        
        # Fallback to base model
        return self.base_model
    
    def add_epoch(self, epoch: int, job_id: str, learning_rate: float = None) -> None:
        """Add a new epoch training job."""
        self.epochs[epoch] = EpochInfo(
            job_id=job_id,
            status="running",
            start_time=time.time(),
            learning_rate=learning_rate
        )
        self.save_state()
    
    def update_epoch(self, epoch: int, model_id: str = None, status: str = None) -> None:
        """Update epoch information."""
        if epoch not in self.epochs:
            raise ValueError(f"Epoch {epoch} not found in training state")
        
        epoch_info = self.epochs[epoch]
        
        if model_id:
            epoch_info.model_id = model_id
        
        if status:
            epoch_info.status = status
            if status == "completed":
                epoch_info.end_time = time.time()
        
        self.save_state()
    
    def get_epoch_info(self, epoch: int) -> Optional[EpochInfo]:
        """Get information about a specific epoch."""
        return self.epochs.get(epoch)
    
    def get_completed_epochs(self) -> List[int]:
        """Get list of completed epoch numbers."""
        return sorted([
            epoch for epoch, info in self.epochs.items()
            if info.status == "completed"
        ])
    
    def get_all_models(self) -> Dict[int, str]:
        """Get all completed model IDs indexed by epoch."""
        models = {}
        for epoch, info in self.epochs.items():
            if info.status == "completed" and info.model_id:
                models[epoch] = info.model_id
        return models
    
    def add_file(self, filename: str, file_id: str, file_hash: str, file_size: int) -> None:
        """Add or update file information."""
        self.files[filename] = FileInfo(
            file_id=file_id,
            file_hash=file_hash,
            upload_time=time.time(),
            file_size=file_size
        )
        self.save_state()
    
    def get_file_id(self, filename: str) -> Optional[str]:
        """Get file ID for a filename if it exists."""
        file_info = self.files.get(filename)
        return file_info.file_id if file_info else None
    
    def is_file_cached(self, filename: str, file_hash: str) -> bool:
        """Check if a file is already uploaded with the same hash."""
        file_info = self.files.get(filename)
        if not file_info:
            return False
        return file_info.file_hash == file_hash
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training progress."""
        completed = self.get_completed_epochs()
        running = [epoch for epoch, info in self.epochs.items() if info.status == "running"]
        failed = [epoch for epoch, info in self.epochs.items() if info.status == "failed"]
        
        return {
            'fold_path': str(self.fold_path),
            'base_model': self.base_model,
            'total_epochs': len(self.epochs),
            'completed_epochs': completed,
            'running_epochs': running,
            'failed_epochs': failed,
            'next_epoch': self.get_current_epoch(),
            'files_cached': list(self.files.keys()),
            'models': self.get_all_models()
        }
    
    def cleanup_failed_epochs(self) -> None:
        """Remove failed epochs from state (to allow retry)."""
        failed_epochs = [epoch for epoch, info in self.epochs.items() if info.status == "failed"]
        for epoch in failed_epochs:
            del self.epochs[epoch]
        if failed_epochs:
            self.save_state()
            print(f"Cleaned up {len(failed_epochs)} failed epochs")