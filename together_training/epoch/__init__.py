#!/usr/bin/env python3
"""
Epoch-by-Epoch TogetherAI Training Package

This package provides tools for training models epoch-by-epoch with automatic
resumption, state persistence, and smart file caching.

Main Components:
- TrainingState: Manages persistent training state
- FileManager: Handles file validation and caching  
- EpochTrainer: Main training orchestrator
- train_epochs: CLI interface

Usage:
    # CLI usage
    python -m together_training.epoch.train_epochs <fold_path>
    
    # Programmatic usage
    from together_training.epoch import EpochTrainer
    trainer = EpochTrainer(api_key="your_key")
    result = trainer.train_single_epoch("path/to/fold")
"""

from .training_state import TrainingState, EpochInfo, FileInfo
from .file_manager import FileManager
from .epoch_trainer import EpochTrainer

__version__ = "1.0.0"
__author__ = "Lie Detection Research Team"

__all__ = [
    'TrainingState',
    'EpochInfo', 
    'FileInfo',
    'FileManager',
    'EpochTrainer'
]