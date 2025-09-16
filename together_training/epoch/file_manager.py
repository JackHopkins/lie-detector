#!/usr/bin/env python3
"""
File Manager for TogetherAI Training

This module handles file validation, caching, and uploading for training data.
It implements smart caching to avoid re-uploading unchanged files.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
from together import Together

from .training_state import TrainingState


class FileManager:
    """Manages file validation, caching, and uploading for TogetherAI training."""
    
    def __init__(self, api_key: str, training_state: TrainingState, base_model: str = "openai/gpt-oss-120b"):
        self.client = Together(api_key=api_key)
        self.training_state = training_state
        self.base_model = base_model
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            raise Exception(f"Error computing hash for {filepath}: {e}")
        
        return hash_sha256.hexdigest()
    
    def _is_oss_model(self) -> bool:
        """Check if the base model is an OSS model requiring Harmony format."""
        return "gpt-oss" in self.base_model.lower()
    
    def _convert_to_harmony(self, fold_path: Path) -> Tuple[Path, Path]:
        """
        Convert JSONL files to Harmony format for OSS models.
        
        Args:
            fold_path: Path to the fold directory
            
        Returns:
            (harmony_train_path, harmony_val_path)
        """
        print("Converting to Harmony format for OSS model...")
        
        harmony_dir = fold_path / "harmony"
        harmony_dir.mkdir(exist_ok=True)
        
        train_jsonl = fold_path / "train.jsonl"
        val_jsonl = fold_path / "val.jsonl"
        
        if not train_jsonl.exists() or not val_jsonl.exists():
            raise Exception(f"Required JSONL files not found in {fold_path}")
        
        train_harmony = harmony_dir / "train_harmony.parquet"
        val_harmony = harmony_dir / "val_harmony.parquet"
        
        # Check if harmony files already exist and are newer than source files
        if (train_harmony.exists() and val_harmony.exists() and
            train_harmony.stat().st_mtime > train_jsonl.stat().st_mtime and
            val_harmony.stat().st_mtime > val_jsonl.stat().st_mtime):
            print("✓ Harmony files already exist and are up to date")
            return train_harmony, val_harmony
        
        # Convert training file
        print("Converting training file to Harmony format...")
        try:
            result = subprocess.run([
                "python", "prep/harmony_converter.py", str(train_jsonl),
                "--output-dir", str(harmony_dir),
                "--reasoning-effort", "low"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Training file conversion failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Training file conversion timed out")
        except FileNotFoundError:
            raise Exception("prep/harmony_converter.py not found. Ensure you're in the correct directory.")
        
        # Convert validation file
        print("Converting validation file to Harmony format...")
        try:
            result = subprocess.run([
                "python", "prep/harmony_converter.py", str(val_jsonl),
                "--output-dir", str(harmony_dir),
                "--reasoning-effort", "low"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Validation file conversion failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Validation file conversion timed out")
        
        if not train_harmony.exists() or not val_harmony.exists():
            raise Exception("Harmony conversion completed but output files not found")
        
        print("✓ Harmony conversion completed successfully")
        return train_harmony, val_harmony
    
    def _get_file_size(self, filepath: Path) -> int:
        """Get file size in bytes."""
        try:
            return filepath.stat().st_size
        except Exception as e:
            raise Exception(f"Error getting size for {filepath}: {e}")
    
    def validate_files(self, fold_path: Path) -> Tuple[bool, str]:
        """
        Validate that required training files exist and are properly formatted.
        
        Returns:
            (success, message)
        """
        train_file = fold_path / "train.jsonl"
        val_file = fold_path / "val.jsonl"
        
        # Check if files exist
        if not train_file.exists():
            return False, f"Training file not found: {train_file}"
        
        if not val_file.exists():
            return False, f"Validation file not found: {val_file}"
        
        # Check file sizes (basic sanity check)
        train_size = self._get_file_size(train_file)
        val_size = self._get_file_size(val_file)
        
        if train_size == 0:
            return False, f"Training file is empty: {train_file}"
        
        if val_size == 0:
            return False, f"Validation file is empty: {val_file}"
        
        # Use Together CLI to validate file format if available
        try:
            # Check training file
            result = subprocess.run(
                ["together", "files", "check", str(train_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return False, f"Training file validation failed: {result.stderr}"
            
            # Check validation file
            result = subprocess.run(
                ["together", "files", "check", str(val_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return False, f"Validation file validation failed: {result.stderr}"
            
            return True, "All files validated successfully"
            
        except subprocess.TimeoutExpired:
            return False, "File validation timed out"
        except FileNotFoundError:
            # Together CLI not available, skip validation
            print("Warning: Together CLI not found, skipping format validation")
            return True, "Files exist (format validation skipped)"
        except Exception as e:
            return False, f"Error during file validation: {e}"
    
    def get_or_upload_file(self, filepath: Path, filename: str) -> str:
        """
        Get file ID for a file, uploading if necessary or using cached ID.
        
        Args:
            filepath: Path to the local file
            filename: Logical name for the file (e.g., 'train.jsonl')
        
        Returns:
            file_id: TogetherAI file ID
        """
        # Compute current file hash
        try:
            current_hash = self._compute_file_hash(filepath)
            file_size = self._get_file_size(filepath)
        except Exception as e:
            raise Exception(f"Error processing file {filepath}: {e}")
        
        # Check if we have a cached version
        if self.training_state.is_file_cached(filename, current_hash):
            cached_file_id = self.training_state.get_file_id(filename)
            print(f"Using cached file ID for {filename}: {cached_file_id}")
            return cached_file_id
        
        # Need to upload the file
        print(f"Uploading {filename} ({file_size:,} bytes)...")
        
        try:
            # Upload file with validation
            file_response = self.client.files.upload(str(filepath), check=True)
            file_id = file_response.id
            
            # Cache the file information
            self.training_state.add_file(filename, file_id, current_hash, file_size)
            
            print(f"✓ Upload successful! File ID: {file_id}")
            return file_id
            
        except Exception as e:
            # Provide helpful error messages
            error_msg = str(e)
            
            if "Missing required fields" in error_msg:
                raise Exception(
                    f"File format validation failed for {filename}. "
                    "Ensure the JSONL file has the correct structure with 'messages' field."
                )
            elif "File too large" in error_msg:
                raise Exception(
                    f"File {filename} is too large for upload. "
                    f"Size: {file_size:,} bytes. Consider splitting the data."
                )
            elif "rate limit" in error_msg.lower():
                raise Exception(
                    f"Rate limit exceeded while uploading {filename}. "
                    "Please wait a moment before retrying."
                )
            else:
                raise Exception(f"Upload failed for {filename}: {error_msg}")
    
    def upload_training_files(self, fold_path: Path) -> Tuple[str, str]:
        """
        Upload both training and validation files, returning their IDs.
        For OSS models, automatically converts to Harmony format first.
        
        Args:
            fold_path: Path to the fold directory
            
        Returns:
            (train_file_id, val_file_id)
        """
        if self._is_oss_model():
            # Convert to Harmony format for OSS models
            train_file, val_file = self._convert_to_harmony(fold_path)
            
            # Upload harmony files with proper naming
            train_file_id = self.get_or_upload_file(train_file, "train_harmony.parquet")
            val_file_id = self.get_or_upload_file(val_file, "val_harmony.parquet")
            
        else:
            # Use JSONL files for regular models
            train_file = fold_path / "train.jsonl"
            val_file = fold_path / "val.jsonl"
            
            # Upload training file
            train_file_id = self.get_or_upload_file(train_file, "train.jsonl")
            
            # Upload validation file
            val_file_id = self.get_or_upload_file(val_file, "val.jsonl")
        
        return train_file_id, val_file_id
    
    def validate_and_upload_files(self, fold_path: Path) -> Tuple[str, str]:
        """
        Validate files and upload them, returning file IDs.
        
        Args:
            fold_path: Path to the fold directory
            
        Returns:
            (train_file_id, val_file_id)
        """
        print("Validating training files...")
        
        # Validate files first
        success, message = self.validate_files(fold_path)
        if not success:
            raise Exception(f"File validation failed: {message}")
        
        print("✓ File validation passed")
        
        # Upload files
        return self.upload_training_files(fold_path)
    
    def clean_cache(self, keep_recent: int = 10) -> None:
        """
        Clean old files from cache, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent files to keep
        """
        files_by_time = sorted(
            self.training_state.files.items(),
            key=lambda x: x[1].upload_time,
            reverse=True
        )
        
        if len(files_by_time) <= keep_recent:
            return
        
        # Remove old files from state
        files_to_remove = files_by_time[keep_recent:]
        for filename, _ in files_to_remove:
            del self.training_state.files[filename]
        
        self.training_state.save_state()
        print(f"Cleaned {len(files_to_remove)} old files from cache")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached files."""
        total_size = sum(info.file_size for info in self.training_state.files.values())
        
        return {
            'total_files': len(self.training_state.files),
            'total_size': total_size,
            'files': {
                filename: {
                    'file_id': info.file_id,
                    'size': info.file_size,
                    'upload_time': info.upload_time,
                    'hash': info.file_hash[:16] + "..."  # Truncated hash for display
                }
                for filename, info in self.training_state.files.items()
            }
        }