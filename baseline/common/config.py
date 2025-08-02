"""
Centralized configuration management for baseline evaluations.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluations."""
    
    # Model settings
    model_name: str = "openrouter/meta-llama/llama-3.1-8b-instruct"
    
    # Data settings  
    num_samples: Optional[int] = None
    processed_data_dir: str = ".data/openai/gpt_4o"
    
    # S3 settings
    s3_uri: Optional[str] = None  # Disabled by default
    
    # Baseline settings
    baseline_types: list = None
    split_by_model: bool = True
    
    # Output settings
    results_base_dir: str = "baseline/results"
    transcripts_base_dir: str = "baseline/transcripts"
    logs_base_dir: str = "logs"
    
    def __post_init__(self):
        if self.baseline_types is None:
            self.baseline_types = ["escaped_transcript", "rowans_escaped_transcript"]
    
    def get_results_dir(self, baseline_type: str) -> str:
        """Get results directory for a specific baseline type."""
        return os.path.join(self.results_base_dir, baseline_type)
    
    def get_transcripts_dir(self, baseline_type: str) -> str:
        """Get transcripts directory for a specific baseline type."""
        return os.path.join(self.transcripts_base_dir, baseline_type)
    
    def get_log_dir(self, baseline_type: str) -> str:
        """Get log directory for a specific baseline type."""
        return os.path.join(self.logs_base_dir, baseline_type)


class ConfigManager:
    """Manages configuration for baseline evaluations."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._environment_overrides = {}
    
    def add_environment_override(self, key: str, env_var: str, converter=str):
        """Add an environment variable override for a config value."""
        self._environment_overrides[key] = (env_var, converter)
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides to the configuration."""
        for key, (env_var, converter) in self._environment_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    setattr(self.config, key, converter(env_value))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to convert {env_var}={env_value} to {converter.__name__}: {e}")
    
    def setup_directories(self, baseline_type: str):
        """Ensure all required directories exist for a baseline type."""
        directories = [
            self.config.get_results_dir(baseline_type),
            self.config.get_transcripts_dir(baseline_type),
            self.config.get_log_dir(baseline_type)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_data_directory(self) -> bool:
        """Validate that the data directory exists and is accessible."""
        return os.path.exists(self.config.processed_data_dir)
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """Get information about the data source configuration."""
        return {
            "using_s3": self.config.s3_uri is not None,
            "s3_uri": self.config.s3_uri,
            "local_dir": self.config.processed_data_dir,
            "dir_exists": self.validate_data_directory()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.config.model_name,
            "num_samples": self.config.num_samples,
            "processed_data_dir": self.config.processed_data_dir,
            "s3_uri": self.config.s3_uri,
            "baseline_types": self.config.baseline_types,
            "split_by_model": self.config.split_by_model,
            "results_base_dir": self.config.results_base_dir,
            "transcripts_base_dir": self.config.transcripts_base_dir,
            "logs_base_dir": self.config.logs_base_dir
        }


def create_default_config() -> ConfigManager:
    """Create a default configuration manager with common environment overrides."""
    config_manager = ConfigManager()
    
    # Add common environment variable overrides
    config_manager.add_environment_override("model_name", "LIE_DETECTOR_MODEL")
    config_manager.add_environment_override("num_samples", "LIE_DETECTOR_SAMPLES", int)
    config_manager.add_environment_override("processed_data_dir", "LIE_DETECTOR_DATA_DIR")
    config_manager.add_environment_override("s3_uri", "LIE_DETECTOR_S3_URI")
    
    # Apply any environment overrides
    config_manager.apply_environment_overrides()
    
    return config_manager


def get_baseline_types() -> list:
    """Get the list of all available baseline types."""
    return [
        "baseline",
        "escaped_transcript", 
        "llama_chat",
        "llama_chat_reasoning",
        "base_transcript_reasoning",
        "rowans_escaped_transcript"
    ]


def validate_baseline_type(baseline_type: str) -> bool:
    """Validate that a baseline type is supported."""
    return baseline_type in get_baseline_types()