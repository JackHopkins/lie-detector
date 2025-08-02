"""
Dataset factory to eliminate duplication in dataset creation across different formats.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from inspect_ai.dataset import Dataset, MemoryDataset


class DatasetFactory:
    """Factory for creating datasets with different formats and configurations."""
    
    def __init__(self):
        # Import dataset functions dynamically to avoid circular imports
        self._dataset_functions = None
        self._dataset_by_model_functions = None
    
    def _load_dataset_functions(self):
        """Lazy load dataset functions to avoid circular imports."""
        if self._dataset_functions is None:
            try:
                from ..dataset import (
                    baseline_dataset, conversation_dataset, llama_chat_dataset,
                    llama_chat_reasoning_dataset, base_transcript_reasoning_dataset,
                    rowans_escaped_transcript_dataset
                )
                from ..dataset import (
                    baseline_dataset_by_model, conversation_dataset_by_model,
                    llama_chat_dataset_by_model, llama_chat_reasoning_dataset_by_model,
                    base_transcript_reasoning_dataset_by_model,
                    rowans_escaped_transcript_dataset_by_model
                )
                
                self._dataset_functions = {
                    "baseline_dataset": baseline_dataset,
                    "conversation_dataset": conversation_dataset,
                    "llama_chat_dataset": llama_chat_dataset,
                    "llama_chat_reasoning_dataset": llama_chat_reasoning_dataset,
                    "base_transcript_reasoning_dataset": base_transcript_reasoning_dataset,
                    "rowans_escaped_transcript_dataset": rowans_escaped_transcript_dataset
                }
                
                self._dataset_by_model_functions = {
                    "baseline_dataset": baseline_dataset_by_model,
                    "conversation_dataset": conversation_dataset_by_model,
                    "llama_chat_dataset": llama_chat_dataset_by_model,
                    "llama_chat_reasoning_dataset": llama_chat_reasoning_dataset_by_model,
                    "base_transcript_reasoning_dataset": base_transcript_reasoning_dataset_by_model,
                    "rowans_escaped_transcript_dataset": rowans_escaped_transcript_dataset_by_model
                }
                
            except ImportError as e:
                raise ImportError(f"Failed to import dataset functions: {e}")
    
    def create_dataset(
        self, 
        dataset_func_name: str, 
        data_dir: str, 
        limit: Optional[int] = None
    ) -> Dataset:
        """
        Create a dataset using the specified function.
        
        Args:
            dataset_func_name: Name of the dataset function to use
            data_dir: Directory containing the data
            limit: Optional limit on number of samples
            
        Returns:
            Dataset object
            
        Raises:
            ValueError: If dataset function is not found
        """
        self._load_dataset_functions()
        
        if dataset_func_name not in self._dataset_functions:
            raise ValueError(f"Unknown dataset function: {dataset_func_name}")
        
        dataset_func = self._dataset_functions[dataset_func_name]
        return dataset_func(data_dir, limit)
    
    def create_datasets_by_model(
        self, 
        dataset_func_name: str, 
        data_dir: str, 
        limit: Optional[int] = None
    ) -> Dict[str, Dataset]:
        """
        Create separate datasets for each model using the specified function.
        
        Args:
            dataset_func_name: Name of the dataset function to use
            data_dir: Directory containing the data
            limit: Optional limit on number of samples per model
            
        Returns:
            Dictionary mapping model names to Dataset objects
            
        Raises:
            ValueError: If dataset function is not found
        """
        self._load_dataset_functions()
        
        if dataset_func_name not in self._dataset_by_model_functions:
            raise ValueError(f"Unknown by-model dataset function: {dataset_func_name}")
        
        dataset_func = self._dataset_by_model_functions[dataset_func_name]
        return dataset_func(data_dir, limit)
    
    def get_supported_functions(self) -> List[str]:
        """Get list of supported dataset function names."""
        self._load_dataset_functions()
        return list(self._dataset_functions.keys())


class DatasetBuilder:
    """Builder pattern for creating custom datasets with specific configurations."""
    
    def __init__(self):
        self.data_dir = None
        self.limit = None
        self.format_type = None
        self.by_model = False
    
    def with_data_dir(self, data_dir: str) -> 'DatasetBuilder':
        """Set the data directory."""
        self.data_dir = data_dir
        return self
    
    def with_limit(self, limit: int) -> 'DatasetBuilder':
        """Set the sample limit."""
        self.limit = limit
        return self
    
    def with_format(self, format_type: str) -> 'DatasetBuilder':
        """Set the dataset format type."""
        self.format_type = format_type
        return self
    
    def split_by_model(self, by_model: bool = True) -> 'DatasetBuilder':
        """Enable/disable splitting by model."""
        self.by_model = by_model
        return self
    
    def build(self) -> Dataset | Dict[str, Dataset]:
        """
        Build the dataset with the specified configuration.
        
        Returns:
            Dataset or Dict[str, Dataset] if split by model
            
        Raises:
            ValueError: If required configuration is missing
        """
        if not self.data_dir:
            raise ValueError("Data directory must be specified")
        if not self.format_type:
            raise ValueError("Format type must be specified")
        
        factory = DatasetFactory()
        
        if self.by_model:
            return factory.create_datasets_by_model(
                f"{self.format_type}_dataset", self.data_dir, self.limit
            )
        else:
            return factory.create_dataset(
                f"{self.format_type}_dataset", self.data_dir, self.limit
            )