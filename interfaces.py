"""
Abstract base classes for TRI (Text Re-Identification) components.

This module defines the interfaces that all concrete implementations must follow,
enabling dependency injection and polymorphism throughout the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, Pipeline
from tri.config import RuntimeConfig


class DataProcessor(ABC):
    """Interface for data processing operations."""
    
    @abstractmethod
    def load_data(self, config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load required annotations from specified folder."""
    
    @abstractmethod
    def preprocess_data(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Apply preprocessing like anonymization and curation."""
        ...
    
    @abstractmethod
    def get_individuals_info(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], config: RuntimeConfig) -> Dict[str, Any]:
        """Extract individual statistics and label mappings."""
        ...


class DatasetBuilder(ABC):
    """Interface for building PyTorch datasets."""
    
    @abstractmethod
    def create_dataset(self, df: pd.DataFrame, tokenizer: Any, name_to_label: Dict[str, int], 
                      uses_labels: bool, sliding_window: str, block_size: int) -> Dataset:
        """Create a PyTorch Dataset from DataFrame."""
        ...


class ModelManager(ABC):
    """Interface for model management operations."""
    
    @abstractmethod
    def create_base_model(self, config: RuntimeConfig) -> Tuple[Any, Any]:
        """Create base model and tokenizer."""
        ...
    
    @abstractmethod
    def perform_pretraining(self, base_model: Any, tokenizer: Any, dataset: Dataset, config: RuntimeConfig) -> Any:
        """Perform additional pretraining on base model."""
        ...
    
    @abstractmethod
    def perform_finetuning(self, base_model: Any, tokenizer: Any, train_dataset: Dataset, 
                          eval_datasets: Dict[str, Dataset], config: RuntimeConfig) -> Tuple[Any, Trainer]:
        """Perform finetuning for text re-identification."""
        ...
    
    @abstractmethod
    def save_model(self, model: Any, tokenizer: Any, config: RuntimeConfig) -> Pipeline:
        """Save trained model and return pipeline."""
        ...
    
    @abstractmethod
    def load_model(self, config: RuntimeConfig) -> Tuple[Any, Any]:
        """Load pre-trained model and tokenizer."""
        ...


class Predictor(ABC):
    """Interface for making predictions."""
    
    @abstractmethod
    def predict(self, trainer: Trainer, eval_datasets: Dict[str, Dataset]) -> Dict[str, Dict[str, float]]:
        """Make predictions and return results."""
        ...


class ConfigManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def validate_config(self, config: RuntimeConfig) -> None:
        """Validate configuration parameters."""
        ...
    
    @abstractmethod
    def ensure_output_directory(self, config: RuntimeConfig) -> None:
        """Create output directory if it doesn't exist."""
        ...


class StorageManager(ABC):
    """Interface for data persistence operations."""
    
    @abstractmethod
    def save_pretreatment(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], config: RuntimeConfig) -> None:
        """Save pretreated data to disk."""
        ...
    
    @abstractmethod
    def load_pretreatment(self, config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load pretreated data from disk."""
        ...
    
    @abstractmethod
    def pretreatment_exists(self, config: RuntimeConfig) -> bool:
        """Check if pretreated data exists."""
        ...
    
    @abstractmethod
    def save_pretrained_model(self, model: Any, config: RuntimeConfig) -> None:
        """Save pretrained model to disk."""
        ...
    
    @abstractmethod
    def load_pretrained_model(self, model: Any, config: RuntimeConfig) -> Any:
        """Load pretrained model from disk."""
        ...
    
    @abstractmethod
    def pretrained_model_exists(self, config: RuntimeConfig) -> bool:
        """Check if pretrained model exists."""
        ...


class TextProcessor(ABC):
    """Interface for text processing operations."""
    
    @abstractmethod
    def anonymize_text(self, text: str) -> str:
        """Anonymize text using NER."""
        ...
    
    @abstractmethod
    def curate_text(self, text: str) -> str:
        """Curate text (lemmatization, stopword removal, etc.)."""
        ...


class AnnotationProcessor(ABC):
    """Interface for generating annotations from text data."""
    
    @abstractmethod
    def generate_annotations(self, data_df: pd.DataFrame, config: RuntimeConfig) -> Dict[str, Any]:
        """Generate annotations for text data using specified method."""
        ...
    
    @abstractmethod
    def get_available_methods(self) -> Dict[str, Any]:
        """Get information about available annotation methods."""
        ...
    
    @abstractmethod
    def save_annotations(self, annotations: Dict[str, Any], output_path: str, config: RuntimeConfig) -> None:
        """Save annotations to file in PETRE-compatible format."""
        ...
    
    @abstractmethod
    def validate_annotations(self, annotations: Dict[str, Any], data_df: pd.DataFrame, config: RuntimeConfig) -> bool:
        """Validate annotations against source data."""
        ...


class WorkflowOrchestrator(ABC):
    """Interface for orchestrating the entire TRI workflow."""
    
    @abstractmethod
    def run_data_processing(self, config: RuntimeConfig) -> Dict[str, Any]:
        """Execute data processing phase."""
        ...
    
    @abstractmethod
    def run_model_building(self, data_info: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
        """Execute model building phase."""
        ...
    
    @abstractmethod
    def run_prediction(self, model_info: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Dict[str, float]]:
        """Execute prediction phase."""
        ...