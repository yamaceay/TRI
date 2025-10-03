"""
Configuration management with immutable dataclasses.

This module provides the RuntimeConfig dataclass that centralizes all configuration
parameters and provides canonicalization functions for setting defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable configuration for TRI workflow."""
    
    # Mandatory configurations
    output_folder_path: str
    data_file_path: str
    individual_name_column: str
    background_knowledge_column: str
    
    # Optional configurations with defaults
    load_saved_pretreatment: bool = True
    anonymize_background_knowledge: bool = True
    only_use_anonymized_background_knowledge: bool = True
    use_document_curation: bool = True
    save_pretreatment: bool = True
    base_model_name: str = "distilbert-base-uncased"
    tokenization_block_size: int = 250
    use_additional_pretraining: bool = True
    save_additional_pretraining: bool = True
    load_saved_pretraining: bool = True
    pretraining_epochs: int = 3
    pretraining_batch_size: int = 8
    pretraining_learning_rate: float = 5e-05
    pretraining_mlm_probability: float = 0.15
    pretraining_sliding_window: str = "512-128"
    save_finetuning: bool = True
    load_saved_finetuning: bool = True
    finetuning_epochs: int = 15
    finetuning_batch_size: int = 16
    finetuning_learning_rate: float = 5e-05
    finetuning_sliding_window: str = "100-25"
    dev_set_column_name: Optional[str] = None
    
    # Annotation handling
    anonymized_columns: list[str] = field(default_factory=list)
    annotation_folder_path: Optional[str] = None  # Folder containing existing annotations
    annotation_mask_token: str = "[MASK]"
    
    # Derived paths - computed automatically
    pretreated_data_path: str = field(init=False)
    pretrained_model_path: str = field(init=False)
    results_file_path: str = field(init=False)
    tri_pipe_path: str = field(init=False)
    annotations_output_path: str = field(init=False)
    
    # Training configurations - computed automatically
    pretraining_config: TrainingConfig = field(init=False)
    finetuning_config: TrainingConfig = field(init=False)
    
    auto_confirm: bool = True  # If true, skip confirmation prompts in CLI

    def __post_init__(self) -> None:
        """Initialize derived configurations after creation."""
        # Derived paths
        object.__setattr__(self, 'pretreated_data_path', 
                          os.path.join(self.output_folder_path, "Pretreated_Data.json"))
        object.__setattr__(self, 'pretrained_model_path', 
                          os.path.join(self.output_folder_path, "Pretrained_Model.pt"))
        object.__setattr__(self, 'results_file_path', 
                          os.path.join(self.output_folder_path, "Results.csv"))
        object.__setattr__(self, 'tri_pipe_path', 
                          os.path.join(self.output_folder_path, "TRI_Pipeline"))
        # Set annotation output path
        if self.annotation_folder_path:
            annotations_path = self.annotation_folder_path
        else:
            annotations_path = os.path.join(self.output_folder_path, "annotations")
        object.__setattr__(self, 'annotations_output_path', annotations_path)
        
        # Training configurations
        pretraining_config = TrainingConfig(
            is_for_mlm=True,
            uses_labels=False,
            epochs=self.pretraining_epochs,
            batch_size=self.pretraining_batch_size,
            learning_rate=self.pretraining_learning_rate,
            sliding_window=self.pretraining_sliding_window,
            trainer_folder_path=os.path.join(self.output_folder_path, "Pretraining")
        )
        object.__setattr__(self, 'pretraining_config', pretraining_config)
        
        finetuning_config = TrainingConfig(
            is_for_mlm=False,
            uses_labels=True,
            epochs=self.finetuning_epochs,
            batch_size=self.finetuning_batch_size,
            learning_rate=self.finetuning_learning_rate,
            sliding_window=self.finetuning_sliding_window,
            trainer_folder_path=os.path.join(self.output_folder_path, "Finetuning")
        )
        object.__setattr__(self, 'finetuning_config', finetuning_config)


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""
    is_for_mlm: bool
    uses_labels: bool
    epochs: int
    batch_size: int
    learning_rate: float
    sliding_window: str
    trainer_folder_path: str


def canonicalize_config_from_dict(config_dict: Dict[str, Any]) -> RuntimeConfig:
    """Create RuntimeConfig from dictionary with validation and defaults."""
    # Validate mandatory fields
    mandatory_fields = [
        "output_folder_path", "data_file_path", 
        "individual_name_column", "background_knowledge_column"
    ]
    
    for field_name in mandatory_fields:
        if field_name not in config_dict:
            raise ValueError(f"Mandatory configuration field '{field_name}' is missing")
        if not isinstance(config_dict[field_name], str):
            raise ValueError(f"Configuration field '{field_name}' must be a string")
    
    # Validate file paths
    data_file_path = config_dict["data_file_path"]
    if not os.path.isfile(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    if not data_file_path.endswith(('.json', '.csv')):
        raise ValueError(f"Data file must be JSON or CSV format: {data_file_path}")
    
    # Handle dev_set_column_name (convert False to None for proper typing)
    if "dev_set_column_name" in config_dict and config_dict["dev_set_column_name"] is False:
        config_dict = config_dict.copy()
        config_dict["dev_set_column_name"] = None
    
    # Filter out unknown fields and create config
    valid_fields = {f.name for f in fields(RuntimeConfig) if f.init}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    return RuntimeConfig(**filtered_dict)


def canonicalize_config_from_namespace(namespace: Namespace) -> RuntimeConfig:
    """Create RuntimeConfig from argparse Namespace."""
    return canonicalize_config_from_dict(vars(namespace))


def ensure_output_directory(config: RuntimeConfig) -> None:
    """Create output directory if it doesn't exist."""
    Path(config.output_folder_path).mkdir(parents=True, exist_ok=True)


def validate_data_columns(config: RuntimeConfig, df_columns: list[str]) -> None:
    """Validate that required columns exist in the dataframe."""
    if config.individual_name_column not in df_columns:
        raise ValueError(f"Individual name column '{config.individual_name_column}' not found in data")
    
    if config.background_knowledge_column not in df_columns:
        raise ValueError(f"Background knowledge column '{config.background_knowledge_column}' not found in data")
    
    if config.dev_set_column_name and config.dev_set_column_name not in df_columns:
        raise ValueError(f"Dev set column '{config.dev_set_column_name}' not found in data")
    
    # Check for anonymization columns
    anon_cols = [col for col in df_columns 
                if col not in [config.individual_name_column, config.background_knowledge_column]]
    if not anon_cols:
        raise ValueError("No anonymization columns found in data")


def get_device_config() -> str:
    """Determine the best device to use for computation."""
    import torch
    if torch.cuda.is_available():
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        return "cuda:0"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Use CPU instead of MPS to avoid tensor placement issues
        return "cpu"
    return "cpu"