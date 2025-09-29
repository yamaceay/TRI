"""
TRI (Text Re-Identification) - Legacy compatibility module.

This module maintains backward compatibility with the original tri.py interface
while using the new Go-ish modular architecture internally.

Original functionality preserved:
- TRI class for direct instantiation
- Original configuration interface
- Same API methods (run_data, run_build_classifier, run_predict_trir)
- Command-line interface compatibility

New architecture benefits:
- Immutable configuration with validation
- Proper resource management via context managers
- Dependency injection and interface-based design
- Structured logging with extra fields
- Clean separation of concerns
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple

# New modular imports
from config import RuntimeConfig, canonicalize_config_from_dict
from main import run_tri_from_config, create_tri_orchestrator
from cli import parse_arguments, load_config_file, setup_logging

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import Dataset
    from transformers import Trainer

# Legacy imports for backward compatibility
from core import TRIDataset, TRITrainer, compute_metrics

logger = logging.getLogger(__name__)


class TRI:
    """
    Legacy TRI class for backward compatibility.
    
    Internally uses the new modular architecture while preserving
    the original interface and behavior.
    """
    
    def __init__(self, **kwargs):
        """Initialize TRI with legacy configuration interface."""
        # Convert legacy kwargs to RuntimeConfig
        self._validate_legacy_kwargs(kwargs)
        self.config = canonicalize_config_from_dict(kwargs)
        
        # Create orchestrator
        self.orchestrator = create_tri_orchestrator()
        
        # Legacy properties for backward compatibility
        self._data_info: Optional[Dict[str, Any]] = None
        self._model_info: Optional[Dict[str, Any]] = None
        self._results: Optional[Dict[str, Any]] = None
        
        logger.info("legacy_tri_initialized", extra={"output_path": self.config.output_folder_path})
    
    def _validate_legacy_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate legacy keyword arguments."""
        mandatory_fields = [
            "output_folder_path", "data_file_path",
            "individual_name_column", "background_knowledge_column"
        ]
        
        for field in mandatory_fields:
            if field not in kwargs:
                raise AttributeError(f"Mandatory argument {field} is not defined")
    
    def set_configs(self, are_mandatory_configs_required: bool = False, **kwargs) -> None:
        """Set configurations (legacy method)."""
        # Merge with existing config
        current_dict = {
            field.name: getattr(self.config, field.name)
            for field in self.config.__dataclass_fields__.values()
            if field.init
        }
        current_dict.update(kwargs)
        
        # Recreate config
        if are_mandatory_configs_required:
            self._validate_legacy_kwargs(current_dict)
        
        self.config = canonicalize_config_from_dict(current_dict)
        logger.info("legacy_config_updated", extra={"new_keys": list(kwargs.keys())})
    
    def run(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """Run complete TRI workflow (legacy method)."""
        logger.info("legacy_run_start")
        
        try:
            # Run all phases
            self.run_data(verbose=verbose)
            self.run_build_classifier(verbose=verbose)
            results = self.run_predict_trir(verbose=verbose)
            
            logger.info("legacy_run_complete")
            return results
        
        except Exception as e:
            logger.error("legacy_run_error", extra={"error": str(e)})
            raise
    
    def run_data(self, verbose: bool = True) -> None:
        """Run data processing phase (legacy method)."""
        if verbose:
            logging.info("######### START: DATA #########")
        
        try:
            self._data_info = self.orchestrator.run_data_processing(self.config)
            
            if verbose:
                self._print_legacy_data_stats()
                logging.info("######### END: DATA #########")
        
        except Exception as e:
            logger.error("legacy_data_error", extra={"error": str(e)})
            raise
    
    def run_build_classifier(self, verbose: bool = True) -> None:
        """Run model building phase (legacy method)."""
        if verbose:
            logging.info("######### START: BUILD CLASSIFIER #########")
        
        if self._data_info is None:
            raise RuntimeError("Must run run_data() before run_build_classifier()")
        
        try:
            self._model_info = self.orchestrator.run_model_building(self._data_info, self.config)
            
            if verbose:
                logging.info("######### END: BUILD CLASSIFIER #########")
        
        except Exception as e:
            logger.error("legacy_build_classifier_error", extra={"error": str(e)})
            raise
    
    def run_predict_trir(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """Run prediction phase (legacy method)."""
        if verbose:
            logging.info("######### START: PREDICT TRIR #########")
        
        if self._model_info is None:
            raise RuntimeError("Must run run_build_classifier() before run_predict_trir()")
        
        try:
            self._results = self.orchestrator.run_prediction(self._model_info, self.config)
            
            if verbose:
                self._print_legacy_results()
                logging.info("######### END: PREDICT TRIR #########")
            
            return self._results
        
        except Exception as e:
            logger.error("legacy_predict_error", extra={"error": str(e)})
            raise
    
    def _print_legacy_data_stats(self) -> None:
        """Print data statistics in legacy format."""
        if not self._data_info:
            return
        
        train_df = self._data_info["train_df"]
        eval_dfs = self._data_info["eval_dfs"]
        no_eval_individuals = self._data_info["no_eval_individuals"]
        no_train_individuals = self._data_info["no_train_individuals"]
        eval_individuals = self._data_info["eval_individuals"]
        
        logging.info(f"Number of background knowledge documents for training: {len(train_df)}")
        
        eval_n_dict = {name: len(df) for name, df in eval_dfs.items()}
        logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")
        
        if len(no_eval_individuals) > 0:
            logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
        
        if len(no_train_individuals) > 0:
            max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
            logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. "
                        f"Re-identification risk limited to {max_risk:.3f}% (excluding dev set).")
    
    def _print_legacy_results(self) -> None:
        """Print results in legacy format."""
        if not self._results:
            return
        
        for dataset_name, metrics in self._results.items():
            accuracy = metrics.get('eval_Accuracy', 0)
            logging.info(f"TRIR {dataset_name} = {accuracy}%")
    
    # Legacy property accessors for backward compatibility
    @property
    def data_df(self) -> Optional[pd.DataFrame]:
        """Legacy property access."""
        return None  # Not stored in new architecture
    
    @property
    def train_df(self) -> Optional[pd.DataFrame]:
        """Legacy property access."""
        return self._data_info["train_df"] if self._data_info else None
    
    @property
    def eval_dfs(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Legacy property access."""
        return self._data_info["eval_dfs"] if self._data_info else None
    
    @property
    def all_individuals(self) -> Optional[set]:
        """Legacy property access."""
        return self._data_info["all_individuals"] if self._data_info else None
    
    @property
    def label_to_name(self) -> Optional[Dict[int, str]]:
        """Legacy property access."""
        return self._data_info["label_to_name"] if self._data_info else None
    
    @property
    def name_to_label(self) -> Optional[Dict[str, int]]:
        """Legacy property access."""
        return self._data_info["name_to_label"] if self._data_info else None
    
    @property
    def num_labels(self) -> Optional[int]:
        """Legacy property access."""
        return self._data_info["num_labels"] if self._data_info else None
    
    @property
    def tri_model(self) -> Optional[Any]:
        """Legacy property access."""
        return self._model_info["tri_model"] if self._model_info else None
    
    @property
    def tokenizer(self) -> Optional[Any]:
        """Legacy property access."""
        return self._model_info["tokenizer"] if self._model_info else None
    
    @property
    def finetuning_trainer(self) -> Optional[Trainer]:
        """Legacy property access."""
        return self._model_info["trainer"] if self._model_info else None
    
    @property
    def trir_results(self) -> Optional[Dict[str, Any]]:
        """Legacy property access."""
        return self._results


# Legacy argument parsing functions for backward compatibility
def argument_parsing() -> str:
    """Legacy argument parsing function."""
    return parse_arguments()


def get_config_from_file(target_dir: str) -> Dict[str, Any]:
    """Legacy config loading function."""
    return load_config_file(target_dir)


# Legacy main execution block
if __name__ == "__main__":
    # Setup logging in legacy format
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', 
        level=logging.INFO
    )
    
    try:
        # Load configuration using legacy interface
        logging.info("######### START: CONFIGURATION #########")
        target_dir = argument_parsing()
        config_dict = get_config_from_file(target_dir)
        
        # Create TRI instance using legacy interface
        tri = TRI(**config_dict)
        logging.info("######### END: CONFIGURATION #########")
        
        # Run all sections using legacy interface
        tri.run(verbose=True)
    
    except Exception as e:
        logging.exception(f"TRI execution failed: {e}")
        sys.exit(1)