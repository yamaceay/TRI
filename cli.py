"""
Command-line interface for TRI (Text Re-Identification) application.

This module handles argument parsing, user interaction, and result presentation,
separating CLI concerns from the core business logic.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from config import RuntimeConfig

logger = logging.getLogger(__name__)


def parse_arguments() -> str:
    """Parse command-line arguments and return config file path."""
    args_count = len(sys.argv)
    
    if args_count > 2:
        raise ValueError(f"Expected 1 argument (config file), got {args_count - 1}")
    elif args_count < 2:
        raise ValueError("Configuration file path must be provided as first argument")
    
    config_file_path = sys.argv[1]
    
    # Validate config file
    if not config_file_path.endswith(".json"):
        raise ValueError(f"Configuration file must be JSON format: {config_file_path}")
    
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    
    logger.info("arguments_parsed", extra={"config_file": config_file_path})
    return config_file_path


def load_config_file(config_file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    logger.info("loading_config_file", extra={"path": config_file_path})
    
    try:
        with open(config_file_path, "r") as f:
            config_dict = json.load(f)
        
        logger.info("config_file_loaded", extra={
            "path": config_file_path,
            "keys": list(config_dict.keys())
        })
        
        return config_dict
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_file_path}: {e}")


def print_data_statistics(data_info: Dict[str, Any], config: RuntimeConfig) -> None:
    """Print data processing statistics to user."""
    train_df = data_info["train_df"]
    eval_dfs = data_info["eval_dfs"]
    no_eval_individuals = data_info["no_eval_individuals"]
    no_train_individuals = data_info["no_train_individuals"]
    eval_individuals = data_info["eval_individuals"]
    
    print(f"\n📊 Data Statistics:")
    print(f"   Background knowledge documents for training: {len(train_df)}")
    
    eval_counts = {name: len(df) for name, df in eval_dfs.items()}
    print(f"   Protected documents for evaluation: {eval_counts}")
    
    if no_eval_individuals:
        print(f"   ⚠️  No protected documents found for {len(no_eval_individuals)} individuals")
    
    if no_train_individuals:
        max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
        print(f"   ⚠️  No background knowledge for {len(no_train_individuals)} individuals")
        print(f"      Re-identification risk limited to {max_risk:.1f}% (excluding dev set)")


def print_phase_start(phase_name: str) -> None:
    """Print phase start message."""
    print(f"\n🚀 Starting {phase_name.replace('_', ' ').title()} Phase...")


def print_phase_complete(phase_name: str) -> None:
    """Print phase completion message."""
    print(f"✅ {phase_name.replace('_', ' ').title()} Phase Complete")


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print final TRI results to user."""
    print(f"\n🎯 TRI Results:")
    
    for dataset_name, metrics in results.items():
        accuracy = metrics.get('eval_Accuracy', 0)
        print(f"   {dataset_name}: {accuracy:.1f}% TRIR")
    
    print()


def print_error(error: Exception) -> None:
    """Print error message to user."""
    print(f"\n❌ Error: {error}")
    if isinstance(error, (FileNotFoundError, ValueError)):
        print("   Please check your configuration and try again.")
    else:
        print("   An unexpected error occurred. Check logs for details.")


def print_configuration_summary(config: RuntimeConfig) -> None:
    """Print configuration summary to user."""
    print(f"\n⚙️  Configuration Summary:")
    print(f"   Data file: {config.data_file_path}")
    print(f"   Output folder: {config.output_folder_path}")
    print(f"   Base model: {config.base_model_name}")
    print(f"   Individual column: {config.individual_name_column}")
    print(f"   Background knowledge column: {config.background_knowledge_column}")
    
    # Optional settings
    settings = []
    if config.anonymize_background_knowledge:
        settings.append("background anonymization")
    if config.use_document_curation:
        settings.append("document curation")
    if config.use_additional_pretraining:
        settings.append("additional pretraining")
    
    if settings:
        print(f"   Features: {', '.join(settings)}")


def setup_logging(verbose: bool = True) -> None:
    """Setup logging configuration for CLI."""
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        level=log_level,
        handlers=[
            logging.StreamHandler(sys.stderr),
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    
    if verbose:
        print("🔧 Verbose logging enabled")


def confirm_configuration(config: RuntimeConfig) -> bool:
    """Ask user to confirm configuration before proceeding."""
    print_configuration_summary(config)
    
    # In automated environments, skip confirmation
    if os.getenv("TRI_AUTO_CONFIRM", "false").lower() == "true":
        return True
    
    try:
        response = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
        return response in ("", "y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


def print_welcome() -> None:
    """Print welcome message."""
    print("🔍 TRI (Text Re-Identification) System")
    print("=====================================")


def print_goodbye() -> None:
    """Print goodbye message."""
    print("\n👋 TRI execution complete!")


def handle_keyboard_interrupt() -> None:
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Execution interrupted by user")
    print("   Cleaning up resources...")


def estimate_execution_time(config: RuntimeConfig) -> str:
    """Provide rough execution time estimate based on configuration."""
    base_time = 5  # Base time in minutes
    
    if config.use_additional_pretraining:
        base_time += config.pretraining_epochs * 2
    
    base_time += config.finetuning_epochs * 1.5
    
    if config.use_document_curation:
        base_time += 2
    
    if base_time < 60:
        return f"~{int(base_time)} minutes"
    else:
        hours = int(base_time // 60)
        minutes = int(base_time % 60)
        return f"~{hours}h {minutes}m"


def print_execution_estimate(config: RuntimeConfig) -> None:
    """Print estimated execution time."""
    estimate = estimate_execution_time(config)
    print(f"⏱️  Estimated execution time: {estimate}")
    print("   (Actual time may vary based on hardware and data size)")


def print_model_info(model_info: Dict[str, Any]) -> None:
    """Print model information."""
    print(f"\n🤖 Model Information:")
    
    if "tri_model" in model_info:
        model = model_info["tri_model"]
        if hasattr(model, 'num_parameters'):
            param_count = model.num_parameters()
            if param_count > 1e6:
                print(f"   Parameters: {param_count/1e6:.1f}M")
            else:
                print(f"   Parameters: {param_count:,}")
    
    if "tokenizer" in model_info:
        tokenizer = model_info["tokenizer"]
        if hasattr(tokenizer, 'model_max_length'):
            print(f"   Max sequence length: {tokenizer.model_max_length}")


def print_progress_summary(phase: str, step: str) -> None:
    """Print progress summary for long-running operations."""
    progress_messages = {
        "data_processing": {
            "read_data": "📖 Reading data file...",
            "split_data": "🔄 Splitting into train/eval sets...",
            "preprocess_data": "🔧 Preprocessing text data...",
            "anonymize": "🎭 Anonymizing background knowledge...",
            "curate": "✨ Curating documents...",
        },
        "model_building": {
            "create_base_model": "🏗️  Creating base model...",
            "pretraining": "📚 Additional pretraining...",
            "finetuning": "🎯 Finetuning for re-identification...",
            "save_model": "💾 Saving trained model...",
        },
        "prediction": {
            "evaluate": "🔍 Evaluating model performance...",
            "compute_metrics": "📊 Computing final metrics...",
        }
    }
    
    message = progress_messages.get(phase, {}).get(step, f"Processing {step}...")
    print(f"   {message}")


def print_resource_usage() -> None:
    """Print resource usage information."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n🖥️  GPU: {device_name}")
            print(f"   Memory allocated: {memory_allocated:.1f}GB")
            print(f"   Memory cached: {memory_cached:.1f}GB")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print(f"\n🖥️  MPS available but using CPU for compatibility")
            print(f"   Note: MPS disabled to avoid tensor placement issues")
        else:
            print(f"\n🖥️  Using CPU (CUDA not available)")
    except ImportError:
        pass