"""
Command-line interface for TRI (Text Re-Identification) application.

This module handles argument parsing, configuration management, user interaction, and result presentation,
separating CLI concerns from the core business logic.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, Any
from tri.config import RuntimeConfig

logger = logging.getLogger(__name__)


def parse_arguments() -> str:
    """Parse command-line arguments and return config file path."""
    args_count = len(sys.argv)
    
    if args_count > 2:
        raise ValueError(f"Expected 1 argument (config file), got {args_count - 1}")
    elif args_count < 2:
        raise ValueError("Configuration file path must be provided as first argument")
    
    config_file_path = sys.argv[1]
    
    
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
        with open(config_file_path, "r", encoding="utf-8") as f:
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
    """Print data processing statistics."""
    print("\n" + "="*60)
    print("📊 DATA PROCESSING STATISTICS")
    print("="*60)
    
    train_df = data_info["train_df"]
    eval_dfs = data_info["eval_dfs"]
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Evaluation datasets: {len(eval_dfs)}")
    
    for name, eval_df in eval_dfs.items():
        print(f"    - {name}: {len(eval_df)} samples")
    
    if hasattr(config, 'annotation_folder_path') and config.annotation_folder_path:
        print(f"  Annotation folder: {config.annotation_folder_path}")
        
    
    total_individuals = data_info.get("num_labels", len(set(train_df.iloc[:, 0])))
    print(f"  Total individuals: {total_individuals}")


def print_runtime_anonymization_stats(stats: Dict[str, Any]) -> None:
    """Print runtime anonymization statistics."""
    print("\n" + "="*60)
    print("🔄 RUNTIME ANONYMIZATION STATISTICS")
    print("="*60)
    
    for method, method_stats in stats.items():
        print(f"\n  {method.upper()}:")
        print(f"    Documents processed: {method_stats['total_documents']}")
        print(f"    Documents with annotations: {method_stats['documents_with_annotations']}")
        print(f"    Total spans: {method_stats['total_spans']}")
        print(f"    Average spans/document: {method_stats['average_spans_per_document']:.1f}")
        print(f"    Coverage: {method_stats['coverage']:.1%}")


def print_error(error: Exception) -> None:
    """Print error message."""
    print(f"\n❌ Error: {error}")
    logger.error("cli_error", extra={"error": str(error), "type": type(error).__name__})


def print_phase_start(phase_name: str) -> None:
    """Print phase start message."""
    print(f"\n🚀 Starting {phase_name.replace('_', ' ').title()} Phase...")


def print_phase_complete(phase_name: str) -> None:
    """Print phase completion message."""
    print(f"✅ {phase_name.replace('_', ' ').title()} Phase Complete")


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print evaluation results."""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    for dataset_name, metrics in results.items():
        accuracy = metrics.get('eval_Accuracy', 0)
        print(f"  {dataset_name}: {accuracy:.2f}%")


def print_annotation_results(results: Dict[str, Any]) -> None:
    """Print annotation generation results."""
    print("\n" + "="*60)
    print("📝 ANNOTATION GENERATION RESULTS")
    print("="*60)
    
    stats = results.get("statistics", {})
    print(f"  Method: {results.get('method_used', 'unknown')}")
    print(f"  Individuals: {stats.get('individuals_annotated', 0)}")
    print(f"  Total spans: {stats.get('total_spans', 0)}")
    print(f"  Average spans/individual: {stats.get('average_spans_per_individual', 0):.1f}")
    print(f"  Validation: {'✅ Passed' if results.get('validation_passed', False) else '❌ Failed'}")
    
    if results.get('output_path'):
        print(f"  Output: {results['output_path']}")
    
    
    available_methods = results.get("available_methods", {})
    available_count = sum(1 for info in available_methods.values() if info.get("available", False))
    print(f"  Available methods: {available_count}/{len(available_methods)}")


def print_configuration_summary(config: RuntimeConfig) -> None:
    """Print configuration summary to user."""
    print(f"\n⚙️  Configuration Summary:")
    print(f"   Data file: {config.data_file_path}")
    print(f"   Output folder: {config.output_folder_path}")
    print(f"   Base model: {config.base_model_name}")
    print(f"   Individual column: {config.individual_name_column}")
    print(f"   Background knowledge column: {config.background_knowledge_column}")
    
    
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
    
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        level=log_level,
        handlers=[
            logging.StreamHandler(sys.stderr),
        ]
    )
    
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    
    if verbose:
        print("🔧 Verbose logging enabled")


def confirm_configuration(config: RuntimeConfig) -> bool:
    """Ask user to confirm configuration before proceeding."""
    print_configuration_summary(config)
    
    
    if os.getenv("TRI_AUTO_CONFIRM", "false").lower() == "true":
        return True
    
    try:
        if config.auto_confirm:
            return True
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
    base_time = 5  
    
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