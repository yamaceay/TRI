"""
Context managers for resource lifecycle management.

This module provides context managers for heavy resources like models, tokenizers,
and spaCy NLP components, ensuring proper cleanup and resource management.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch
import en_core_web_lg
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification

from tri.config import RuntimeConfig

logger = logging.getLogger(__name__)


@contextmanager
def spacy_nlp_context(disable_components: Optional[list[str]] = None) -> Generator[Any, None, None]:
    """
    Context manager for spaCy NLP model.
    
    Args:
        disable_components: List of components to disable for memory optimization
        
    Yields:
        spaCy NLP model instance
    """
    logger.info("resource_loading", extra={"resource": "spacy_nlp", "action": "start"})
    
    if disable_components is None:
        nlp = en_core_web_lg.load()
    else:
        nlp = en_core_web_lg.load(disable=disable_components)
        
        if "senter" in disable_components and "parser" in disable_components:
            nlp.add_pipe('sentencizer')
    
    try:
        logger.info("resource_loaded", extra={"resource": "spacy_nlp", "model_size": "large"})
        yield nlp
    finally:
        
        del nlp
        gc.collect()
        logger.info("resource_cleanup", extra={"resource": "spacy_nlp", "action": "complete"})


@contextmanager
def base_model_context(config: RuntimeConfig) -> Generator[tuple[Any, Any], None, None]:
    """
    Context manager for base model and tokenizer.
    
    Args:
        config: Runtime configuration
        
    Yields:
        Tuple of (model, tokenizer)
    """
    logger.info("resource_loading", extra={"resource": "base_model", "model_name": config.base_model_name})
    
    model = AutoModel.from_pretrained(config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    
    
    device = torch.device("cpu")
    model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    
    try:
        logger.info("resource_loaded", extra={
            "resource": "base_model", 
            "model_name": config.base_model_name,
            "model_size": model_size,
            "device": str(device)
        })
        yield model, tokenizer
    finally:
        
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("resource_cleanup", extra={"resource": "base_model", "action": "complete"})


@contextmanager
def mlm_model_context(config: RuntimeConfig, base_model: Any) -> Generator[Any, None, None]:
    """
    Context manager for masked language model.
    
    Args:
        config: Runtime configuration
        base_model: Base model to extend
        
    Yields:
        MLM model instance
    """
    logger.info("resource_loading", extra={"resource": "mlm_model", "action": "start"})
    
    mlm_model = AutoModelForMaskedLM.from_pretrained(config.base_model_name)
    
    
    if "distilbert" in config.base_model_name:
        old_base = mlm_model.distilbert
        mlm_model.distilbert = base_model
    elif "roberta" in config.base_model_name:
        old_base = mlm_model.roberta
        mlm_model.roberta = base_model
    elif "bert" in config.base_model_name:
        old_base = mlm_model.bert
        mlm_model.bert = base_model
    else:
        raise ValueError(f"Unsupported base model: {config.base_model_name}")
    
    
    del old_base
    gc.collect()
    
    
    device = torch.device("cpu")
    mlm_model.to(device)
    
    try:
        logger.info("resource_loaded", extra={"resource": "mlm_model", "device": str(device)})
        yield mlm_model
    finally:
        del mlm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("resource_cleanup", extra={"resource": "mlm_model", "action": "complete"})


@contextmanager
def classification_model_context(config: RuntimeConfig, base_model: Any, num_labels: int) -> Generator[Any, None, None]:
    """
    Context manager for sequence classification model.
    
    Args:
        config: Runtime configuration
        base_model: Base model to extend
        num_labels: Number of classification labels
        
    Yields:
        Classification model instance
    """
    logger.info("resource_loading", extra={
        "resource": "classification_model", 
        "num_labels": num_labels,
        "action": "start"
    })
    
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name, 
        num_labels=num_labels
    )
    
    
    if "distilbert" in config.base_model_name:
        clf_model.distilbert.load_state_dict(base_model.state_dict())
    elif "roberta" in config.base_model_name:
        base_state = base_model.state_dict().copy()
        
        base_state.pop("pooler.dense.weight", None)
        base_state.pop("pooler.dense.bias", None)
        clf_model.roberta.load_state_dict(base_state)
    elif "bert" in config.base_model_name:
        clf_model.bert.load_state_dict(base_model.state_dict())
    else:
        raise ValueError(f"Unsupported base model: {config.base_model_name}")
    
    
    device = torch.device("cpu")
    clf_model.to(device)
    
    try:
        logger.info("resource_loaded", extra={
            "resource": "classification_model", 
            "num_labels": num_labels,
            "device": str(device)
        })
        yield clf_model
    finally:
        del clf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("resource_cleanup", extra={"resource": "classification_model", "action": "complete"})


@contextmanager
def pretrained_model_context(config: RuntimeConfig) -> Generator[tuple[Any, Any], None, None]:
    """
    Context manager for loading pre-trained TRI model.
    
    Args:
        config: Runtime configuration
        
    Yields:
        Tuple of (model, tokenizer)
    """
    logger.info("resource_loading", extra={"resource": "pretrained_tri_model", "action": "start"})
    
    model = AutoModelForSequenceClassification.from_pretrained(config.tri_pipe_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tri_pipe_path)
    
    
    device = torch.device("cpu")
    model.to(device)
    
    try:
        logger.info("resource_loaded", extra={
            "resource": "pretrained_tri_model",
            "path": config.tri_pipe_path,
            "device": str(device)
        })
        yield model, tokenizer
    finally:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("resource_cleanup", extra={"resource": "pretrained_tri_model", "action": "complete"})


def _get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        
        return torch.device("cpu")
    return torch.device("cpu")


@contextmanager
def memory_cleanup_context() -> Generator[None, None, None]:
    """Context manager for aggressive memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("memory_cleanup", extra={"action": "complete"})