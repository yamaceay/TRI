"""
Concrete implementations of TRI interfaces.

This module contains all the concrete implementations that follow the defined interfaces,
using dependency injection and immutable configuration patterns.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from tqdm.autonotebook import tqdm
from transformers import (
    AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    get_constant_schedule, pipeline, Pipeline
)
from accelerate import Accelerator

if TYPE_CHECKING:
    from config import RuntimeConfig, TrainingConfig

from interfaces import (
    DataProcessor, DatasetBuilder, ModelManager, Predictor, 
    ConfigManager, StorageManager, TextProcessor, WorkflowOrchestrator,
    AnnotationProcessor
)
from contexts import (
    spacy_nlp_context, base_model_context, mlm_model_context,
    classification_model_context, pretrained_model_context, memory_cleanup_context
)

logger = logging.getLogger(__name__)


class TRIDataProcessor(DataProcessor):
    """Concrete implementation of data processing operations."""
    
    def __init__(self):
        """Initialize data processor."""
    
    def preprocess_data(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], 
                       config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Apply preprocessing including anonymization and curation."""
        logger.info("operation_start", extra={"operation": "preprocess_data"})
        
        processed_train_df = train_df.copy()
        processed_eval_dfs = eval_dfs.copy()
        
        if config.anonymize_background_knowledge:
            logger.info("preprocessing_step", extra={"step": "anonymize_background_knowledge"})
            processed_train_df = self._anonymize_background_knowledge(processed_train_df, config)
        
        if config.use_document_curation:
            logger.info("preprocessing_step", extra={"step": "document_curation"})
            with spacy_nlp_context() as nlp:
                self._curate_dataframe(processed_train_df, nlp)
                for eval_df in processed_eval_dfs.values():
                    self._curate_dataframe(eval_df, nlp)
        
        logger.info("operation_complete", extra={"operation": "preprocess_data"})
        return processed_train_df, processed_eval_dfs
    
    def load_data(self, config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Process dataset with annotations - USER CONTROLS EVERYTHING."""
        import json
        import os
        
        # Load the main dataset
        if config.data_file_path.endswith('.json'):
            with open(config.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_df = pd.DataFrame(data)
        else:
            data_df = pd.read_csv(config.data_file_path)
        
        annotations = {}
        
        # 1. Load from dataset columns if user specified annotation columns
        if hasattr(config, 'anonymized_columns') and config.anonymized_columns:
            logger.info("loading_annotations_from_dataset_columns", extra={"columns": config.anonymized_columns})
            dataset_annotations = self._extract_annotations_from_columns(data_df, config.anonymized_columns, config.individual_name_column)
            print(f"Extracted annotations from columns: {config.anonymized_columns}")  # Debug print
            print(f"Extracted annotations: {dataset_annotations}")  # Debug print
            annotations.update(dataset_annotations)
        
        # 2. Load from external files if user specified annotation folder
        if hasattr(config, 'annotation_folder_path') and config.annotation_folder_path:
            logger.info("loading_annotations_from_external_files", extra={"folder": config.annotation_folder_path})
            external_annotations = self._load_required_annotations(data_df, config)
            annotations.update(external_annotations)
        
        # 3. Fail if no annotations found
        if not annotations:
            raise ValueError(
                "No annotations found! You must specify either:\n"
                "  - annotation_columns: list of dataset columns containing annotations\n"
                "Check your configuration."
            )
        
        # Apply anonymization to create training and evaluation datasets
        train_df, eval_dfs = self._apply_anonymization_from_annotations(data_df, annotations, config)
        
        logger.info("dataset_processing_complete", extra={
            "train_samples": len(train_df),
            "eval_datasets": {name: len(df) for name, df in eval_dfs.items()},
            "total_annotations": sum(len(spans) for spans in annotations.values())
        })
        
        return train_df, eval_dfs
    
    def _extract_annotations_from_columns(self, data_df: pd.DataFrame, annotation_columns: List[str], individual_name_column: str) -> Dict[str, List]:
        """Extract annotations from user-specified dataset columns."""
        import json
        
        annotations = {}
        
        # Validate that specified columns exist
        missing_columns = [col for col in annotation_columns if col not in data_df.columns]
        if missing_columns:
            raise ValueError(f"Specified annotation columns not found in dataset: {missing_columns}. Available columns: {list(data_df.columns)}")
        
        logger.info("extracting_annotations_from_specified_columns", extra={"columns": annotation_columns})
        
        for _, row in data_df.iterrows():
            doc_id = str(row[individual_name_column])
            doc_annotations = {}

            for col in annotation_columns:
                value = row[col]
                if pd.isna(value):
                    continue
                doc_annotations[col] = value

            # Store the grouped annotations for this document
            annotations[doc_id] = doc_annotations
        
        return annotations
    
    def _load_required_annotations(self, data_df: pd.DataFrame, config: RuntimeConfig) -> Dict[str, List]:
        """Load annotations from annotation files, failing if any are missing."""
        import os
        import json
        
        # Determine the correct annotation folder based on data file name
        data_filename = os.path.basename(config.data_file_path)
        if 'train' in data_filename.lower():
            annotation_folder = config.annotation_folder_path.replace('annotations', 'annotations_train')
        elif 'dev' in data_filename.lower():
            annotation_folder = config.annotation_folder_path.replace('annotations', 'annotations_dev')
        elif 'test' in data_filename.lower():
            annotation_folder = config.annotation_folder_path.replace('annotations', 'annotations_test')
        else:
            annotation_folder = config.annotation_folder_path
        
        logger.info("loading_annotations_from_folder", extra={"folder": annotation_folder})
        
        # Check if annotation folder exists
        if not os.path.exists(annotation_folder):
            raise FileNotFoundError(
                f"Annotation folder not found: {annotation_folder}\\n"
                f"Please run 'python3 tri/annotate.py {config.data_file_path}' to generate annotations first."
            )
        
        annotations = {}
        missing_methods = []
        
        # Try to load annotations for each specified method
        for file in os.listdir(annotation_folder):
            annotation_file = os.path.join(annotation_folder, file)
            method = file.rsplit('.', 1)[0]  # Filename without extension

            if not os.path.exists(annotation_file):
                missing_methods.append(method)
                continue
            
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    method_annotations = json.load(f)
                annotations.update({method: method_annotations})
                logger.info("loaded_method_annotations", extra={
                    "method": method,
                    "file": annotation_file,
                    "documents": len(method_annotations)
                })
            except Exception as e:
                logger.error("failed_to_load_method_annotations", extra={
                    "method": method,
                    "file": annotation_file,
                    "error": str(e)
                })
                missing_methods.append(method)
        
        # Fail if any required annotations are missing
        if missing_methods:
            raise FileNotFoundError(
                f"Missing annotation files for methods: {missing_methods}\\n"
                f"Expected files in {annotation_folder}:\\n" + 
                "\\n".join(f"  - {method}.json" for method in missing_methods) +
                f"\\n\\nPlease run 'python3 tri/annotate.py' with your config file to generate annotations first."
            )

        return annotations
    
    def _apply_anonymization_from_annotations(self, data_df: pd.DataFrame, all_annotations: Dict[str, Dict], config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Apply anonymization to create training and evaluation datasets."""
        
        # Create training data with original text
        train_data = []
        
        # Group annotations by annotation method/tool to create separate eval datasets
        all_methods = list(all_annotations.keys())
        method_eval_data = {method: [] for method in all_methods}
        
        for _, row in data_df.iterrows():
            doc_id = str(row[config.individual_name_column])
            original_text = str(row[config.background_knowledge_column])
            
            # Add original text to training data
            train_data.append({
                config.individual_name_column: doc_id,
                config.background_knowledge_column: original_text  # ORIGINAL TEXT
            })

            for method, annotations in all_annotations.items():
                # Get all annotations for this document
                doc_annotations = annotations.get(doc_id, [])
                
                if doc_annotations:
                    if isinstance(doc_annotations, list):
                        temp_text = original_text
                        for span in reversed(doc_annotations):
                            start, end = span
                            if start is not None and end is not None and 0 <= start < end <= len(temp_text):
                                temp_text = temp_text[:start] + config.annotation_mask_token + temp_text[end:]

                method_eval_data[method].append({
                    config.individual_name_column: doc_id,
                    method: temp_text
                })
        
        # Create DataFrames
        train_df = pd.DataFrame(train_data)
        
        # Create evaluation datasets for each annotation method
        eval_dfs = {}
        for method, eval_data in method_eval_data.items():
            eval_dfs[method] = pd.DataFrame(eval_data)  # Use method name directly (e.g., "spacy", "presidio", "manual")
        
        # Handle dev set if specified
        if hasattr(config, 'dev_set_column_name') and config.dev_set_column_name and config.dev_set_column_name in data_df.columns:
            dev_data = []
            for _, row in data_df.iterrows():
                if pd.notna(row[config.dev_set_column_name]):
                    dev_data.append({
                        config.individual_name_column: row[config.individual_name_column],
                        'text': str(row[config.dev_set_column_name])
                    })
            
            if dev_data:
                eval_dfs['dev'] = pd.DataFrame(dev_data)
        
        return train_df, eval_dfs
    
    def _anonymize_text_with_spans(self, text: str, spans: List[Dict], mask_token: str = "[MASK]") -> str:
        """Anonymize text by replacing annotated spans with mask tokens."""
        if not spans:
            return text
        
        # Sort spans by start position in reverse order to avoid offset issues
        sorted_spans = sorted(spans, key=lambda x: x["start"], reverse=True)
        
        anonymized_text = text
        for span in sorted_spans:
            start = span["start"]
            end = span["end"]
            
            # Replace the span with mask token
            anonymized_text = anonymized_text[:start] + mask_token + anonymized_text[end:]
        
        return anonymized_text
    
    def get_individuals_info(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], 
                           config: RuntimeConfig) -> Dict[str, Any]:
        """Extract individual statistics and label mappings."""
        logger.info("operation_start", extra={"operation": "get_individuals_info"})
        
        train_individuals = set(train_df[config.individual_name_column])
        eval_individuals = set()
        
        for name, eval_df in eval_dfs.items():
            if name != config.dev_set_column_name:
                eval_individuals.update(set(eval_df[config.individual_name_column]))
        
        all_individuals = train_individuals.union(eval_individuals)
        no_train_individuals = eval_individuals - train_individuals
        no_eval_individuals = train_individuals - eval_individuals
        
        # Create label mappings
        sorted_individuals = sorted(list(all_individuals))
        label_to_name = {idx: name for idx, name in enumerate(sorted_individuals)}
        name_to_label = {name: idx for idx, name in label_to_name.items()}
        num_labels = len(name_to_label)
        
        result = {
            "train_individuals": train_individuals,
            "eval_individuals": eval_individuals,
            "all_individuals": all_individuals,
            "no_train_individuals": no_train_individuals,
            "no_eval_individuals": no_eval_individuals,
            "label_to_name": label_to_name,
            "name_to_label": name_to_label,
            "num_labels": num_labels
        }
        
        logger.info("operation_complete", extra={
            "operation": "get_individuals_info",
            "total_individuals": num_labels,
            "train_only": len(no_eval_individuals),
            "eval_only": len(no_train_individuals)
        })
        
        return result
    
    def _anonymize_background_knowledge(self, train_df: pd.DataFrame, config: RuntimeConfig) -> pd.DataFrame:
        """Anonymize background knowledge using NER."""
        with spacy_nlp_context() as nlp:
            anonymized_df = self._anonymize_dataframe(train_df, nlp)
        
        if config.only_use_anonymized_background_knowledge:
            return anonymized_df
        else:
            return pd.concat([train_df, anonymized_df], ignore_index=True, copy=False)
    
    def _anonymize_dataframe(self, df: pd.DataFrame, nlp: Any) -> pd.DataFrame:
        """Anonymize text in dataframe using spaCy NER."""
        anonymized_df = df.copy(deep=True)
        column_name = anonymized_df.columns[1]  # Text column
        texts = anonymized_df[column_name]
        
        for i, text in enumerate(tqdm(texts, desc=f"Anonymizing {column_name} documents")):
            doc = nlp(text)
            new_text = text
            
            # Replace entities with their labels (in reverse order to preserve positions)
            for entity in reversed(doc.ents):
                start = entity.start_char
                end = start + len(entity.text)
                new_text = new_text[:start] + entity.label_ + new_text[end:]
            
            texts[i] = new_text
            
            # Periodic cleanup
            del doc
            if i % 5 == 0:
                gc.collect()
        
        return anonymized_df
    
    def _curate_dataframe(self, df: pd.DataFrame, nlp: Any) -> None:
        """Curate text in dataframe (lemmatization, stopword removal, etc.)."""
        special_chars_pattern = re.compile(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ./]+")
        stopwords = nlp.Defaults.stop_words
        
        column_name = df.columns[1]  # Text column
        texts = df[column_name]
        
        for i, text in enumerate(tqdm(texts, desc=f"Curating {column_name} documents")):
            doc = nlp(text)
            new_text = ""
            
            for token in doc:
                if token.text not in stopwords:
                    # Lemmatize
                    token_text = token.lemma_ if token.lemma_ else token.text
                    # Remove special characters
                    token_text = re.sub(special_chars_pattern, '', token_text)
                    # Add to new text
                    new_text += ("" if token_text == "." else " ") + token_text
            
            texts[i] = new_text
            
            # Periodic cleanup
            del doc
            if i % 5 == 0:
                gc.collect()


@dataclass(frozen=True)
class TRIDatasetBuilder(DatasetBuilder):
    """Concrete implementation for building PyTorch datasets."""
    
    def create_dataset(self, df: pd.DataFrame, tokenizer: Any, name_to_label: Dict[str, int],
                      uses_labels: bool, sliding_window: str, block_size: int) -> Dataset:
        """Create TRI dataset from DataFrame."""
        logger.info("operation_start", extra={
            "operation": "create_dataset",
            "samples": len(df),
            "uses_labels": uses_labels,
            "sliding_window": sliding_window
        })
        
        dataset = TRIDataset(df, tokenizer, name_to_label, uses_labels, sliding_window, block_size)
        
        logger.info("operation_complete", extra={
            "operation": "create_dataset",
            "dataset_size": len(dataset)
        })
        
        return dataset


class TRIDataset(Dataset):
    """PyTorch Dataset for TRI text data."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: Any, name_to_label: Dict[str, int], 
                 return_labels: bool, sliding_window_config: str, tokenization_block_size: int):
        assert len(df.columns) == 2, "DataFrame must have exactly 2 columns: name and text"
        
        self.df = df
        self.tokenizer = tokenizer
        self.name_to_label = name_to_label
        self.return_labels = return_labels
        self.tokenization_block_size = tokenization_block_size
        
        # Parse sliding window configuration
        self.sliding_window_config = sliding_window_config
        try:
            sw_elements = [int(x) for x in sliding_window_config.split("-")]
            self.sliding_window_length = sw_elements[0]
            self.sliding_window_overlap = sw_elements[1]
            self.use_sliding_window = True
        except:
            self.use_sliding_window = False
        
        if (self.use_sliding_window and 
            self.sliding_window_length > self.tokenizer.model_max_length):
            raise ValueError(
                f"Sliding window length ({self.sliding_window_length}) must be "
                f"<= model max length ({self.tokenizer.model_max_length})"
            )
        
        # Generate tokenized inputs and labels
        self.inputs, self.labels = self._generate_inputs_and_labels()
    
    def _generate_inputs_and_labels(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Generate tokenized inputs and labels from DataFrame."""
        texts = list(self.df[self.df.columns[1]])
        names = list(self.df[self.df.columns[0]])
        labels = [self.name_to_label[name] for name in names]
        
        if self.use_sliding_window:
            processed_texts = texts
            processed_labels = labels
        else:
            # Use sentence splitting
            processed_texts, processed_labels = self._split_into_sentences(texts, labels)
        
        return self._tokenize_data(processed_texts, processed_labels)
    
    def _split_into_sentences(self, texts: list[str], labels: list[int]) -> Tuple[list[str], list[int]]:
        """Split texts into sentences using spaCy."""
        sentence_texts = []
        sentence_labels = []
        
        # Load minimal spaCy model for sentence splitting
        disable_components = ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"]
        with spacy_nlp_context(disable_components) as nlp:
            for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Sentence splitting"):
                for paragraph in text.split("\n"):
                    if paragraph.strip():
                        doc = nlp(paragraph)
                        for sentence in doc.sents:
                            sentence_text = " ".join(token.text for token in sentence)
                            
                            # Check sentence length
                            token_count = len(self.tokenizer.encode(sentence_text, add_special_tokens=True))
                            if token_count <= self.tokenizer.model_max_length:
                                sentence_texts.append(sentence_text)
                                sentence_labels.append(label)
                            else:
                                logger.warning("sentence_too_long", extra={
                                    "length": token_count,
                                    "max_length": self.tokenizer.model_max_length,
                                    "label": label
                                })
                        
                        del doc
        
        return sentence_texts, sentence_labels
    
    def _tokenize_data(self, texts: list[str], labels: list[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Tokenize texts and return input tensors and labels."""
        if self.use_sliding_window:
            input_length = self.sliding_window_length
            padding_strategy = "longest"
        else:
            input_length = self.tokenizer.model_max_length
            padding_strategy = "max_length"
        
        all_input_ids = torch.zeros((0, input_length), dtype=torch.int)
        all_attention_masks = torch.zeros((0, input_length), dtype=torch.int)
        all_labels = []
        
        # Process in blocks for memory efficiency
        with tqdm(total=len(texts), desc="Tokenizing") as pbar:
            for start_idx in range(0, len(texts), self.tokenization_block_size):
                end_idx = min(start_idx + self.tokenization_block_size, len(texts))
                
                block_inputs = self.tokenizer(
                    texts[start_idx:end_idx],
                    add_special_tokens=not self.use_sliding_window,
                    padding=padding_strategy,
                    truncation=False,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt"
                )
                
                gc.collect()
                
                if self.use_sliding_window:
                    (all_input_ids, all_attention_masks, 
                     all_labels) = self._apply_sliding_window(
                        labels[start_idx:end_idx], input_length, 
                        all_input_ids, all_attention_masks, all_labels, 
                        pbar, block_inputs
                    )
                else:
                    all_input_ids = torch.cat((all_input_ids, block_inputs["input_ids"]))
                    all_attention_masks = torch.cat((all_attention_masks, block_inputs["attention_mask"]))
                    all_labels.extend(labels[start_idx:end_idx])
                    pbar.update(end_idx - start_idx)
        
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_masks}
        labels_tensor = torch.tensor(all_labels)
        
        return inputs, labels_tensor
    
    def _apply_sliding_window(self, block_labels: list[int], input_length: int,
                             all_input_ids: torch.Tensor, all_attention_masks: torch.Tensor,
                             all_labels: list[int], pbar: tqdm, 
                             block_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Apply sliding window to tokenized inputs."""
        # Calculate number of windows needed
        old_seq_length = block_inputs["input_ids"].size(1)
        window_increment = self.sliding_window_length - self.sliding_window_overlap - 2  # Account for CLS/SEP
        
        n_windows = 0
        for attention_mask in block_inputs["attention_mask"]:
            seq_pos = 0
            while seq_pos < old_seq_length:
                window_end = min(seq_pos + self.sliding_window_length - 2, old_seq_length)
                window_mask = attention_mask[seq_pos:window_end]
                
                if window_mask[-1] == 0 or window_end == old_seq_length:
                    n_windows += 1
                    break
                
                seq_pos += window_increment
                n_windows += 1
        
        # Allocate memory
        window_ids = torch.empty((n_windows, input_length), dtype=torch.int)
        window_masks = torch.empty((n_windows, input_length), dtype=torch.int)
        
        # Process sliding windows
        window_idx = 0
        for seq_idx, (input_ids, attention_mask) in enumerate(zip(
            block_inputs["input_ids"], block_inputs["attention_mask"]
        )):
            seq_pos = 0
            windows_in_sequence = 0
            
            while seq_pos < old_seq_length:
                window_end = min(seq_pos + self.sliding_window_length - 2, old_seq_length)
                
                win_ids = input_ids[seq_pos:window_end]
                win_mask = attention_mask[seq_pos:window_end]
                
                # Add CLS and SEP tokens
                num_attention_tokens = torch.count_nonzero(win_mask)
                if num_attention_tokens == len(win_mask):  # Full window
                    win_ids = torch.cat([
                        torch.tensor([self.tokenizer.cls_token_id]),
                        win_ids,
                        torch.tensor([self.tokenizer.sep_token_id])
                    ])
                    win_mask = torch.cat([torch.tensor([1]), win_mask, torch.tensor([1])])
                else:  # Partial window
                    win_ids[num_attention_tokens] = self.tokenizer.sep_token_id
                    win_mask[num_attention_tokens] = 1
                    win_ids = torch.cat([
                        torch.tensor([self.tokenizer.cls_token_id]),
                        win_ids,
                        torch.tensor([self.tokenizer.pad_token_id])
                    ])
                    win_mask = torch.cat([torch.tensor([1]), win_mask, torch.tensor([0])])
                
                # Pad to sliding window length
                padding_length = self.sliding_window_length - len(win_ids)
                if padding_length > 0:
                    padding = torch.zeros(padding_length, dtype=win_ids.dtype)
                    win_ids = torch.cat([win_ids, padding])
                    win_mask = torch.cat([win_mask, padding])
                
                window_ids[window_idx] = win_ids
                window_masks[window_idx] = win_mask
                
                window_idx += 1
                windows_in_sequence += 1
                
                # Check if this is the last window
                if window_end == old_seq_length or win_mask[-1] == 0:
                    break
                
                seq_pos += window_increment
            
            all_labels.extend([block_labels[seq_idx]] * windows_in_sequence)
            pbar.update(1)
        
        # Concatenate with existing data
        all_input_ids = torch.cat([all_input_ids, window_ids])
        all_attention_masks = torch.cat([all_attention_masks, window_masks])
        
        gc.collect()
        return all_input_ids, all_attention_masks, all_labels
    
    def __len__(self) -> int:
        return len(self.inputs["input_ids"])
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.inputs.items()}
        
        if self.return_labels:
            item["labels"] = self.labels[index]
        
        return item


@dataclass(frozen=True)
class TRIModelManager(ModelManager):
    """Concrete implementation for model management operations."""
    
    def create_base_model(self, config: RuntimeConfig) -> Tuple[Any, Any]:
        """Create base model and tokenizer."""
        logger.info("operation_start", extra={"operation": "create_base_model", "model": config.base_model_name})
        
        with base_model_context(config) as (model, tokenizer):
            # Return copies since context manager will clean up originals
            model_copy = type(model).from_pretrained(config.base_model_name)
            tokenizer_copy = AutoTokenizer.from_pretrained(config.base_model_name)
            
            logger.info("operation_complete", extra={"operation": "create_base_model"})
            return model_copy, tokenizer_copy
    
    def perform_pretraining(self, base_model: Any, tokenizer: Any, dataset: Dataset, config: RuntimeConfig) -> Any:
        """Perform additional pretraining on base model."""
        logger.info("operation_start", extra={"operation": "additional_pretraining"})
        
        with mlm_model_context(config, base_model) as mlm_model:
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer, 
                mlm_probability=config.pretraining_mlm_probability
            )
            
            # Create trainer
            trainer = self._create_trainer(
                mlm_model, config.pretraining_config, dataset, 
                data_collator=data_collator
            )
            
            # Train
            trainer.train()
            
            # Move base model back to CPU
            base_model.cpu()
        
        logger.info("operation_complete", extra={"operation": "additional_pretraining"})
        return base_model
    
    def perform_finetuning(self, base_model: Any, tokenizer: Any, train_dataset: Dataset,
                          eval_datasets: Dict[str, Dataset], config: RuntimeConfig) -> Tuple[Any, Trainer]:
        """Perform finetuning for text re-identification."""
        logger.info("operation_start", extra={"operation": "finetuning"})
        
        # Determine number of labels from first eval dataset
        first_eval_dataset = next(iter(eval_datasets.values()))
        num_labels = len(set(first_eval_dataset.labels.tolist()))
        
        # Force all operations to CPU to avoid MPS device issues
        device = torch.device("cpu")
        
        with classification_model_context(config, base_model, num_labels) as clf_model:
            # Ensure model is on CPU
            clf_model.to(device)
            
            # Create trainer
            trainer = self._create_trainer(
                clf_model, config.finetuning_config, train_dataset,
                eval_datasets_dict=eval_datasets, config=config
            )
            
            # Train
            training_results = trainer.train()
            
            # Return copies since context manager will clean up originals
            final_model = type(clf_model).from_pretrained(
                config.base_model_name, 
                num_labels=num_labels
            )
            final_model.load_state_dict(clf_model.state_dict())
            final_model.to(device)  # Ensure final model is on CPU
            
            logger.info("operation_complete", extra={
                "operation": "finetuning",
                "final_loss": training_results.training_loss,
                "device": str(device)
            })
            
            return final_model, trainer
    
    def save_model(self, model: Any, tokenizer: Any, config: RuntimeConfig) -> Pipeline:
        """Save trained model and return pipeline."""
        logger.info("operation_start", extra={"operation": "save_model", "path": config.tri_pipe_path})
        
        # Ensure model is on CPU before saving
        device = torch.device("cpu")
        model.to(device)
        
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # Force CPU
        pipe.save_pretrained(config.tri_pipe_path)
        
        logger.info("operation_complete", extra={"operation": "save_model", "device": str(device)})
        return pipe
    
    def load_model(self, config: RuntimeConfig) -> Tuple[Any, Any]:
        """Load pre-trained model and tokenizer."""
        logger.info("operation_start", extra={"operation": "load_model", "path": config.tri_pipe_path})
        
        with pretrained_model_context(config) as (model, tokenizer):
            # Return copies since context manager will clean up originals
            model_copy = type(model).from_pretrained(config.tri_pipe_path)
            tokenizer_copy = AutoTokenizer.from_pretrained(config.tri_pipe_path)
            
            logger.info("operation_complete", extra={"operation": "load_model"})
            return model_copy, tokenizer_copy
    
    def _create_trainer(self, model: Any, task_config: TrainingConfig, train_dataset: Dataset,
                       eval_datasets_dict: Dict[str, Dataset] = None, data_collator: Any = None,
                       config: RuntimeConfig = None) -> Trainer:
        """Create HuggingFace Trainer with appropriate configuration."""
        
        if task_config.is_for_mlm:
            # Settings for masked language modeling
            eval_strategy = "no"
            save_strategy = "no"
            load_best_model_at_end = False
            metric_for_best_model = None
            eval_datasets_dict = None
            results_filepath = None
        else:
            # Settings for classification finetuning
            eval_strategy = "epoch"
            save_strategy = "epoch"
            load_best_model_at_end = True
            
            if config and config.dev_set_column_name:
                metric_for_best_model = f"{config.dev_set_column_name}_eval_Accuracy"
            else:
                metric_for_best_model = "avg_Accuracy"
            
            results_filepath = config.results_file_path if config else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=task_config.trainer_folder_path,
            overwrite_output_dir=True,
            load_best_model_at_end=load_best_model_at_end,
            save_strategy=save_strategy,
            save_total_limit=1,
            num_train_epochs=task_config.epochs,
            per_device_train_batch_size=task_config.batch_size,
            per_device_eval_batch_size=task_config.batch_size,
            logging_strategy="epoch",
            logging_steps=500,
            eval_strategy=eval_strategy,
            disable_tqdm=False,
            eval_accumulation_steps=5,
            dataloader_num_workers=0,
            metric_for_best_model=metric_for_best_model,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(), 
            lr=task_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-06,
            weight_decay=0.0
        )
        scheduler = get_constant_schedule(optimizer)
        
        # Use Accelerate with CPU-only for MPS compatibility
        accelerator = Accelerator(cpu=True)
        model, optimizer, scheduler, train_dataset = accelerator.prepare(
            model, optimizer, scheduler, train_dataset
        )
        
        # Create trainer
        if task_config.is_for_mlm:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                optimizers=[optimizer, scheduler],
                data_collator=data_collator
            )
        else:
            trainer = TRITrainer(
                results_filepath,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_datasets_dict,
                optimizers=[optimizer, scheduler],
                compute_metrics=compute_metrics
            )
        
        return trainer


@dataclass(frozen=True)
class TRIPredictor(Predictor):
    """Concrete implementation for making predictions."""
    
    def predict(self, trainer: Trainer, eval_datasets: Dict[str, Dataset]) -> Dict[str, Dict[str, float]]:
        """Make predictions and return results."""
        logger.info("operation_start", extra={"operation": "predict"})
        
        results = trainer.evaluate()
        
        if hasattr(trainer, 'all_results') and trainer.all_results:
            structured_results = trainer.all_results[-1]
        else:
            # Fallback for standard trainer
            structured_results = {"default": results}
        
        logger.info("operation_complete", extra={
            "operation": "predict",
            "results": {name: res.get('eval_Accuracy', 0) for name, res in structured_results.items()}
        })
        
        return structured_results


@dataclass(frozen=True)
class TRIConfigManager(ConfigManager):
    """Concrete implementation for configuration management."""
    
    def validate_config(self, config: RuntimeConfig) -> None:
        """Validate configuration parameters."""
        from config import validate_data_columns
        
        # Read data to validate columns
        if config.data_file_path.endswith(".json"):
            df = pd.read_json(config.data_file_path)
        elif config.data_file_path.endswith(".csv"):
            df = pd.read_csv(config.data_file_path)
        else:
            raise ValueError(f"Unsupported file format: {config.data_file_path}")
        
        validate_data_columns(config, df.columns.tolist())
        
        logger.info("config_validated", extra={"config_file": config.data_file_path})
    
    def ensure_output_directory(self, config: RuntimeConfig) -> None:
        """Create output directory if it doesn't exist."""
        from config import ensure_output_directory
        ensure_output_directory(config)
        
        logger.info("directory_created", extra={"path": config.output_folder_path})


@dataclass(frozen=True)
class TRIStorageManager(StorageManager):
    """Concrete implementation for data persistence operations."""
    
    def save_pretreatment(self, train_df: pd.DataFrame, eval_dfs: Dict[str, pd.DataFrame], 
                         config: RuntimeConfig) -> None:
        """Save pretreated data to disk."""
        logger.info("operation_start", extra={"operation": "save_pretreatment"})
        
        with open(config.pretreated_data_path, "w") as f:
            json.dump((
                train_df.to_json(orient="records"),
                {name: df.to_json(orient="records") for name, df in eval_dfs.items()}
            ), f)
        
        logger.info("operation_complete", extra={
            "operation": "save_pretreatment",
            "path": config.pretreated_data_path
        })
    
    def load_pretreatment(self, config: RuntimeConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load pretreated data from disk."""
        logger.info("operation_start", extra={"operation": "load_pretreatment"})
        
        with open(config.pretreated_data_path, "r") as f:
            train_df_json, eval_dfs_json = json.load(f)
        
        train_df = pd.read_json(StringIO(train_df_json))
        eval_dfs = OrderedDict([
            (name, pd.read_json(StringIO(df_json))) 
            for name, df_json in eval_dfs_json.items()
        ])
        
        logger.info("operation_complete", extra={
            "operation": "load_pretreatment",
            "train_samples": len(train_df),
            "eval_datasets": list(eval_dfs.keys())
        })
        
        return train_df, eval_dfs
    
    def pretreatment_exists(self, config: RuntimeConfig) -> bool:
        """Check if pretreated data exists."""
        return os.path.isfile(config.pretreated_data_path)
    
    def save_pretrained_model(self, model: Any, config: RuntimeConfig) -> None:
        """Save pretrained model to disk."""
        logger.info("operation_start", extra={"operation": "save_pretrained_model"})
        
        torch.save(model.state_dict(), config.pretrained_model_path)
        
        logger.info("operation_complete", extra={
            "operation": "save_pretrained_model",
            "path": config.pretrained_model_path
        })
    
    def load_pretrained_model(self, model: Any, config: RuntimeConfig) -> Any:
        """Load pretrained model from disk."""
        logger.info("operation_start", extra={"operation": "load_pretrained_model"})
        
        model.load_state_dict(torch.load(config.pretrained_model_path))
        
        logger.info("operation_complete", extra={"operation": "load_pretrained_model"})
        return model
    
    def pretrained_model_exists(self, config: RuntimeConfig) -> bool:
        """Check if pretrained model exists."""
        return os.path.exists(config.pretrained_model_path)


@dataclass(frozen=True)
class TRIWorkflowOrchestrator(WorkflowOrchestrator):
    """Concrete implementation for orchestrating the entire TRI workflow."""
    
    data_processor: DataProcessor
    dataset_builder: DatasetBuilder
    model_manager: ModelManager
    predictor: Predictor
    config_manager: ConfigManager
    storage_manager: StorageManager
    
    def run_data_processing(self, config: RuntimeConfig) -> Dict[str, Any]:
        """Execute data processing phase."""
        logger.info("phase_start", extra={"phase": "data_processing"})
        
        # Ensure output directory exists
        self.config_manager.ensure_output_directory(config)
        
        # Try to load pretreated data if requested
        if config.load_saved_pretreatment and self.storage_manager.pretreatment_exists(config):
            logger.info("loading_saved_pretreatment")
            train_df, eval_dfs = self.storage_manager.load_pretreatment(config)
            pretreatment_done = False
            
        else:
            # Read and process raw data
            train_df, eval_dfs = self.data_processor.load_data(config)
            pretreatment_done = True
        
        # Get individual statistics
        individuals_info = self.data_processor.get_individuals_info(train_df, eval_dfs, config)
        
        # Save pretreatment if requested and modifications were made
        if config.save_pretreatment and pretreatment_done:
            self.storage_manager.save_pretreatment(train_df, eval_dfs, config)
        
        result = {
            "train_df": train_df,
            "eval_dfs": eval_dfs,
            **individuals_info
        }
        
        logger.info("phase_complete", extra={"phase": "data_processing"})
        return result
    
    def run_model_building(self, data_info: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Any]:
        """Execute model building phase."""
        logger.info("phase_start", extra={"phase": "model_building"})
        
        train_df = data_info["train_df"]
        eval_dfs = data_info["eval_dfs"]
        name_to_label = data_info["name_to_label"]
        
        # Check if already trained model exists
        if config.load_saved_finetuning and os.path.exists(config.tri_pipe_path):
            logger.info("loading_saved_model")
            tri_model, tokenizer = self.model_manager.load_model(config)
            
            # Create datasets for evaluation
            finetuning_dataset = self.dataset_builder.create_dataset(
                train_df, tokenizer, name_to_label, 
                config.finetuning_config.uses_labels, 
                config.finetuning_config.sliding_window, 
                config.tokenization_block_size
            )
            eval_datasets_dict = OrderedDict([
                (name, self.dataset_builder.create_dataset(
                    eval_df, tokenizer, name_to_label,
                    config.finetuning_config.uses_labels,
                    config.finetuning_config.sliding_window,
                    config.tokenization_block_size
                ))
                for name, eval_df in eval_dfs.items()
            ])
            
            # Create trainer for evaluation
            trainer = self.model_manager._create_trainer(
                tri_model, config.finetuning_config, finetuning_dataset,
                eval_datasets_dict=eval_datasets_dict, config=config
            )
            
        else:
            # Create and train new model
            base_model, tokenizer = self.model_manager.create_base_model(config)
            
            # Additional pretraining if requested
            if config.use_additional_pretraining:
                if (config.load_saved_pretraining and 
                    self.storage_manager.pretrained_model_exists(config)):
                    logger.info("loading_pretrained_model")
                    base_model = self.storage_manager.load_pretrained_model(base_model, config)
                else:
                    # Create pretraining dataset
                    pretraining_dataset = self.dataset_builder.create_dataset(
                        train_df, tokenizer, name_to_label,
                        config.pretraining_config.uses_labels,
                        config.pretraining_config.sliding_window,
                        config.tokenization_block_size
                    )
                    
                    # Perform additional pretraining
                    base_model = self.model_manager.perform_pretraining(
                        base_model, tokenizer, pretraining_dataset, config
                    )
                    
                    # Save pretrained model if requested
                    if config.save_additional_pretraining:
                        self.storage_manager.save_pretrained_model(base_model, config)
            
            # Create finetuning datasets
            finetuning_dataset = self.dataset_builder.create_dataset(
                train_df, tokenizer, name_to_label,
                config.finetuning_config.uses_labels,
                config.finetuning_config.sliding_window,
                config.tokenization_block_size
            )
            
            eval_datasets_dict = OrderedDict()
            for name, eval_df in eval_dfs.items():
                # Debug: check dataframe structure
                print(f"DEBUG: {name} has columns: {list(eval_df.columns)}")
                print(f"DEBUG: {name} shape: {eval_df.shape}")
                
                # Ensure exactly 2 columns: name and text
                if len(eval_df.columns) != 2:
                    # Take first column as name, last column as text
                    eval_df = eval_df.iloc[:, [0, -1]]
                    eval_df.columns = ['name', 'text']
                
                dataset = self.dataset_builder.create_dataset(
                    eval_df, tokenizer, name_to_label,
                    config.finetuning_config.uses_labels,
                    config.finetuning_config.sliding_window,
                    config.tokenization_block_size
                )
                eval_datasets_dict[name] = dataset
            
            # Perform finetuning
            tri_model, trainer = self.model_manager.perform_finetuning(
                base_model, tokenizer, finetuning_dataset, eval_datasets_dict, config
            )
            
            # Save finetuned model if requested
            if config.save_finetuning:
                pipe = self.model_manager.save_model(tri_model, tokenizer, config)
        
        result = {
            "tri_model": tri_model,
            "tokenizer": tokenizer,
            "trainer": trainer,
            "eval_datasets_dict": eval_datasets_dict
        }
        
        logger.info("phase_complete", extra={"phase": "model_building"})
        return result
    
    def run_prediction(self, model_info: Dict[str, Any], config: RuntimeConfig) -> Dict[str, Dict[str, float]]:
        """Execute prediction phase."""
        logger.info("phase_start", extra={"phase": "prediction"})
        
        trainer = model_info["trainer"]
        eval_datasets_dict = model_info["eval_datasets_dict"]
        
        results = self.predictor.predict(trainer, eval_datasets_dict)
        
        logger.info("phase_complete", extra={"phase": "prediction"})
        return results


class TRITrainer(Trainer):
    """Custom Trainer with enhanced evaluation capabilities."""
    
    def __init__(self, results_filepath: str = None, **kwargs):
        super().__init__(**kwargs)
        self.results_filepath = results_filepath
        
        if (self.results_filepath and "eval_dataset" in self.__dict__ and 
            isinstance(self.eval_dataset, dict)):
            self.do_custom_eval = True
            self.eval_datasets_dict = self.eval_dataset
            self.all_results = []
            self.evaluation_epoch = 1
            self._initialize_results_file()
        else:
            self.do_custom_eval = False
    
    def _current_time_str(self) -> str:
        """Get current timestamp as string."""
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    def _initialize_results_file(self) -> None:
        """Initialize results file with headers."""
        header = f"{self._current_time_str()}\nTime,Epoch"
        for dataset_name in self.eval_datasets_dict.keys():
            header += f",{dataset_name}"
        header += ",Average\n"
        self._write_results(header)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with multiple datasets."""
        if not self.do_custom_eval:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        metrics = OrderedDict()
        structured_results = OrderedDict()
        avg_loss = 0
        avg_acc = 0
        
        loss_key = f"{metric_key_prefix}_loss"
        acc_key = f"{metric_key_prefix}_Accuracy"
        
        # Evaluate on each dataset
        for dataset_name, dataset in self.eval_datasets_dict.items():
            dataset_metrics = super().evaluate(
                eval_dataset=dataset, 
                ignore_keys=ignore_keys, 
                metric_key_prefix=metric_key_prefix
            )
            
            avg_loss += dataset_metrics[loss_key] / len(self.eval_datasets_dict)
            avg_acc += dataset_metrics[acc_key] / len(self.eval_datasets_dict)
            
            structured_results[dataset_name] = dataset_metrics
            
            # Add dataset name to metrics keys
            for key, val in dataset_metrics.items():
                metrics[f"{metric_key_prefix}_{dataset_name}_{key}"] = val
        
        # Add average metrics
        metrics.update({
            f"{metric_key_prefix}_avg_loss": avg_loss,
            f"{metric_key_prefix}_avg_Accuracy": avg_acc,
            loss_key: avg_loss,
            acc_key: avg_acc
        })
        
        # Store results
        self._store_results(metrics)
        self.all_results.append(structured_results)
        self.evaluation_epoch += 1
        
        return metrics
    
    def _store_results(self, eval_results: Dict[str, float]) -> None:
        """Store evaluation results to file."""
        current_time = self._current_time_str()
        try:
            results_text = f"{current_time},{self.evaluation_epoch}"
            for key, value in eval_results.items():
                if key.endswith("_Accuracy"):
                    results_text += f",{value:.3f}"
            results_text += "\n"
            self._write_results(results_text)
        except Exception as e:
            error_msg = f"{current_time}, Error writing results for epoch {self.evaluation_epoch} ({e})"
            self._write_results(error_msg)
            logger.error("results_write_error", extra={"epoch": self.evaluation_epoch, "error": str(e)})
    
    def _write_results(self, text: str) -> None:
        """Write text to results file."""
        with open(self.results_filepath, "a+") as f:
            f.write(text)


def compute_metrics(results):
    """Compute accuracy metrics for TRI evaluation."""
    logits, labels = results
    
    # Convert to torch tensors
    logits = torch.from_numpy(logits)
    
    # Group logits by label (sum predictions for same individual)
    logits_dict = {}
    for logit, label in zip(logits, labels):
        current_logits = logits_dict.get(label, torch.zeros_like(logit))
        logits_dict[label] = current_logits + logit
    
    # Compute final predictions
    num_predictions = len(logits_dict)
    all_predictions = torch.zeros(num_predictions, device="cpu")
    all_labels = torch.zeros(num_predictions, device="cpu")
    
    for idx, (label, summed_logits) in enumerate(logits_dict.items()):
        all_labels[idx] = label
        probabilities = F.softmax(summed_logits, dim=-1)
        all_predictions[idx] = torch.argmax(probabilities)
    
    # Calculate accuracy
    correct_predictions = torch.sum(all_predictions == all_labels)
    accuracy = (float(correct_predictions) / num_predictions) * 100
    
    return {"Accuracy": accuracy}