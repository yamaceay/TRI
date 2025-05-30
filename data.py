#region ###################################### Imports ######################################
import sys
import os
import json
import gc
import re
import logging
from collections import OrderedDict

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from io import StringIO
import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

import torch
from torch.utils.data import Dataset

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

#endregion

#region ###################################### Configuration file argument ######################################

#region ################### Arguments parsing ###################
def argument_parsing():
    if (args_count := len(sys.argv)) > 2:
        logging.exception(Exception(f"One argument expected, got {args_count - 1}"))
    elif args_count < 2:
        logging.exception(Exception("You must specify the JSON configuration filepath as first argument"))

    target_dir = sys.argv[1]
    return target_dir

#endregion

#region ################### JSON file loading ###################
def get_config_from_file(target_dir):
    if not target_dir.endswith(".json"):
        logging.exception(f"The configuration file {target_dir} needs to have json format (end with .json)")
    elif not os.path.isfile(target_dir):
        logging.exception(f"The JSON configuration file {target_dir} doesn't exist")

    with open(target_dir, "r") as f:
        config = json.load(f)
    return config
#endregion

#endregion

#region ###################################### TRI class ######################################

class TRI():
    #region ################### Properties ###################

    #region ########## Mandatory configs ##########

    mandatory_configs_names = ["output_folder_path", "data_file_path",
        "individual_name_column", "background_knowledge_column"]
    output_folder_path = None
    data_file_path = None
    individual_name_column = None
    background_knowledge_column = None

    #endregion

    #region ########## Optional configs with default values ##########

    optional_configs_names = ["load_saved_pretreatment", "add_non_saved_anonymizations",
        "anonymize_background_knowledge", "only_use_anonymized_background_knowledge", 
        "use_document_curation", "save_pretreatment", "load_saved_finetuning", "base_model_name", 
        "tokenization_block_size", "use_additional_pretraining", "save_additional_pretraining",
        "load_saved_pretraining", "pretraining_epochs", "pretraining_batch_size",
        "pretraining_learning_rate", "pretraining_mlm_probability", "pretraining_sliding_window",
        "save_finetuning", "load_saved_finetuning", "finetuning_epochs", "finetuning_batch_size",
        "finetuning_learning_rate", "finetuning_sliding_window", "dev_set_column_name"]
    load_saved_pretreatment = True
    add_non_saved_anonymizations = True
    anonymize_background_knowledge = True
    only_use_anonymized_background_knowledge = True
    use_document_curation = True
    save_pretreatment = True
    base_model_name = "distilbert-base-uncased"
    tokenization_block_size = 250
    use_additional_pretraining = True
    save_additional_pretraining = True
    load_saved_pretraining = True
    pretraining_epochs = 3
    pretraining_batch_size = 8
    pretraining_learning_rate = 5e-05
    pretraining_mlm_probability = 0.15
    pretraining_sliding_window = "512-128"
    save_finetuning = True
    load_saved_finetuning = True
    finetuning_epochs = 15
    finetuning_batch_size = 16
    finetuning_learning_rate = 5e-05
    finetuning_sliding_window = "100-25"
    dev_set_column_name = False

    #endregion

    #region ########## Derived configs ##########

    pretreated_data_path:str = None

    #endregion

    #region ########## Functional properties ##########

    # Data
    data_df:pd.DataFrame = None
    train_df:pd.DataFrame = None
    eval_dfs:dict = None
    train_individuals:set = None
    eval_individuals:set = None
    all_individuals:set = None
    no_train_individuals:set = None
    no_eval_individuals:set = None
    label_to_name:dict = None
    name_to_label:dict = None
    spacy_nlp = None
    pretreated_data_loaded:bool = None

    #endregion

    #endregion

    #region ################### Constructor and configurations ###################

    def __init__(self, **kwargs):
        self.set_configs(**kwargs, are_mandatory_configs_required=True)

    def set_configs(self, are_mandatory_configs_required=False, **kwargs):
        arguments = kwargs.copy()

        # Mandatory configs
        for setting_name in self.mandatory_configs_names:
            value = arguments.get(setting_name, None)
            if isinstance(value, str):
                self.__dict__[setting_name] = arguments[setting_name]
                del arguments[setting_name]
            elif are_mandatory_configs_required:
                raise AttributeError(f"Mandatory argument {setting_name} is not defined or it is not a string")
        
        # Store remaining optional configs
        for (opt_setting_name, opt_setting_value) in arguments.items():
            if opt_setting_name in self.optional_configs_names:                
                if isinstance(opt_setting_value, str) or isinstance(opt_setting_value, int) or \
                isinstance(opt_setting_value, float) or isinstance(opt_setting_value, bool):
                    self.__dict__[opt_setting_name] = opt_setting_value
                else:
                    raise AttributeError(f"Optional argument {opt_setting_name} is not a string, integer, float or boolean.")
            else:
                logging.warning(f"Unrecognized setting name {opt_setting_name}")

        # Generate derived configs
        self.pretreated_data_path = os.path.join(self.output_folder_path, "Pretreated_Data.json")

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            self.device = torch.device("cpu")

    #endregion

    #region ################### Data ###################

    def run_data(self, verbose=True):
        if verbose: logging.info("######### START: DATA #########")
        self.pretreated_data_loaded = False
        self.pretreatment_done = False

        # Create output directory if it does not exist
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path, exist_ok=True)

        # Read pretreated data if it exists        
        if self.load_saved_pretreatment and os.path.isfile(self.pretreated_data_path):
            if verbose: logging.info("######### START: LOADING SAVED PRETREATED DATA #########")
            self.train_df, self.eval_dfs = self.load_pretreatment()            
            self.pretreated_data_loaded = True

            # If curate non-saved anonymizations and document curation are required
            if self.add_non_saved_anonymizations:
                # Pretreated saved anonymizations
                self.saved_anons = set(self.eval_dfs.keys())

                # Non-pretreated anonymizations from raw file
                new_data_df = self.read_data()
                _, new_eval_dfs = self.split_data(new_data_df)
                self.non_pretreated_anons = set(new_eval_dfs.keys())

                # Find non-pretreated anonymizations not present in pretreated saved anonymizations
                self.non_saved_anons = []
                for anon_name in self.non_pretreated_anons:
                    if not anon_name in self.saved_anons:
                        self.non_saved_anons.append(anon_name)

                # If there are non-pretreated anonymizations not present in saved anonymizations, add them
                if len(self.non_saved_anons) > 0:
                    if verbose: logging.info("######### START: ADDING NON-SAVED ANONYMIZATIONS #########")
                    if verbose: logging.info(f"The following non-saved anonymizations will be added: {self.non_saved_anons}")
                    for anon_name in self.non_saved_anons:
                        # Curate anonymizations if needed
                        if self.use_document_curation:
                            self.curate_df(new_eval_dfs[anon_name], self.load_spacy_nlp())
                        # Add to eval_dfs
                        self.eval_dfs[anon_name] = new_eval_dfs[anon_name]
                    self.pretreatment_done = True
                    if verbose: logging.info("Non-saved anonymizations added")
                    if verbose: logging.info("######### END: ADDING NON-SAVED ANONYMIZATIONS #########")
                else:
                    if verbose: logging.info("There are not non-saved anonymizations to add")
                    if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")
            else:
                if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")

            if verbose: logging.info("######### END: LOADING SAVED PRETREATED DATA #########")

        # Otherwise, read raw data
        else:
            if self.load_saved_pretreatment:
                if verbose: logging.info(f"Impossible to load saved pretreated data, file {self.pretreated_data_path} not found.")

            if verbose: logging.info("######### START: READ RAW DATA FROM FILE #########")

            if verbose: logging.info("Reading data")
            self.data_df = self.read_data()
            if verbose: logging.info("Data reading complete")

            # Split into train and evaluation (dropping rows where no documents are available)
            if verbose: logging.info("Splitting into train (background knowledge) and evaluation (anonymized) sets")
            self.train_df, self.eval_dfs = self.split_data(self.data_df)
            del self.data_df # Remove general dataframe for saving memory
            if verbose: logging.info("Train and evaluation splitting complete")
            
            if verbose: logging.info("######### END: READ RAW DATA FROM FILE #########")

        if verbose: logging.info("######### START: DATA STATISTICS #########")

        # Get individuals found in each set
        res = self.get_individuals(self.train_df, self.eval_dfs)
        self.train_individuals, self.eval_individuals, self.all_individuals, self.no_train_individuals, self.no_eval_individuals = res

        # Generat Label->Name and Name->Label dictionaries
        self.label_to_name, self.name_to_label, self.num_labels = self.get_individuals_labels(self.all_individuals)

        # Show relevant information
        if verbose:
            self.show_data_stats(self.train_df, self.eval_dfs, self.no_eval_individuals, self.no_train_individuals, self.eval_individuals)        

        if verbose: logging.info("######### END: DATA STATISTICS #########")

        # Pretreat data if required and not pretreatment loaded
        if (self.anonymize_background_knowledge or self.use_document_curation) and not self.pretreated_data_loaded:
            if verbose: logging.info("######### START: DATA PRETREATMENT #########")
            
            if self.anonymize_background_knowledge:
                if verbose: logging.info("######### START: BACKGROUND KNOWLEDGE ANONYMIZATION #########")        
                self.train_df = self.anonymize_bk(self.train_df)
                if verbose: logging.info("######### END: BACKGROUND KNOWLEDGE ANONYMIZATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: BACKGROUND KNOWLEDGE ANONYMIZATION #########")

            if self.use_document_curation:
                if verbose: logging.info("######### START: DOCUMENT CURATION #########")
                self.document_curation(self.train_df, self.eval_dfs)
                if verbose: logging.info("######### END: DOCUMENT CURATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: DOCUMENT CURATION #########")            

            self.pretreatment_done = True

            if verbose: logging.info("######### END: DATA PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: DATA PRETREATMENT #########")

        # If save pretreatment is required and there is any pretreatment modification to save
        if self.save_pretreatment and self.pretreatment_done:
            if verbose: logging.info("######### START: SAVE PRETREATMENT #########")
            self.save_pretreatment_dfs(self.train_df, self.eval_dfs)
            if verbose: logging.info("######### END: SAVE PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: SAVE PRETREATMENT #########")
        
        if verbose: logging.info("######### END: DATA #########")

    #region ########## Load pretreatment ##########

    def load_pretreatment(self):
        with open(self.pretreated_data_path, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
        
        train_df = pd.read_json(StringIO(train_df_json_str))
        eval_dfs = OrderedDict([(name, pd.read_json(StringIO(df_json))) for name, df_json in eval_dfs_jsons.items()])

        return train_df, eval_dfs
    
    #endregion

    #region ########## Data reading ##########

    def read_data(self) -> pd.DataFrame:
        if self.data_file_path.endswith(".json"):
            data_df = pd.read_json(self.data_file_path)
        elif self.data_file_path.endswith(".csv"):
            data_df = pd.read_csv(self.data_file_path)
        else:
            raise Exception(f"Unrecognized file extension for data file [{self.data_file_path}]. Compatible formats are JSON and CSV.")
        
        # Check required columns exist
        if not self.individual_name_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the individual name column {self.individual_name_column}")
        if not self.background_knowledge_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the background knowledge column {self.background_knowledge_column}")
        if self.dev_set_column_name is not False and not self.dev_set_column_name in data_df.columns:
            raise Exception(f"Dataframe does not contain the dev set column {self.dev_set_column_name}")
        
        # Check there are additional columns providing texts to re-identify
        anon_cols = [col_name for col_name in data_df.columns if not col_name in [self.individual_name_column, self.background_knowledge_column]]        
        if len(anon_cols) == 0:
            raise Exception(f"Dataframe does not contain columns with texts to re-identify, only individual name and background knowledge columns")
        
        # Sort by individual name
        data_df.sort_values(self.individual_name_column).reset_index(drop=True, inplace=True)

        return data_df

    def split_data(self, data_df:pd.DataFrame):
        data_df.replace('', np.nan, inplace=True)   # Replace empty texts by NaN

        # Training data formed by labeled background knowledge
        train_cols = [self.individual_name_column, self.background_knowledge_column]
        train_df = data_df[train_cols].dropna().reset_index(drop=True)

        # Evaluation data formed by texts to re-identify
        eval_columns = [col for col in data_df.columns if col not in train_cols]
        eval_dfs = {col:data_df[[self.individual_name_column, col]].dropna().reset_index(drop=True) for col in eval_columns}

        return train_df, eval_dfs

    #endregion

    #region ########## Data statistics ##########

    def get_individuals(self, train_df:pd.DataFrame, eval_dfs:dict):
        train_individuals = set(train_df[self.individual_name_column])
        eval_individuals = set()
        for name, eval_df in eval_dfs.items():
            if name != self.dev_set_column_name: # Exclude dev_set from these statistics
                eval_individuals.update(set(eval_df[self.individual_name_column]))
        all_individuals = train_individuals.union(eval_individuals)
        no_train_individuals = eval_individuals - train_individuals
        no_eval_individuals = train_individuals - eval_individuals

        return train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals

    def get_individuals_labels(self, all_individuals:set):
        sorted_indvidiuals = sorted(list(all_individuals)) # Sort individuals for ensuring same order every time (required for automatic loading)
        label_to_name = {idx:name for idx, name in enumerate(sorted_indvidiuals)}
        name_to_label = {name:idx for idx, name in label_to_name.items()}
        num_labels = len(name_to_label)

        return label_to_name, name_to_label, num_labels

    def show_data_stats(self, train_df:pd.DataFrame, eval_dfs:dict, no_eval_individuals:set, no_train_individuals:set, eval_individuals:set):
        logging.info(f"Number of background knowledge documents for training: {len(train_df)}")

        eval_n_dict = {name:len(df) for name, df in eval_dfs.items()}
        logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")

        if len(no_eval_individuals) > 0:
            logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
        
        if len(no_train_individuals) > 0:
            max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
            logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. Re-identification risk limited to {max_risk:.3f}% (excluding dev set).")

    #endregion

    #region ########## Data pretreatment ##########

    def load_spacy_nlp(self):
        # Load if it is not already loaded
        if self.spacy_nlp is None:
            self.spacy_nlp = en_core_web_lg.load()
        return self.spacy_nlp

    #region ##### Anonymize background knowledge #####
    
    def anonymize_bk(self, train_df:pd.DataFrame) -> pd.DataFrame:
        # Perform anonymization
        spacy_nlp = self.load_spacy_nlp()        
        train_anon_df = self.anonymize_df(train_df, spacy_nlp)

        if self.only_use_anonymized_background_knowledge:
            train_df = train_anon_df # Overwrite train dataframe with the anonymized version
        else:
            train_df = pd.concat([train_df, train_anon_df], ignore_index=True, copy=False) # Concatenate to train dataframe

        return train_df

    def anonymize_df(self, df, spacy_nlp, gc_freq=5) -> pd.DataFrame:
        assert len(df.columns) == 2 # Columns expected: name and text

        # Copy
        anonymized_df = df.copy(deep=True)

        # Process the text column
        column_name = anonymized_df.columns[1]
        texts = anonymized_df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Anonymizing {column_name} documents")):
            new_text = text

            # Anonymize by NER
            doc = spacy_nlp(text) # Usage of spaCy NER (https://spacy.io/api/entityrecognizer)
            for e in reversed(doc.ents): # Reversed to not modify the offsets of other entities when substituting
                start = e.start_char
                end = start + len(e.text)
                new_text = new_text[:start] + e.label_ + new_text[end:]

            # Remove doc and (periodically) use GarbageCollector to reduce memory consumption
            del doc
            if i % gc_freq == 0:
                gc.collect()

            # Assign new text
            texts[i] = new_text

        return anonymized_df

    #endregion

    #region ##### Document curation #####

    def document_curation(self, train_df:pd.DataFrame, eval_dfs:dict):
        spacy_nlp = self.load_spacy_nlp()

        # Perform preprocessing for both training and evaluation
        self.curate_df(train_df, spacy_nlp)
        for eval_df in eval_dfs.values():
            self.curate_df(eval_df, spacy_nlp)

    def curate_df(self, df, spacy_nlp, gc_freq=5):
        assert len(df.columns) == 2 # Columns expected: name and text

        # Predefined patterns
        special_characters_pattern = re.compile(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ./]+")
        stopwords = spacy_nlp.Defaults.stop_words

        # Process the text column (discarting the first one, that is the name column)
        column_name = df.columns[1]
        texts = df[column_name]
        for i, text in enumerate(tqdm(texts, desc=f"Preprocessing {column_name} documents")):
            doc = spacy_nlp(text) # Usage of spaCy (https://spacy.io/)
            new_text = ""   # Start text string
            for token in doc:
                if token.text not in stopwords:
                    # Lemmatize
                    token_text = token.lemma_ if token.lemma_ != "" else token.text
                    # Remove special characters
                    token_text = re.sub(special_characters_pattern, '', token_text)
                    # Add to new text (without space if dot)
                    new_text += ("" if token_text == "." else " ") + token_text

            # Remove doc and (periodically) use force GarbageCollector to reduce memory consumption
            del doc
            if i % gc_freq == 0:
                gc.collect()

            # Store result
            texts[i] = new_text

    #endregion

    #region ##### Save pretreatment #####

    def save_pretreatment_dfs(self, train_df:pd.DataFrame, eval_dfs:dict):
        with open(self.pretreated_data_path, "w") as f:
            f.write(json.dumps((train_df.to_json(orient="records"),
                                {name:df.to_json(orient="records") for name, df in eval_dfs.items()})))        

    #endregion

    #endregion

    #endregion

#endregion

#region ###################################### TRI dataset ######################################

class TRIDataset(Dataset):
    def __init__(self, df, tokenizer, name_to_label, return_labels, sliding_window_config, tokenization_block_size):
        # Dataframe must have two columns: name and text
        assert len(df.columns) == 2
        self.df = df

        # Set general attributes
        self.tokenizer = tokenizer
        self.name_to_label = name_to_label
        self.return_labels = return_labels

        # Set sliding window
        self.sliding_window_config = sliding_window_config
        try:
            sw_elems = [int(x) for x in sliding_window_config.split("-")]
            self.sliding_window_length = sw_elems[0]
            self.sliding_window_overlap = sw_elems[1]
            self.use_sliding_window = True
        except:
            self.use_sliding_window = False # If no sliding window (e.g., "No"), use sentence splitting

        if self.use_sliding_window and self.sliding_window_length > self.tokenizer.model_max_length:
            logging.exception(f"Sliding window length ({self.sliding_window_length}) must be lower than the maximum sequence length ({self.tokenizer.model_max_length})")     

        self.tokenization_block_size = tokenization_block_size

        # Compute inputs and labels
        self.generate()

    def generate(self, gc_freq=5):
        texts_column = list(self.df[self.df.columns[1]])
        names_column = list(self.df[self.df.columns[0]])
        labels_idxs = list(map(lambda x: self.name_to_label[x], names_column))   # Compute labels, translated to the identity index
        
        # Sliding window
        if self.use_sliding_window:
            texts = texts_column            
            labels = labels_idxs
        # Sentence splitting
        else:
            texts = []
            labels = []

            # Load spacy model for sentence splitting
            # Create spaCy model. Compontents = tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner
            # disable = ["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"]) # Required components: "senter" and "parser"
            spacy_nlp = en_core_web_lg.load(disable = ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])
            spacy_nlp.add_pipe('sentencizer')

            # Get texts and labels per sentence
            for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)), total=len(texts_column),
                                                    desc="Processing sentence splitting"):
                for paragraph in text.split("\n"):
                    if len(paragraph.strip()) > 0:
                        doc = spacy_nlp(paragraph)
                        for sentence in doc.sents:
                            # Parse sentence to text
                            sentence_txt = ""
                            for token in sentence:
                                sentence_txt += " " + token.text
                            sentence_txt = sentence_txt[1:] # Remove initial space
                            # Ensure length is less than the maximum
                            sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                            if sent_token_count > self.tokenizer.model_max_length:
                                logging.exception(f"ERROR: Sentence with length {sent_token_count} > {self.tokenizer.model_max_length} at index {idx} with label {label} not included because is too long | {sentence_txt}")
                            else:
                                # Store sample
                                texts.append(sentence_txt)
                                labels.append(label)
                    
                        # Delete document for reducing memory consumption
                        del doc
                    
                # Periodically force GarbageCollector for reducing memory consumption
                if idx % gc_freq == 0:
                    gc.collect()
                
        # Tokenize texts
        self.inputs, self.labels = self.tokenize_data(texts, labels)        

    def tokenize_data(self, texts, labels):
        # Sliding window
        if self.use_sliding_window:
            input_length = self.sliding_window_length
            padding_strategy = "longest"
        # Sentence splitting
        else:
            input_length = self.tokenizer.model_max_length            
            padding_strategy = "max_length"

        all_input_ids = torch.zeros((0, input_length), dtype=torch.int)
        all_attention_masks = torch.zeros((0, input_length), dtype=torch.int)
        all_labels = []

        # For each block of data
        with tqdm(total=len(texts)) as pbar:
            for ini in range(0, len(texts), self.tokenization_block_size):
                end = min(ini+self.tokenization_block_size, len(texts))
                pbar.set_description("Tokenizing (progress bar frozen)")
                block_inputs = self.tokenizer(texts[ini:end],
                                            add_special_tokens=not self.use_sliding_window,
                                            padding=padding_strategy,  # Warning: If an text is longer than tokenizer.model_max_length, an error will raise on prediction
                                            truncation=False,
                                            max_length=self.tokenizer.model_max_length,
                                            return_tensors="pt")
                
                # Force GarbageCollector after tokenization
                gc.collect()

                # Sliding window
                if self.use_sliding_window:                    
                    all_input_ids, all_attention_masks, all_labels = self.do_sliding_window(labels[ini:end], input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs)
                # Sentence splitting
                else:
                    # Concatenate to all data            
                    all_input_ids = torch.cat((all_input_ids, block_inputs["input_ids"]))
                    all_attention_masks = torch.cat((all_attention_masks, block_inputs["attention_mask"]))
                    all_labels = labels
                    pbar.update(len(block_inputs))

        # Get inputs
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_masks}

        # Transform labels to tensor
        labels = torch.tensor(all_labels)

        return inputs, labels

    def do_sliding_window(self, block_labels, input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs):
        # Predict number of windows
        n_windows = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        window_increment = self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
        for old_attention_mask in block_inputs["attention_mask"]:
            is_sequence_finished = False
            is_padding_required = False
            ini = 0
            end = ini + self.sliding_window_length - 2
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_mask = old_attention_mask[ini:end]
                            
                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Increment indexes
                ini += window_increment
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens

                n_windows += 1
                    
        # Allocate memory for ids and masks
        all_sequences_windows_ids = torch.empty((n_windows, input_length), dtype=torch.int)
        all_sequences_windows_masks = torch.empty((n_windows, input_length), dtype=torch.int)                                   

        # Sliding window for block sequences' splitting
        window_idx = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        pbar.set_description("Processing sliding window")
        for block_seq_idx, (old_input_ids, old_attention_mask) in enumerate(zip(block_inputs["input_ids"], block_inputs["attention_mask"])):
            ini = 0
            end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
            is_sequence_finished = False
            is_padding_required = False
            n_windows_in_seq = 0
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_ids = old_input_ids[ini:end]
                window_mask = old_attention_mask[ini:end]

                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Add CLS and SEP tokens
                num_attention_tokens = torch.count_nonzero(window_mask)
                if num_attention_tokens == window_mask.size()[0]:  # If window is full
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.sep_token_id]) ))
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([1]) )) # Attention to CLS and SEP
                else: # If window has empty space (to be padded later)
                    window_ids[num_attention_tokens] = torch.tensor(self.tokenizer.sep_token_id) # SEP at last position
                    window_mask[num_attention_tokens] = 1 # Attention to SEP
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.pad_token_id]) )) # PAD at the end of sentence
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([0]) )) # No attention to PAD

                # Padding if it is required
                if is_padding_required:
                    padding_length = self.sliding_window_length - window_ids.size()[0]
                    padding = torch.zeros((padding_length), dtype=window_ids.dtype)
                    window_ids = torch.cat((window_ids, padding))
                    window_mask = torch.cat((window_mask, padding))

                # Store ids and mask
                all_sequences_windows_ids[window_idx] = window_ids
                all_sequences_windows_masks[window_idx] = window_mask

                # Increment indexes
                ini += self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
                n_windows_in_seq += 1
                window_idx += 1
                        
            # Stack lists and concatenate with new data
            all_labels += [block_labels[block_seq_idx]] * n_windows_in_seq
            pbar.update(1)
                    
        # Concat the block data        
        all_input_ids = torch.cat((all_input_ids, all_sequences_windows_ids))
        all_attention_masks = torch.cat((all_attention_masks, all_sequences_windows_masks))

        # Force GarbageCollector after sliding window
        gc.collect()

        return all_input_ids, all_attention_masks, all_labels
    
    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, index):
        # Get each value (tokens, attention...) of the item
        input = {key: value[index] for key, value in self.inputs.items()}

        # Get label if is required
        if self.return_labels:
            label = self.labels[index]
            input["labels"] = label
        
        return input

#endregion

#region ###################################### Main CLI ######################################
if __name__ == "__main__":
    # Load configuration
    logging.info("######### START: CONFIGURATION #########")
    target_dir = argument_parsing()
    config = get_config_from_file(target_dir)
    tri = TRI(**config)
    logging.info("######### END: CONFIGURATION #########")
    
    # Run all sections
    tri.run_data(verbose=True)
#endregion
