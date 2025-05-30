import json
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import torch
from torch.utils.data import DataLoader
from io import StringIO
import pandas as pd
from collections import OrderedDict
import faiss

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

device = (
    torch.device("cuda") if torch.cuda.is_available() else 
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

def load_pretreated_pairs(json_path, max_samples=None, with_eval=False):
    with open(json_path, "r") as f:
        (train_df_json_str, eval_dfs_jsons) = json.load(f)        
        
    train_list = json.loads(StringIO(train_df_json_str).read())
    pairs = []
    for row in train_list:
        name = row.pop('name')
        public_knowledge = row.pop('public_knowledge')
        pairs.append(InputExample(texts=[name, public_knowledge], label=1.0))
    
    eval_pairs_dict = {}
    if with_eval:
        eval_dict = OrderedDict([(name, json.loads(StringIO(df_json).read())) for name, df_json in eval_dfs_jsons.items()])
        for key, eval_list in eval_dict.items():
            for row in eval_list:
                name = row.pop('name')
                anonymized_knowledge = row.pop(key)
                if key not in eval_pairs_dict:
                    eval_pairs_dict[key] = []
                eval_pairs_dict[key].append(InputExample(texts=[name, anonymized_knowledge], label=1.0))

    return pairs, eval_pairs_dict if with_eval else pairs

def load_pairs_actors(jsonl_path, max_samples=None, with_eval=False):
    pairs = []
    eval_pairs_dict = {} if with_eval else None
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        objs = json.load(f)
        for obj in objs:
            name = obj.pop('name')
            public_knowledge = obj.pop('public_knowledge')
            if name is None or public_knowledge is None:
                logging.warning(f"Skipping entry with missing name or public_knowledge")
                continue
            pairs.append(InputExample(texts=[name, public_knowledge], label=1.0))
            if with_eval:
                if not len(eval_pairs_dict):
                    eval_pairs_dict = {key: [] for key in obj}
                if any(value is None for value in obj.values()):
                    logging.warning(f"Skipping entry with missing evaluation data for {name}")
                    continue
                for key, value in obj.items():
                    eval_pairs_dict[key].append(InputExample(texts=[name, value], label=1.0))
    return pairs, eval_pairs_dict if with_eval else pairs

def load_pairs_wiki(jsonl_path, max_samples=None):
    pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            pairs.append(InputExample(texts=[obj['people'], obj['text']], label=1.0))
    return pairs

def evaluate_model(model, pairs):
    queries = {f'q{i}': ex.texts[0] for i, ex in enumerate(pairs)}
    corpus = {f'd{i}': ex.texts[1] for i, ex in enumerate(pairs)}
    relevant_docs = {f'q{i}': [f'd{i}'] for i in range(len(pairs))}
    evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs)
    evaluation_results = model.evaluate(evaluator)
    return evaluation_results

class Retriever:
    def __init__(self, model, docs):
        self.model = model
        self.docs = docs
        doc_emb = self.model.encode(self.docs, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(doc_emb.shape[1])
        self.index.add(doc_emb)

    def retrieve(self, query, k=5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        dists, idxs = self.index.search(query_emb, k)
        return [(self.docs[i], dists[0][rank]) for rank, i in enumerate(idxs[0])]

files = {
    "outputs/WikiActors/50_eval_25%BK/Pretreated_Data.json": ("ACTORS_WIKI_PRETREATED", None),
    'data/WikiActors_500_filtered.json': ("ACTORS_WIKI", None),
    'data/WikiPeople_250_train.jsonl': ("PEOPLE_WIKI", 'data/WikiPeople_250_eval.jsonl'),
}

if __name__ == "__main__":
    # Define different modes
    FILE = 'data/WikiActors_500_filtered.json'
    MODE, TEST_FILE = files[FILE]  # Can be "ACTORS_WIKI_PRETREATED", "ACTORS_WIKI", or "PEOPLE_WIKI")

    # Load data based on mode
    if MODE == "ACTORS_WIKI_PRETREATED":
        train_pairs, eval_pairs_dict = load_pretreated_pairs(FILE, with_eval=True)
    elif MODE == "ACTORS_WIKI":
        train_pairs, eval_pairs_dict = load_pairs_actors(FILE, with_eval=True)
    else:  # PEOPLE_WIKI
        train_pairs = load_pairs_wiki(FILE)
        eval_pairs = load_pairs_wiki(TEST_FILE)

    # Process training documents
    train_docs = [ex.texts[1] for ex in train_pairs]

    # Process evaluation documents based on mode
    if MODE in ["ACTORS_WIKI_PRETREATED", "ACTORS_WIKI"]:
        eval_docs = {column: [ex.texts[1] for ex in eval_pairs] for column, eval_pairs in eval_pairs_dict.items()}
    else:  # PEOPLE_WIKI
        eval_docs = [ex.texts[1] for ex in eval_pairs]

    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name, device=device)

    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True
    )

    retriever = Retriever(model, train_docs)

    # Evaluate the model performance
    train_results = evaluate_model(model, train_pairs)

    # Handle evaluation results based on mode
    if MODE in ["ACTORS_WIKI_PRETREATED", "ACTORS_WIKI"]:
        eval_results = {key: evaluate_model(model, eval_pairs) for key, eval_pairs in eval_pairs_dict.items()}
    else:  # PEOPLE_WIKI
        eval_results = evaluate_model(model, eval_pairs)

    # Print training results
    for key, value in train_results.items():
        print(f"train_{key}: {value:.4f}")

    # Print evaluation results based on mode
    if MODE in ["ACTORS_WIKI_PRETREATED", "ACTORS_WIKI"]:
        for key, value in eval_results.items():
            for sub_key, sub_value in value.items():
                print(f"eval_{key}_{sub_key}: {sub_value:.4f}")
    else:  # PEOPLE_WIKI
        for key, value in eval_results.items():
            print(f"eval_{key}: {value:.4f}")

    # Save and load model based on mode
    if MODE in ["ACTORS_WIKI_PRETREATED", "ACTORS_WIKI"]:
        model.save(f"outputs/{model_name}-actors-wiki")
    else:  # PEOPLE_WIKI
        model.save(f"outputs/{model_name}-people-wiki")