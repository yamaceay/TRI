# RAG Identifier

*By [Yamac Ay](mailto:yamacerenay2001@gmail.com)*

This document outlines the structure and purpose of using a RAG-based text re-identifier model as an alternative to using fixed set of identifiers.

## Status Quo

In a text re-identification task, we use a list of documents for training, each associated with a unique sensitive entity. The current re-identifier attacker located in `tri.py` is designed to work with a fixed number of identifiers, e.g. 553 sensitive entities in total. Also, the model is trained to answer the question "Which of these 553 sensitive entities is this document associated with?". 

## Problem At Hand

Although the names are not fixed at the beginning, the names has to be provided in the exact order as they appear in the training set. This might not be practical in real-world scenarios where the number of sensitive entities vary significantly, i.e. new entities are added, existing entities are updated, and some entities are removed. Besides, the identifiers may not be seen beforehand.

## Proposed Solution

To address this issue, the new RAG-based text re-identifier model is proposed, located in `rag.py`. Instead of using indices to denote the sensitive entities directly, the model performs sentence-similarity search to find the most similar sensitive entity to the given document.

This is done by fine-tuning a pre-trained sentence transformer model on the positive pairs of documents and entities. The vector data is stored in a FAISS index, which allows for efficient similarity search. For training, we use the trainer API of the `sentence-transformers` library, and evaluation is achieved using Information Retrieval metrics such as Mean Reciprocal Rank (MRR) and Recall@k. The model uses Cosine Similarity loss function, which seems to find a great balance between learning and preventing representation collapse, thus allowing the model to learn meaningful representations.

## Preliminary Results

The preliminary results show that the RAG-based text re-identifier model can achieve great performance on the text re-identification task. For example, picking the Wiki People dataset (located in `data/WikiPeople_*`), the model can achieve nearly perfect scores on the training and evaluation set after 3 epochs of training. For training on your own, follow the instructions below:

## Setup

```bash
# setup the virtual environment
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# run the toy script
python3 rag.py
```

## Critical Points

Although the claims are promising in the direction of a generalizable text re-identifier, there are some critical points to consider:
- The entities are special in the sense that they don't usually convey much information about the sensitive entity itself. Therefore, the model memorizes them and learns to couple them with their associated documents during training. 
- However, this might not be the case if the entities are in fact informative, e.g. names of famous people or organizations. In such cases, the model might be able to generalize and perform well even if the entities are not seen during training.