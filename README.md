# Dense Retrieval for hep-th arXiv

### Introduction

This project builds a semantic search engine for the high-energy theoretical physics (hep-th) arXiv. Instead of relying on titles, abstracts, or keyword matching, the system operates on the full text of papers. Each document is split into smaller text chunks, and a neural network encodes these chunks into dense vector representations. User queries are embedded in the same space, and the system retrieves the most semantically relevant passages in the corpus. The goal is to enable fast, meaning-aware search that can locate the exact sections where specific ideas or constructions are discussed.

The retrieval model is a Transformer-based encoder that maps token sequences to fixed-size embeddings. The model uses a 256-dimensional token representation, four Transformer layers, eight attention heads per layer, and feed-forward blocks of size 640 with a dropout rate of 0.1, and produces a 256-dimensional output embedding.

The dataset consists of five years of hep-th arXiv submissions from 2021 to 2025, including cross-listed papers, for a total of over 36,000 documents. After removing equations, citations, figures, and other LaTeX artifacts, the cleaned corpus is approximately 1.4 GB of text. The corpus was divided into about 830,000 chunks, and a custom vocabulary of roughly 48,000 tokens was built from this data.

The model was trained for three epochs. The parameters were chosen to yield an approximate training time of 15 hours per epoch on a 32 GB Mac M1 system, which was used for this project.

---

## Project Structure

The project is organized into several main directories:

- `/notebooks` contains a Jupyter notebook that demonstrates the model’s architecture, training, and retrieval results.
- `/src` includes the core Python code:
  - `scraper.py` for downloading and preparing arXiv data.
  - `model.py` for the neural network and embedding logic.
- `/data` contains example data:
  - `/corpus_sample` provides a small sample corpus to demonstrate the system.
- `/model` contains the pretrained model and related files:
  - `model.pt` with the pretrained model weights.
  - `vocab.jsonl` with the corresponding vocabulary.
  - `history.pt` with per-batch training and validation loss history.

---
 
## What It Does

- Data extraction and processing
  - Retrieves source files for arXiv articles in a given category (hep-th by default) over a specified date range.
  - Splits articles into sections and converts them into clean text by removing LaTeX artifacts, environments, and custom macros.
  - Builds a corpus by subdividing sections into overlapping chunks of a specified character length.
- Model training
  - Builds a word-level vocabulary from the corpus and encodes chunks as token IDs.
  - Pads or truncates chunks to a fixed sequence length and constructs train and validation dataloaders.
  - Trains a Transformer encoder with a contrastive objective in a dual-encoder setup and records per-batch training and validation loss.
- Retrieval functionality
  - Computes embeddings for all corpus chunks.
  - Encodes user queries and retrieves the most relevant chunks via fast similarity search.

