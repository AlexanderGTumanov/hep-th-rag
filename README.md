# Dense Retrieval for hep-th arXiv

## Introduction

This project builds a semantic search engine for the high-energy theoretical physics (hep-th) arXiv. Instead of relying on titles, abstracts, or keyword matching, the system operates on the full text of papers. Each document is split into smaller text chunks, and a neural network encodes these chunks into dense vector representations. User queries are embedded in the same space, and the system retrieves the most semantically relevant passages in the corpus. The goal is to enable fast, meaning-aware search that can locate the exact sections where specific ideas or constructions are discussed.

The retrieval model is a Transformer-based encoder that maps token sequences to fixed-size embeddings. The model uses a 256-dimensional token representation, four Transformer layers, eight attention heads per layer, and feed-forward blocks of size 640 with a dropout rate of 0.1, and produces a 256-dimensional output embedding.

The dataset consists of five years of hep-th arXiv submissions from 2021 to 2025, including cross-listed papers, for a total of over 36,000 documents. After removing equations, citations, figures, and other LaTeX artifacts, the cleaned corpus is approximately 1.4 GB of text. The corpus was divided into about 830,000 chunks, and a custom vocabulary of roughly 48,000 tokens was built from this data.

The model was trained for three epochs. The parameters were chosen to yield an approximate training time of 15 hours per epoch on a 32 GB Mac M1 system, which was used for this project.

---
 
## What it does

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

---

## Project structure

The project is organized into several main directories:

- `/notebooks` contains a Jupyter notebook that demonstrates the model’s architecture, training, and retrieval results.
- `/src` includes the core Python code:
  - `scraper.py` for downloading and preparing arXiv data.
  - `model.py` for the neural network and embedding logic.
- When running from scratch, the following will be created in the `/data` folder:
  - `/raw` contains `source.tar.gz` files for all downloaded articles.
  - `/processed` contains the processed versions of these articles, split into sections.
  - `/corpus` contains:
    - `chunks.jsonl` contains unencoded text chunks.
    - `chunk_embeddings.pt` contains encoded chunks.
    - `chunk_ids.json` contains chunk metadata for fast retrieval.
  - `metadata.jsonl` contains article metadata.
  - `/corpus_sample` is included in the GitHub version of the project and provides a small sample corpus for illustration.
- `/model` contains the pretrained model and related files:
  - `model.pt` with the pretrained model weights.
  - `vocab.jsonl` with the corresponding vocabulary.
  - `history.pt` with per-batch training and validation loss history.

---

## Contents of `scraper.py`

The `scraper.py` file contains the functions for retrieving articles, processing them, and building the corpus. Below is a breakdown of its core functionality.

#### `scrape_arxiv_sources(start, end, category = "hep-th", include_crosslists = True, request_delay = 3, max_retries = 3, timeout = 30, data_dir = "../data", overwrite = False)`

Retrieves all articles submitted between `start` and `end` in a given category and saves their source files to the `/raw` subfolder of the `/data` directory. Article metadata is appended to `metadata.jsonl` in the root of data. If `overwrite = False`, articles that already exist in `/raw` are skipped and not re-downloaded.

#### `load_metadata(metadata_path = "../data/metadata.jsonl")`

Loads metadata from the specified location.

#### `extract_macros(latex)`

Scans a LaTeX source chunk for custom macro definitions. Identifies four classes of macros: no-argument macros, argument macros, environment-defining macros, and delimited macros. Returns a dictionary of the form `{"noarg_mac": noarg_mac, "arg_mac": arg_mac, "env_mac": env_mac, "delim_mac": delim_mac}`, where each entry lists the macros of that type along with their definitions.

#### `replace_macros(latex, macros)`

Given a macros dictionary `macros` produced by `extract_macros`, replaces all instances of those macros in `latex` with their expanded standard LaTeX form.

#### `clean_body(body)`

Performs a range of processing steps on a body of LaTeX code to convert it into plain text. Replaces non-text environments with placeholders (`MATH`, `EQN`, `CITE`, `REF`, `FIG`, `URL`, `CODE`, `ALG`, `ENV`) to preserve grammatical structure. Cleans remaining LaTeX commands and text-based environment definitions while preserving their contents. Standardizes punctuation, latinizes, and lowercases the text.

#### `process_sources(data_dir = "../data", skip_threshold = 5, overwrite = False)`

Processes all papers in the `/raw` directory. First, custom macros are extracted and expanded. The function then attempts to remove highly technical parts, such as acknowledgments, bibliographies, and appendices. Abstracts are also removed, since they would otherwise dominate the training process. The remaining content is split into sections, and each section is processed with the `clean_body` function. Papers for which sectioning fails due to the authors not using `\section` or `\chapter` commands are processed as a single block of text. The `skip_threshold` parameter is used to detect papers for which the cleanup process has failed, usually due to overly complex macro setups. The function counts occurrences of command-specific characters (`_ ^ & \`) that remain after cleaning, and discards the paper if this number exceeds the threshold. With `overwrite = False`, already processed papers are skipped.

#### `build_corpus(data_dir = "../data", chunk_size = 1500, overlap = 150, placeholder_threshold = 0.15, min_tokens = 32, overwrite = False)`

Builds a corpus of text chunks from all papers in the `/processed` folder and saves it as `chunks.jsonl` in the `/corpus` directory inside `data`. Each section of each paper is subdivided into chunks of character length `chunk_size`. The `overlap` parameter creates a small overlap between neighboring chunks. Each chunk is assigned a unique ID. Although chunk length is specified in characters, the function avoids splitting individual words to keep tokenization consistent. As a result, chunk sizes vary slightly. With `overwrite = False`, the function does not run if the corpus already exists. With `overwrite = True`, the corpus is rebuilt from scratch.

---

## Contents of `model.py`

#### `train_model(model, train_loader, valid_loader, epochs, batches = 0, model_dir = "../model", dropout = 0.1, lr = 3e-4, tau = 0.05, max_grad_norm = 1.0, clip_start_batch = None)`

Trains `model` using an InfoNCE / NT-Xent–style contrastive objective in a dual-encoder setup. Training runs for the number of epochs specified by `epochs` and, optionally, for a fixed number of additional batches specified by `batches`. The `train_loader` and `valid_loader` provide the input data. The `dropout` and `lr` parameters control the dropout rate and learning rate, respectively.

The `tau` parameter sets the temperature of the similarity distribution used in the contrastive loss, controlling how sharply the model distinguishes between positive and negative pairs. The `max_grad_norm` parameter specifies the gradient clipping threshold applied during optimizer steps; gradients with a larger norm are scaled down to this value.

The `clip_start_batch` parameter allows gradient clipping to be disabled for the first few batches of the first epoch. Early training steps of a freshly initialized model often produce unusually large gradients due to random parameter initialization, and clipping during this phase can slow training. When continuing training from a previously trained model, it is recommended to set `clip_start_batch` to `None`.

The model saves a checkpoint in the `/model` folder at the end of each training epoch. Along with the checkpoint, it records the per-batch training and validation losses, as well as the contents of all batches for which gradient clipping occurred.



