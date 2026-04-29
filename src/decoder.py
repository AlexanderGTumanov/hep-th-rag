import numpy as np
import datetime
import warnings
import random
import math
import json
import sys
import re
import os
import unicodedata
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer, BartForConditionalGeneration

warnings.filterwarnings("ignore", message = "The PyTorch API of nested tensors is in prototype stage")

_LAST_PROGRESS_MESSAGE_LEN = 0
_TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base")
_PAD_TOKEN_ID = _TOKENIZER.pad_token_id

def load_chunks(chunks_path = "../data/corpus/chunks.jsonl"):
    records = []
    with open(chunks_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

class ArxivSummaryDataset(Dataset):
    def __init__(
        self,
        max_target_tokens = 512,
        corpus_dir = "../data/corpus"
    ):
        self.tokenizer = _TOKENIZER
        self.max_target_tokens = max_target_tokens
        chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
        abstracts_path = os.path.join(corpus_dir, "abstracts.jsonl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"chunks.jsonl not found in {corpus_dir}.")
        if not os.path.exists(abstracts_path):
            raise FileNotFoundError(f"abstracts.jsonl not found in {corpus_dir}.")
        all_chunks = load_chunks(chunks_path)
        all_abstracts = load_chunks(abstracts_path)
        abstract_map = {rec["doc_id"]: rec["text"] for rec in all_abstracts if rec.get("doc_id") and rec.get("text")}
        chunks_by_doc = {}
        for rec in all_chunks:
            doc_id = rec.get("doc_id")
            text = rec.get("text", "").strip()
            if doc_id and text:
                chunks_by_doc.setdefault(doc_id, []).append(text)
        self.pairs = [(doc_id, chunks) for doc_id, chunks in chunks_by_doc.items() if doc_id in abstract_map and len(chunks) >= 3]
        self.abstract_map = abstract_map
        self.all_chunks_flat = [text for chunks in chunks_by_doc.values() for text in chunks]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        doc_id, chunks = self.pairs[i]
        first = chunks[0]
        rest = random.sample(chunks[1:], k = 2)
        selected = [first] + rest
        random.shuffle(selected)
        input_text = " </s> ".join(selected)
        target_text = self.abstract_map[doc_id]
        input_enc = self.tokenizer(
            input_text,
            max_length = 1024,
            truncation = True,
            return_tensors = "pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length = self.max_target_tokens,
            truncation = True,
            return_tensors = "pt",
        )
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0),
            "doc_id": doc_id,
        }
    
def prepare_dataloaders(batch_size = 4, valid_split = 0.2, max_target_tokens = 512, seed = 42, corpus_dir = "../data/corpus"):
    dataset = ArxivSummaryDataset(max_target_tokens = max_target_tokens, corpus_dir = corpus_dir)
    n = len(dataset)
    valid_len = int(n * valid_split)
    train_len = n - valid_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(dataset, [train_len, valid_len], generator = generator)

    def collate_batch(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first = True, padding_value = _PAD_TOKEN_ID)
        attention_mask = torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first = True, padding_value = 0)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in batch], batch_first = True, padding_value = -100)
        pad_len = max_target_tokens - labels.shape[1]
        if pad_len > 0:
            labels = torch.nn.functional.pad(labels, (0, pad_len), value = -100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = collate_batch, drop_last = True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_batch, drop_last = True)
    return train_loader, valid_loader

def prepare_model(model_name = "facebook/bart-base"):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_bos_token_id = None
    model.config.tie_word_embeddings = False
    model = model.to(torch.float32).to(device)
    return model

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model = torch.load(model_path, map_location = "cpu", weights_only = False)
    return model

def show_progress(bi, total_batches, epoch = None, grad_norm = None):
    global _LAST_PROGRESS_MESSAGE_LEN
    pct = 100.0 * bi / max(1, total_batches)
    if epoch is None:
        base = f"progress: {pct:6.2f}%"
    else:
        base = f"epoch {epoch}: {pct:6.2f}%"
    if grad_norm is not None:
        GRAD_COL = 30
        spaces = " " * max(1, GRAD_COL - len(base))
        msg = f"{base}{spaces}gradient norm: {grad_norm:.4g}"
    else:
        msg = base
    padding = " " * max(0, _LAST_PROGRESS_MESSAGE_LEN - len(msg))
    sys.stdout.write("\r" + msg + padding)
    sys.stdout.flush()
    _LAST_PROGRESS_MESSAGE_LEN = len(msg)

def train_model(
    model,
    train_loader,
    valid_loader,
    epochs,
    batches = 0,
    start_batch = 0,
    model_dir = "../decoder",
    dropout = 0.1,
    lr = 1e-5,
    max_grad_norm = 1,
    clip_start_batch = None
):
    global _LAST_PROGRESS_MESSAGE_LEN
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    os.makedirs(model_dir, exist_ok = True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout
    total_train_batches = len(train_loader)
    total_valid_batches = len(valid_loader)
    if total_valid_batches > 0:
        valid_every = max(1, int(round(total_train_batches / total_valid_batches)))
    else:
        valid_every = None
    if batches > 0:
        if start_batch + batches > total_train_batches:
            raise ValueError(f"start_batch + batches must not exceed epoch length ({total_train_batches})")
    total_epochs_to_run = epochs + (1 if batches > 0 else 0)
    if total_epochs_to_run == 0:
        return model
    for epoch in range(1, total_epochs_to_run + 1):
        _LAST_PROGRESS_MESSAGE_LEN = 0
        model.train()
        train_batch_losses = []
        valid_batch_losses = []
        clipped_batches = []
        valid_iter = iter(valid_loader) if valid_every is not None else None
        epoch_batches = total_train_batches
        is_partial_epoch = (batches > 0 and epoch == total_epochs_to_run and epoch > epochs)
        if is_partial_epoch:
            epoch_batches = batches
        show_progress(0, epoch_batches, epoch = epoch)
        for bi, batch in enumerate(train_loader, start = 1):
            if is_partial_epoch:
                if bi <= start_batch:
                    continue
                if bi > start_batch + epoch_batches:
                    break
                pbi = bi - start_batch
            else:
                if bi > epoch_batches:
                    break
                pbi = bi
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = output.loss
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            clip = False
            if epoch == 1:
                if clip_start_batch is None:
                    clip = True
                elif bi >= clip_start_batch:
                    clip = True
            if clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                if grad_norm_value > max_grad_norm:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    doc_ids = batch.get("doc_id", [])
                    if isinstance(doc_ids, torch.Tensor):
                        doc_ids = doc_ids.tolist()
                    clipped_batches.append({
                        "epoch": epoch,
                        "batch_index": bi,
                        "loss": loss_value,
                        "grad_norm": grad_norm_value,
                        "doc_ids": doc_ids,
                    })
                    print(f"gradient clipped: epoch {epoch}, batch {bi}: grad_norm = {grad_norm_value:.4g}")
            optimizer.step()
            if device.type == "mps":
                torch.mps.empty_cache()
            train_batch_losses.append(loss_value)
            valid_batch_losses.append(float("nan"))
            if valid_iter is not None and (bi % valid_every == 0):
                with torch.no_grad():
                    try:
                        vbatch = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(valid_loader)
                        vbatch = next(valid_iter)
                    vinput_ids = vbatch["input_ids"].to(device)
                    vattention_mask = vbatch["attention_mask"].to(device)
                    vlabels = vbatch["labels"].to(device)
                    voutput = model(input_ids = vinput_ids, attention_mask = vattention_mask, labels = vlabels)
                    vloss = voutput.loss
                valid_batch_losses[-1] = float(vloss.item())
            show_progress(pbi, epoch_batches, epoch = epoch, grad_norm = min(grad_norm_value, max_grad_norm) if clip else None)
        sys.stdout.write("\n")
        sys.stdout.flush()
        history = {"train": train_batch_losses, "valid": valid_batch_losses}
        if is_partial_epoch:
            tag = f"e{epochs}-b{batches}"
        else:
            tag = f"e{epoch}"
        epoch_dir = os.path.join(model_dir, f"{run_id}-{tag}")
        os.makedirs(epoch_dir, exist_ok = True)
        torch.save(model, os.path.join(epoch_dir, f"model-{run_id}-{tag}.pt"))
        torch.save(history, os.path.join(epoch_dir, f"history-{run_id}-{tag}.pt"))
        bad_path = os.path.join(epoch_dir, f"bad_batches-{run_id}-{tag}.json")
        with open(bad_path, "w", encoding = "utf-8") as f:
            json.dump(clipped_batches, f, ensure_ascii = False, indent = 2)
    return model

def summarize(chunk_ids, model, corpus_dir = "../data/corpus"):
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    chunk_map = {}
    with open(chunks_path, "r", encoding = "utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("chunk_id") in chunk_ids:
                chunk_map[rec["chunk_id"]] = rec["text"]
    selected = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
    input_text = " </s> ".join(selected)
    input_enc = _TOKENIZER(input_text, max_length = 1024, truncation = True, return_tensors = "pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_enc["input_ids"], attention_mask = input_enc["attention_mask"], max_new_tokens = 256, num_beams = 4)
    return _TOKENIZER.decode(output_ids[0], skip_special_tokens = True)