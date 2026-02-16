import numpy as np
import datetime
import warnings
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

warnings.filterwarnings("ignore", message = "The PyTorch API of nested tensors is in prototype stage")

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]", re.VERBOSE)
_LAST_PROGRESS_MESSAGE_LEN = 0
_INDEX_CACHE = {}
_LATINIZIZER = str.maketrans({
    "ß": "ss", "ẞ": "ss",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
    "ø": "o",  "Ø": "o",
    "đ": "d",  "Đ": "d",
    "ł": "l",  "Ł": "l",
    "þ": "th", "Þ": "th",
    "ı": "i",
})

def latinize(text):
    text = text.translate(_LATINIZIZER)
    text = unicodedata.normalize("NFKD", text)
    combining = unicodedata.combining
    text = "".join(ch for ch in text if not combining(ch))
    return text.encode("ascii", "ignore").decode("ascii")

@dataclass
class Vocab:
    token_to_id: dict
    id_to_token: list
    pad_token = "<pad>"
    unk_token = "<unk>"

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self):
        return self.token_to_id[self.unk_token]

    def encode(self, tokens):
        ids = []
        t2i = self.token_to_id
        unk = self.unk_id
        for t in tokens:
            ids.append(t2i.get(t, unk))
        return ids

    def decode(self, ids, skip_special = True):
        specials = {self.pad_token, self.unk_token}
        out = []
        itok = self.id_to_token
        for i in ids:
            if 0 <= i < len(itok):
                t = itok[i]
            else:
                t = self.unk_token
            if skip_special and t in specials:
                continue
            out.append(t)
        return " ".join(out)

def build_vocab(chunks, min_freq = 5, max_vocab = None, vocab_path = "../model/vocab.jsonl", overwrite = False):
    counter = Counter()
    for chunk in chunks:
        counter.update(_WORD_RE.findall(chunk))
    special = ["<pad>", "<unk>"]
    items = [(tok, c) for tok, c in counter.items() if c >= min_freq]
    items.sort(key = lambda x: (-x[1], x[0]))
    tokens = special + [tok for tok, _ in items]
    if max_vocab is not None and len(tokens) > max_vocab:
        tokens = tokens[:max_vocab]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = list(tokens)
    vocab = Vocab(token_to_id, id_to_token)
    vocab_size = len(vocab.id_to_token)
    if vocab_path is None:
        print(f"Vocab size: {vocab_size}.")
        return vocab
    if not overwrite and os.path.exists(vocab_path):
        print(f"Vocab size: {vocab_size}. Vocab file already exists at {vocab_path}. Use overwrite = True to rebuild.")
        return vocab
    os.makedirs(os.path.dirname(vocab_path), exist_ok = True)
    with open(vocab_path, "w", encoding = "utf-8") as f:
        for t in id_to_token:
            f.write(json.dumps(t) + "\n")
    print(f"Vocab size: {vocab_size}. Vocab saved to {vocab_path}.")
    return vocab

def load_vocab(vocab_path = "../model/vocab.jsonl"):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab file not found: {vocab_path}")
    token_to_id = {}
    id_to_token = []
    with open(vocab_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = json.loads(line)
            token_to_id[tok] = len(id_to_token)
            id_to_token.append(tok)
    return Vocab(token_to_id, id_to_token)

def iterate_chunks(chunks_path = "../data/corpus/chunks.jsonl"):
    with open(chunks_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if text:
                yield text

def load_chunks(chunks_path = "../data/corpus/chunks.jsonl"):
    records = []
    with open(chunks_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

class ArxivChunkDataset(Dataset):
    def __init__(self, vocab, seq_len, corpus_dir = "../data/corpus", random_crop = False):
        self.vocab = vocab
        self.random_crop = random_crop
        chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"chunks.jsonl not found in {corpus_dir}.")
        self.chunks = load_chunks(chunks_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.chunks)

    def _pad_or_truncate(self, ids):
        L = self.seq_len
        if len(ids) > L:
            if self.random_crop:
                start = torch.randint(0, len(ids) - L + 1, (1,)).item()
                ids = ids[start:start + L]
            else:
                ids = ids[:L]
        mask = [1] * len(ids) # attention mask that ensures that tokens added through padding do not affect attention computations
        if len(ids) < L:
            padding = L - len(ids)
            ids = ids + [self.vocab.pad_id] * padding
            mask = mask + [0] * padding
        return torch.tensor(ids, dtype = torch.long), torch.tensor(mask, dtype = torch.long)

    def __getitem__(self, i):
        record = self.chunks[i]
        text = (record.get("text") or "").strip()
        tokens = _WORD_RE.findall(text)
        ids, mask = self._pad_or_truncate(self.vocab.encode(tokens))
        meta = {"chunk_id": record.get("chunk_id"), "doc_id": record.get("doc_id")}
        return {"input_ids": ids, "attention_mask": mask, "meta": meta}
    
def filter_indices(dataset, indices, vocab, unk_discard_ratio, pad_threshold_ratio):
    kept = []
    unk_id = vocab.unk_id
    for i in indices:
        sample = dataset[i]
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        n = int(mask.sum().item())
        if n == 0:
            continue
        L = ids.numel()
        pad_ratio = (L - n) / L
        if pad_ratio <= pad_threshold_ratio:
            unk_ratio = float((ids[:n] == unk_id).float().mean().item())
            if unk_ratio > unk_discard_ratio:
                continue
        kept.append(i)
    return kept

def collate_batch(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim = 0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim = 0)
    meta = [b["meta"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "meta": meta}

def prepare_dataloaders(
    vocab,
    seq_len,
    batch_size = 128,
    valid_split = 0.2,
    seed = 42,
    unk_discard_ratio = 0.5,
    pad_threshold_ratio = 0.7,
    corpus_dir = "../data/corpus"
):
    dataset_train = ArxivChunkDataset(vocab, seq_len, corpus_dir = corpus_dir, random_crop = True)
    dataset_valid = ArxivChunkDataset(vocab, seq_len, corpus_dir = corpus_dir, random_crop = False)
    n = len(dataset_train)
    valid_len = int(n * valid_split)
    train_len = n - valid_len
    generator = torch.Generator().manual_seed(seed)
    split_train, split_valid = random_split(range(n), [train_len, valid_len], generator = generator)
    train_indices = list(split_train.indices)
    valid_indices = list(split_valid.indices)
    train_indices = filter_indices(dataset_train, train_indices, vocab = vocab, unk_discard_ratio = unk_discard_ratio, pad_threshold_ratio = pad_threshold_ratio)
    valid_indices = filter_indices(dataset_valid, valid_indices, vocab = vocab, unk_discard_ratio = unk_discard_ratio, pad_threshold_ratio = pad_threshold_ratio)
    train_ds = torch.utils.data.Subset(dataset_train, train_indices)
    valid_ds = torch.utils.data.Subset(dataset_valid, valid_indices)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = (len(train_ds) > 0), num_workers = 0, collate_fn = collate_batch, drop_last = True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_batch, drop_last = True)
    return train_loader, valid_loader

def prepare_embedding_dataloader(vocab, seq_len, batch_size = 128, corpus_dir = "../data/corpus"):
    dataset = ArxivChunkDataset(vocab, seq_len, corpus_dir = corpus_dir, random_crop = False)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_batch, drop_last = False)
    return loader

class ChunkEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model = 256,
        n_heads = 8,
        n_layers = 4,
        d_ff = 1024,
        max_len = 512,
        dropout = 0.1,
        out_dim = 256,
        use_positional_encoding = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_positional_encoding = use_positional_encoding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_positional_encoding:
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = d_ff,
            dropout = dropout,
            batch_first = True,
            activation = "gelu",
            norm_first = False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.use_positional_encoding:
            pos = torch.arange(L, device = input_ids.device).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask = key_padding_mask)
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * mask).sum(dim = 1) / mask.sum(dim = 1).clamp_min(1.0)
        z = self.out_proj(pooled)
        z = F.normalize(z, p = 2, dim = -1)
        return z
    
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model = torch.load(model_path, map_location = "cpu", weights_only = False)
    return model

def show_progress(bi, total_batches, epoch=None, grad_norm = None):
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
    model_dir = "../model",
    dropout = 0.1,
    lr = 3e-4,
    tau = 0.05,
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
            z1 = model(input_ids, attention_mask)
            z2 = model(input_ids, attention_mask)
            logits = (z1 @ z2.t()) / tau
            labels = torch.arange(logits.size(0), device = logits.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
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
                    meta = batch.get("meta")
                    chunk_ids = []
                    doc_ids = []
                    if isinstance(meta, list):
                        for m in meta:
                            if isinstance(m, dict):
                                chunk_ids.append(m.get("chunk_id"))
                                doc_ids.append(m.get("doc_id"))
                    clipped_batches.append({
                        "epoch": epoch,
                        "batch_index": bi,
                        "loss": loss_value,
                        "grad_norm": grad_norm_value,
                        "chunk_ids": chunk_ids,
                        "doc_ids": doc_ids,
                    })
                    print(f"gradient clipped: epoch {epoch}, batch {bi}: grad_norm = {grad_norm_value:.4g}")
            optimizer.step()
            train_batch_losses.append(loss_value)
            valid_batch_losses.append(float("nan"))
            if valid_iter is not None and (bi % valid_every == 0):
                with torch.no_grad():
                    try:
                        vbatch = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(valid_loader)
                        vbatch = next(valid_iter)
                    vinp = vbatch["input_ids"].to(device)
                    vmask = vbatch["attention_mask"].to(device)
                    vz1 = model(vinp, vmask)
                    vz2 = model(vinp, vmask)
                    vlogits = (vz1 @ vz2.t()) / tau
                    vlabels = torch.arange(vlogits.size(0), device = vlogits.device)
                    vloss = 0.5 * (F.cross_entropy(vlogits, vlabels) + F.cross_entropy(vlogits.t(), vlabels))
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

def build_chunk_embeddings(model, loader, corpus_dir = "../data/corpus"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    out_emb_path = os.path.join(corpus_dir, "chunk_embeddings.pt")
    out_ids_path = os.path.join(corpus_dir, "chunk_ids.json")
    model = model.to(device)
    model.eval()
    all_z = []
    all_ids = []
    total_batches = len(loader)
    show_progress(0, total_batches)
    with torch.inference_mode():
        for bi, batch in enumerate(loader, start = 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            z = model(input_ids, attention_mask).cpu()
            all_z.append(z)
            meta = batch["meta"]
            for m in meta:
                all_ids.append(m.get("chunk_id"))
            show_progress(bi, total_batches)
    sys.stdout.write("\n")
    sys.stdout.flush()
    chunk_emb = torch.cat(all_z, dim = 0)
    torch.save(chunk_emb, out_emb_path)
    with open(out_ids_path, "w", encoding = "utf-8") as f:
        json.dump(all_ids, f)

def encode_prompt(prompt, vocab, seq_len):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    body = (prompt or "").strip()
    if any(ch in body for ch in ("_", "^", "$")):
        raise ValueError("Prompt should not include complicated math. Simple math like N = 4 can be written directly. Complex math should be avoided or replaced with MATH/EQN placeholders.")
    body = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]", "-", body)
    body = latinize(body.lower())
    tokens = _WORD_RE.findall(body)
    ids = vocab.encode(tokens)
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    mask = [1] * len(ids)
    if len(ids) < seq_len:
        pad = seq_len - len(ids)
        ids = ids + [vocab.pad_id] * pad
        mask = mask + [0] * pad
    input_ids = torch.tensor(ids, dtype = torch.long, device = device).unsqueeze(0)
    attention_mask = torch.tensor(mask, dtype = torch.long, device = device).unsqueeze(0)
    return input_ids, attention_mask

def prefilter_chunks(prompt, lexical_candidates = 1000, corpus_dir = "../data/corpus"):
    placeholders = {"MATH", "EQN", "NUM", "CITE", "REF", "FIG", "URL", "CODE", "ALG", "ENV"}
    ids_path = os.path.join(corpus_dir, "chunk_ids.json")
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    if not os.path.exists(ids_path):
        raise FileNotFoundError("chunk_ids.json not found in corpus_dir.")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("chunks.jsonl not found in corpus_dir.")
    ids_mtime = os.path.getmtime(ids_path)
    chunks_mtime = os.path.getmtime(chunks_path)
    key_meta = ("meta", chunks_path, chunks_mtime)
    if key_meta not in _INDEX_CACHE:
        with open(chunks_path, "r", encoding = "utf-8") as f:
            meta = {}
            for line in f:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid is None:
                    continue
                meta[cid] = {
                    "doc_id": rec.get("doc_id"),
                    "title": rec.get("title"),
                    "authors": rec.get("authors") or [],
                    "section": rec.get("section"),
                    "text": rec.get("text") or "",
                }
        _INDEX_CACHE[key_meta] = meta
    meta = _INDEX_CACHE[key_meta]
    key_ids = ("chunk_ids", ids_path, ids_mtime)
    if key_ids not in _INDEX_CACHE:
        with open(ids_path, "r", encoding = "utf-8") as f:
            chunk_ids = json.load(f)
        _INDEX_CACHE[key_ids] = chunk_ids
    chunk_ids = _INDEX_CACHE[key_ids]
    key_lex = ("tfidf-chunk", chunks_path, chunks_mtime, ids_path, ids_mtime, tuple(sorted(placeholders)))
    if key_lex not in _INDEX_CACHE:
        chunk_texts = []
        for cid in chunk_ids:
            rec = meta.get(cid, {})
            chunk_texts.append(rec.get("text") or "")
        vectorizer = TfidfVectorizer(tokenizer = lambda s: [t for t in _WORD_RE.findall(s or "") if t not in placeholders], token_pattern = None, lowercase = False, min_df = 1)
        X = vectorizer.fit_transform(chunk_texts)
        X = normalize(X, norm = "l2", copy = False)
        _INDEX_CACHE[key_lex] = (vectorizer, X)
    vectorizer, X = _INDEX_CACHE[key_lex]
    qx = vectorizer.transform([prompt])
    qx = normalize(qx, norm = "l2", copy = False)
    lex_scores = (X @ qx.T).toarray().ravel()
    M = min(int(lexical_candidates), int(lex_scores.shape[0]))
    if M <= 0:
        return []
    cand_pos = np.argpartition(-lex_scores, M - 1)[:M]
    cand_pos = cand_pos[np.argsort(-lex_scores[cand_pos])]
    return [chunk_ids[p] for p in cand_pos.tolist()]

def semantic_matches(prompt, model, vocab, seq_len, lexical_candidates = 1000, corpus_dir = "../data/corpus"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    emb_path = os.path.join(corpus_dir, "chunk_embeddings.pt")
    ids_path = os.path.join(corpus_dir, "chunk_ids.json")
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        raise FileNotFoundError("chunk embeddings not found; compute them first using the build_chunk_embeddings function.")
    emb_mtime = os.path.getmtime(emb_path)
    ids_mtime = os.path.getmtime(ids_path)
    key_dense = ("dense", emb_path, emb_mtime, ids_path, ids_mtime)
    if key_dense not in _INDEX_CACHE:
        chunk_emb = torch.load(emb_path, map_location = "cpu")
        with open(ids_path, "r", encoding = "utf-8") as f:
            chunk_ids = json.load(f)
        chunk_id_to_index = {cid: i for i, cid in enumerate(chunk_ids)}
        _INDEX_CACHE[key_dense] = (chunk_emb, chunk_id_to_index)
    chunk_emb, chunk_id_to_index = _INDEX_CACHE[key_dense]
    cand_chunk_ids = prefilter_chunks(prompt, corpus_dir = corpus_dir, lexical_candidates = lexical_candidates)
    idxs = [chunk_id_to_index[cid] for cid in cand_chunk_ids if cid in chunk_id_to_index]
    if not idxs:
        return []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = encode_prompt(prompt, vocab, seq_len)
        q = model(input_ids, attention_mask).squeeze(0).cpu()
    scores = (chunk_emb[idxs] @ q).tolist()
    hits = list(zip(scores, cand_chunk_ids))
    hits.sort(key = lambda x: x[0], reverse = True)
    return [(float(s), cid) for s, cid in hits]

def rerank_matches(matches, pool_level = "section", top_k_chunks = 1, top_k_sections = 1, corpus_dir = "../data/corpus"):
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("chunks.jsonl not found in corpus_dir.")
    chunks_mtime = os.path.getmtime(chunks_path)
    key_meta = ("meta", chunks_path, chunks_mtime)
    if key_meta not in _INDEX_CACHE:
        with open(chunks_path, "r", encoding = "utf-8") as f:
            meta = {}
            for line in f:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid is None:
                    continue
                meta[cid] = {
                    "doc_id": rec.get("doc_id"),
                    "title": rec.get("title"),
                    "authors": rec.get("authors") or [],
                    "section": rec.get("section"),
                }
        _INDEX_CACHE[key_meta] = meta
    meta = _INDEX_CACHE[key_meta]
    if pool_level not in ("paper", "section"):
        raise ValueError("pool_level must be 'paper' or 'section'.")
    kc = int(top_k_chunks)
    if kc <= 0:
        raise ValueError("top_k_chunks must be >= 1.")
    ks = int(top_k_sections)
    if ks <= 0:
        raise ValueError("top_k_sections must be >= 1.")
    per_doc = {}
    for s, cid in matches:
        m = meta.get(cid)
        if m is None:
            continue
        doc_id = m.get("doc_id")
        if doc_id is None:
            continue
        entry = per_doc.get(doc_id)
        if entry is None:
            entry = {
                "doc_id": doc_id,
                "title": m.get("title"),
                "authors": m.get("authors") or [],
                "n_chunks": 0,
                "scores": [],
                "sections": {},
            }
            per_doc[doc_id] = entry
        ss = float(s)
        entry["n_chunks"] += 1
        entry["scores"].append(ss)
        section = m.get("section")
        sec = entry["sections"].get(section)
        if sec is None:
            sec = {"section": section, "n_chunks": 0, "scores": []}
            entry["sections"][section] = sec
        sec["n_chunks"] += 1
        sec["scores"].append(ss)
    matches = []
    for doc_id, entry in per_doc.items():
        section_matches = []
        for sec_name, sec in entry["sections"].items():
            scores = sec.get("scores") or []
            if not scores:
                continue
            scores.sort(reverse = True)
            kk = min(kc, len(scores))
            sec_score = sum(scores[:kk]) / kk
            section_matches.append({
                "section": sec_name,
                "score": float(sec_score),
                "n_chunks": int(sec["n_chunks"]),
            })
        section_matches.sort(key = lambda x: x["score"], reverse = True)
        if pool_level == "paper":
            scores = entry.get("scores") or []
            if not scores:
                continue
            scores.sort(reverse = True)
            kk = min(kc, len(scores))
            doc_score = sum(scores[:kk]) / kk
        else:
            if not section_matches:
                continue
            kk = min(ks, len(section_matches))
            doc_score = sum(h["score"] for h in section_matches[:kk]) / kk
        matches.append({
            "score": float(doc_score),
            "doc_id": doc_id,
            "title": entry.get("title"),
            "authors": entry.get("authors") or [],
            "n_chunks": int(entry["n_chunks"]),
            "sections": section_matches,
        })
    matches.sort(key = lambda h: h["score"], reverse = True)
    return matches

def print_matches(matches, top_k = 10, max_sections = None, corpus_dir = "../data/corpus"):
    if not matches:
        return
    first = matches[0]
    
    if isinstance(first, tuple):
        chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError("chunks.jsonl not found in corpus_dir.")
        want = set([cid for _, cid in matches[:top_k]])
        found = {}
        with open(chunks_path, "r", encoding = "utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid in want:
                    found[cid] = rec
                    if len(found) == len(want):
                        break
        for rank, (score, cid) in enumerate(matches[:top_k], start = 1):
            rec = found.get(cid, {})
            title = rec.get("title")
            authors = rec.get("authors") or []
            section = rec.get("section")
            doc_id = rec.get("doc_id")
            print(f"{rank}. score = {float(score):.6f}")
            print(f"    title   : {title}")
            print(f"    authors : {', '.join(authors)}")
            print(f"    doc_id  : {doc_id}")
            if section:
                print(f"    section : {section}")
            print()
        return

    if isinstance(first, dict):
        for rank, h in enumerate(matches[:top_k], start = 1):
            print(f"{rank}. doc_score = {float(h.get('score')):.6f}")
            print(f"    title   : {h.get('title')}")
            print(f"    authors : {', '.join(h.get('authors') or [])}")
            print(f"    doc_id  : {h.get('doc_id')}")
            sections = h.get("sections") or []
            sections = [s for s in sections if s.get("section")]
            if max_sections is not None:
                sections = sections[:max_sections]
            if sections:
                print(f"    sections:")
                names = [s.get("section") for s in sections]
                max_len = max(len(n) for n in names)
                gap = 6
                width = max_len + gap
                for i, sec in enumerate(sections, start = 1):
                    name = sec.get("section")
                    score = float(sec.get("score"))
                    print(f"      {i}. {name:<{width}}score = {score:.6f}")
            print()
        return

    raise TypeError("please run on the output of semantic_matches or rerank_matches functions.")

def find_paper(matches, doc_ids, corpus_dir = "../data/corpus"):
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("chunks.jsonl not found in corpus_dir.")
    chunks_mtime = os.path.getmtime(chunks_path)
    key_c2d = ("chunk_to_doc", chunks_path, chunks_mtime)
    if key_c2d not in _INDEX_CACHE:
        c2d = {}
        with open(chunks_path, "r", encoding = "utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = rec.get("chunk_id")
                if cid is None:
                    continue
                c2d[cid] = rec.get("doc_id")
        _INDEX_CACHE[key_c2d] = c2d
    c2d = _INDEX_CACHE[key_c2d]
    out = {d: {"present": False, "best_rank": None, "best_score": None} for d in doc_ids}
    if matches:
        first = matches[0]
        if isinstance(first, tuple):
            for rank, (s, cid) in enumerate(matches, start = 1):
                d = c2d.get(cid)
                if d in out and not out[d]["present"]:
                    out[d]["present"] = True
                    out[d]["best_rank"] = rank
                    out[d]["best_score"] = float(s)
        elif isinstance(first, dict):
            for rank, h in enumerate(matches, start = 1):
                d = h.get("doc_id")
                if d in out and not out[d]["present"]:
                    out[d]["present"] = True
                    out[d]["best_rank"] = rank
                    out[d]["best_score"] = float(h.get("score"))
        else:
            raise TypeError("please run on the output of semantic_matches or rerank_matches functions.")
    for d in doc_ids:
        info = out[d]
        if info["present"]:
            print(f"{d}: present, rank = {info['best_rank']}, score = {float(info['best_score']):.6f}")
        else:
            print(f"{d}: not present")

def load_history(*history_paths):
    history = {"train": [], "valid": []}
    for path in history_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"history file not found: {path}")
        h = torch.load(path, map_location = "cpu")
        history["train"].extend(h.get("train", []))
        history["valid"].extend(h.get("valid", []))
    return history

def plot_history(history, log_x = False, log_y = False, batches_per_epoch = None):
    train = history["train"]
    valid = history["valid"]
    vx = [i for i, v in enumerate(valid) if np.isfinite(v)]
    vy = [v for v in valid if np.isfinite(v)] 
    plt.figure(figsize = (10, 5))
    plt.plot(train, label = "Train Loss")
    plt.plot(vx, vy, label = "Valid Loss")
    if batches_per_epoch is not None and batches_per_epoch > 0:
        total_batches = len(train)
        k = 1
        while True:
            x = k * batches_per_epoch
            if x > total_batches:
                break
            if not (log_x and x == 0):
                plt.axvline(x = x, linestyle = "--", linewidth = 1, alpha = 0.5)
            k += 1
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
