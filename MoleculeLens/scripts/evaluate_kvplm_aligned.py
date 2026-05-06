#!/usr/bin/env python3
"""Evaluate KV-PLM on the aligned MoleculeLens scaffold test split.

Outputs full-gallery retrieval metrics (text -> molecule) plus T-choose-one
reference scores for both with-drug and no-drug conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForPreTraining, BertTokenizer


class BigModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(
            tok, token_type_ids=typ, attention_mask=att
        )["pooler_output"]
        return self.dropout(pooled_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--with-out", required=True)
    parser.add_argument("--nodrug-out", required=True)
    parser.add_argument("--dataspace", default="/home/cheriearjun/MoleculeSTM/data")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--text-column", default="text_rich")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_drug_field(text: str) -> str:
    return re.sub(r"Drug:\s*[^.]+\.\s*", "", text).strip()


def load_aligned_pairs(
    source_csv: Path,
    split_manifest: Path,
    smiles_column: str,
    text_column: str,
    remove_drug_name: bool,
) -> tuple[list[str], list[str]]:
    df = pd.read_csv(source_csv).reset_index().rename(columns={"index": "source_row"})
    df = df.dropna(subset=[smiles_column, text_column]).copy()
    df[smiles_column] = df[smiles_column].astype(str)
    df[text_column] = df[text_column].astype(str)
    df = df[df[smiles_column].str.strip() != ""]
    df = df[df[text_column].str.strip() != ""]

    if remove_drug_name:
        df[text_column] = df[text_column].map(strip_drug_field)
        df = df[df[text_column].str.strip() != ""].reset_index(drop=True)

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    test_source_rows = manifest["splits"]["test"]
    source_to_pos = {int(src): pos for pos, src in enumerate(df["source_row"].tolist())}
    positions = [source_to_pos[int(src)] for src in test_source_rows]
    test_df = df.iloc[positions].reset_index(drop=True)
    return test_df[smiles_column].tolist(), test_df[text_column].tolist()


@torch.no_grad()
def encode_strings(strings, tokenizer, model, device, batch_size=32, max_seq_len=512):
    outputs = []
    for start in tqdm(range(0, len(strings), batch_size), desc="Encoding", leave=False):
        batch = strings[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        tok = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        outputs.append(model(tok, att, cud=torch.cuda.is_available()).cpu())
    return torch.cat(outputs, dim=0)


def recall_mrr(sim: np.ndarray) -> tuple[float, float]:
    n = sim.shape[0]
    top1 = sim.argmax(axis=1)
    recall1 = float((top1 == np.arange(n)).mean())
    reciprocal_ranks = []
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
    return recall1, float(np.mean(reciprocal_ranks))


def recall_at_k(sim: np.ndarray, ks=(5, 10)) -> dict[int, float]:
    n = sim.shape[0]
    topk = np.argsort(-sim, axis=1)
    return {k: float(sum(i in topk[i, :k] for i in range(n)) / n) for k in ks}


def t_choose_one(sim: np.ndarray, t_values=(4, 10, 20), seed=42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = sim.shape[0]
    all_indices = np.arange(n)
    metrics: dict[str, float] = {}
    for t_value in t_values:
        s2t_hits = 0
        t2s_hits = 0
        for index in range(n):
            negatives = rng.choice(all_indices[all_indices != index], size=t_value - 1, replace=False)
            candidates = np.concatenate([[index], negatives])
            if candidates[np.argmax(sim[index, candidates])] == index:
                s2t_hits += 1
            if candidates[np.argmax(sim[candidates, index])] == index:
                t2s_hits += 1
        metrics[f"T=4 S->T" if t_value == 4 else f"T={t_value} S->T"] = s2t_hits / n
        metrics[f"T=4 T->S" if t_value == 4 else f"T={t_value} T->S"] = t2s_hits / n
    return metrics


def evaluate(smiles_list: list[str], text_list: list[str], args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    bert_model = BertForPreTraining.from_pretrained("allenai/scibert_scivocab_uncased")
    model = BigModel(bert_model.bert)
    ckpt_path = os.path.join(args.dataspace, "pretrained_KV-PLM", "ckpt_ret01.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    mol_repr = encode_strings(
        smiles_list, tokenizer, model, device, args.batch_size, args.max_seq_len
    )
    txt_repr = encode_strings(
        text_list, tokenizer, model, device, args.batch_size, args.max_seq_len
    )

    mol_repr = F.normalize(mol_repr, dim=-1)
    txt_repr = F.normalize(txt_repr, dim=-1)
    sim = (txt_repr @ mol_repr.T).numpy()

    recall1, mrr = recall_mrr(sim)
    recall_k = recall_at_k(sim)
    t_metrics = t_choose_one(sim, seed=args.seed)
    return {
        "Recall@1": recall1,
        "MRR": mrr,
        "Recall@5": recall_k[5],
        "Recall@10": recall_k[10],
        **t_metrics,
        "gallery_size": float(sim.shape[0]),
    }


def write_json(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    source_csv = Path(args.source_csv).expanduser()
    split_manifest = Path(args.split_manifest).expanduser()
    with_out = Path(args.with_out).expanduser()
    nodrug_out = Path(args.nodrug_out).expanduser()

    smiles_with, text_with = load_aligned_pairs(
        source_csv, split_manifest, args.smiles_column, args.text_column, remove_drug_name=False
    )
    smiles_without, text_without = load_aligned_pairs(
        source_csv, split_manifest, args.smiles_column, args.text_column, remove_drug_name=True
    )

    with_metrics = evaluate(smiles_with, text_with, args)
    without_metrics = evaluate(smiles_without, text_without, args)

    write_json(with_out, with_metrics)
    write_json(nodrug_out, without_metrics)
    print(f"Wrote {with_out}")
    print(f"Wrote {nodrug_out}")


if __name__ == "__main__":
    main()
