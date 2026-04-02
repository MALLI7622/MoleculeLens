"""
downstream_01_retrieval_ChEMBL_KV-PLM.py

Zero-shot structure-text retrieval on ChEMBL using KV-PLM.

KV-PLM trick: the same SciBERT-based BigModel encodes BOTH SMILES strings
and natural-language text, so no separate molecule encoder is needed.

Usage (from inside scripts/):
    python downstream_01_retrieval_ChEMBL_KV-PLM.py \\
        --task=molecule_mechanism_all

Mirrors the output format of downstream_01_retrieval_ChEMBL.py so results
slot directly into Table 1 / Table 2 of the paper.

Two evaluation modes are run back-to-back:
  1. With original text  (drug names present  → leakage condition)
  2. With drug names stripped  (leakage ablation)
"""

import os
import re
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining


# ---------------------------------------------------------------------------
# Task → CSV mapping  (same as downstream_01_retrieval_ChEMBL.py)
# ---------------------------------------------------------------------------

TASK_TO_FILE = {
    "molecule_mechanism_filtered": "ChEMBL_retrieval_test.csv",
    "molecule_mechanism_all":      "ChEMBL_retrieval_test_all.csv",
}


# ---------------------------------------------------------------------------
# BigModel  (identical to all other KV-PLM scripts in this repo)
# ---------------------------------------------------------------------------

class BigModel(nn.Module):
    def __init__(self, main_model):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(
            tok, token_type_ids=typ, attention_mask=att)["pooler_output"]
        return self.dropout(pooled_output)


# ---------------------------------------------------------------------------
# Drug-name stripping  (leakage ablation)
# ---------------------------------------------------------------------------

# Matches a capitalised token that is:
#   - not the first token of the sentence (we strip from the mid-sentence only)
#   - ≥ 4 chars long
#   - optionally followed by a number or hyphen-suffix (e.g. "Compound-1")
# Replace with the neutral placeholder "the molecule".
_DRUG_NAME_RE = re.compile(
    r'(?<![.!?]\s)'          # not immediately after sentence-end punctuation
    r'\b([A-Z][a-zA-Z0-9]'  # starts with uppercase, ≥1 more alphanumeric
    r'[a-zA-Z0-9\-]{2,})'   # at least 2 more chars (total ≥ 4)
    r'\b'
)


def strip_drug_names(text: str) -> str:
    """
    Replace capitalised drug-name-like tokens with 'the molecule'.

    Heuristic: a mid-sentence token that starts with an uppercase letter
    and is ≥ 4 characters long is treated as a potential drug name.
    Sentence-initial tokens are left intact so we don't destroy grammar.
    """
    # Split into sentences, preserve the first token of each sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for sent in sentences:
        # Protect the very first word of the sentence.
        m = re.match(r'^(\S+)(.*)', sent, re.DOTALL)
        if not m:
            cleaned.append(sent)
            continue
        first_word, rest = m.group(1), m.group(2)
        rest_cleaned = _DRUG_NAME_RE.sub('the molecule', rest)
        cleaned.append(first_word + rest_cleaned)
    return ' '.join(cleaned)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_strings(strings, tokenizer, model, device, batch_size=32, max_seq_len=512):
    """Encode a list of strings (SMILES or text) through BigModel."""
    model.eval()
    all_reprs = []
    for start in tqdm(range(0, len(strings), batch_size), desc="Encoding", leave=False):
        batch = strings[start: start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        tok = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        repr_ = model(tok, att, cud=torch.cuda.is_available())
        all_reprs.append(repr_.cpu())
    return torch.cat(all_reprs, dim=0)


# ---------------------------------------------------------------------------
# T-choose-one retrieval  (mirrors downstream_01_retrieval_ChEMBL.py exactly)
# ---------------------------------------------------------------------------

def retrieval_accuracy(query_repr, key_repr, T_list=(4, 10, 20), seed=42):
    rng = np.random.default_rng(seed)
    N   = query_repr.shape[0]
    # Normalise (KV-PLM checkpoint was trained with normalised embeddings)
    q = F.normalize(query_repr, dim=-1)
    k = F.normalize(key_repr,   dim=-1)
    sim = (q @ k.T).numpy()  # (N, N)

    results = {}
    for T in T_list:
        if T > N:
            print(f"  Warning: T={T} > N={N}, skipping.")
            results[T] = float("nan")
            continue
        correct = 0
        for i in range(N):
            negs       = rng.choice([j for j in range(N) if j != i], size=T - 1, replace=False)
            candidates = np.concatenate([[i], negs])
            if candidates[np.argmax(sim[i, candidates])] == i:
                correct += 1
        results[T] = correct / N
    return results


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_eval(smiles_list, text_list, tokenizer, model, device, T_list, seed, batch_size, max_seq_len, label=""):
    mol_repr  = encode_strings(smiles_list, tokenizer, model, device, batch_size, max_seq_len)
    txt_repr  = encode_strings(text_list,   tokenizer, model, device, batch_size, max_seq_len)

    header = f"\n{'='*60}\n{label}\n{'='*60}"
    print(header)

    print("\n--- Given Structure → Retrieve Text ---")
    acc_s2t = retrieval_accuracy(mol_repr, txt_repr, T_list=T_list, seed=seed)
    for T, acc in acc_s2t.items():
        print(f"  T={T:>2}  Accuracy = {acc*100:.2f}%")

    print("\n--- Given Text → Retrieve Structure ---")
    acc_t2s = retrieval_accuracy(txt_repr, mol_repr, T_list=T_list, seed=seed)
    for T, acc in acc_t2s.items():
        print(f"  T={T:>2}  Accuracy = {acc*100:.2f}%")

    return acc_s2t, acc_t2s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve CSV
    if args.task not in TASK_TO_FILE:
        raise ValueError(f"Unknown task '{args.task}'. Choices: {list(TASK_TO_FILE)}")
    csv_path = os.path.join(args.dataspace, "ChEMBL_data", TASK_TO_FILE[args.task])
    print(f"Dataset: {csv_path}")

    df = pd.read_csv(csv_path).dropna(subset=["canonical_smiles", "text"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["canonical_smiles", "text"]).reset_index(drop=True)
    smiles_list = df["canonical_smiles"].tolist()
    text_list   = df["text"].tolist()
    print(f"Loaded {len(df)} structure-text pairs.")

    if len(df) < max(args.T_list):
        raise ValueError(f"Only {len(df)} pairs — need ≥{max(args.T_list)} for T={max(args.T_list)} evaluation.")

    # Load KV-PLM
    ckpt_path = os.path.join(args.dataspace, "pretrained_KV-PLM", "ckpt_ret01.pt")
    print(f"Loading KV-PLM checkpoint from {ckpt_path} ...")
    tokenizer  = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    bert_model = BertForPreTraining.from_pretrained("allenai/scibert_scivocab_uncased")
    model      = BigModel(bert_model.bert)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    T_list = args.T_list

    # ── Condition 1: original text (drug names present) ────────────────────
    acc_s2t_orig, acc_t2s_orig = run_eval(
        smiles_list, text_list, tokenizer, model, device,
        T_list, args.seed, args.batch_size, args.max_seq_len,
        label="KV-PLM on ChEMBL — WITH drug names (leakage condition)")

    # ── Condition 2: drug names stripped (leakage ablation) ────────────────
    text_stripped = [strip_drug_names(t) for t in text_list]
    acc_s2t_strip, acc_t2s_strip = run_eval(
        smiles_list, text_stripped, tokenizer, model, device,
        T_list, args.seed, args.batch_size, args.max_seq_len,
        label="KV-PLM on ChEMBL — WITHOUT drug names (leakage ablation)")

    # ── Summary table (paper-ready) ────────────────────────────────────────
    print(f"\n\n{'Model/Condition':<45} {'Direction':>20}  " +
          "  ".join(f"T={T}" for T in T_list))
    print("-" * 90)
    for cond_label, acc_s2t, acc_t2s in [
        ("KV-PLM + drug names",    acc_s2t_orig,  acc_t2s_orig),
        ("KV-PLM - drug names",    acc_s2t_strip, acc_t2s_strip),
    ]:
        print(f"{cond_label:<45} {'Given Structure':>20}  " +
              "  ".join(f"{acc_s2t[T]*100:5.2f}%" for T in T_list))
        print(f"{cond_label:<45} {'Given Text':>20}  " +
              "  ".join(f"{acc_t2s[T]*100:5.2f}%" for T in T_list))

    # Drop rate summary  (mirrors MoleculeSTM's 84.3% collapse figure)
    print("\n── Name-leakage drop (Given Text → Retrieve Structure, T=4) ──")
    orig  = acc_t2s_orig.get(4, float("nan"))
    strip = acc_t2s_strip.get(4, float("nan"))
    if not (np.isnan(orig) or np.isnan(strip)) and orig > 0:
        drop_pct = (orig - strip) / orig * 100
        print(f"  With names: {orig*100:.2f}%   Without: {strip*100:.2f}%   Drop: {drop_pct:.1f}%")

    # ── Optional CSV save ──────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        rows = []
        for T in T_list:
            for cond, a_s2t, a_t2s in [
                ("with_drug_names",    acc_s2t_orig,  acc_t2s_orig),
                ("without_drug_names", acc_s2t_strip, acc_t2s_strip),
            ]:
                rows += [
                    {"model": "KV-PLM", "task": args.task, "condition": cond,
                     "direction": "Given Structure", "T": T, "accuracy": a_s2t[T]},
                    {"model": "KV-PLM", "task": args.task, "condition": cond,
                     "direction": "Given Text",      "T": T, "accuracy": a_t2s[T]},
                ]
        out = os.path.join(args.output_dir, f"results_KVPLM_{args.task}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", type=str, default="molecule_mechanism_all",
        choices=list(TASK_TO_FILE.keys()))
    parser.add_argument(
        "--dataspace", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        help="Root data directory (must contain pretrained_KV-PLM/ and ChEMBL_data/).")
    parser.add_argument("--T_list",      type=int, nargs="+", default=[4, 10, 20])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="If set, saves a results CSV here.")

    args = parser.parse_args()
    main(args)
