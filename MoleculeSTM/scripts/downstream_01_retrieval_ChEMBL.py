"""
downstream_01_retrieval_ChEMBL.py

Zero-shot structure-text retrieval on ChEMBL using MoleculeSTM.
Mirrors the interface of the original DrugBank retrieval script exactly:

    python downstream_01_retrieval_Description_Pharmacodynamics.py \\
        --task=molecule_description_removed_PubChem \\
        --molecule_type=SMILES \\
        --input_model_dir=../data/pretrained_MoleculeSTM/SMILES

So the ChEMBL equivalent is:

    python downstream_01_retrieval_ChEMBL.py \\
        --task=molecule_mechanism_filtered \\
        --molecule_type=SMILES \\
        --input_model_dir=../data/pretrained_MoleculeSTM/SMILES

Data is resolved relative to the script location (../data/ChEMBL_data/),
exactly as the original script resolves ../data/DrugBank_data/.

Parameters added beyond the original:
    --dataspace   : override the default ../data root (optional)
    --output_dir  : save results CSV (optional, original script prints only)
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# MoleculeSTM imports — run from inside the scripts/ folder of the repo
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from MoleculeSTM.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM


# ---------------------------------------------------------------------------
# Task → CSV file mapping  (mirrors task→file logic in the original script)
# ---------------------------------------------------------------------------

TASK_TO_FILE = {
    # PubChemSTM-filtered split (mirrors *_removed_PubChem tasks in DrugBank)
    "molecule_mechanism_filtered": "ChEMBL_retrieval_test.csv",
    # Full split — no filtering
    "molecule_mechanism_all":      "ChEMBL_retrieval_test_all.csv",
}


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
# Model loading  (identical to original MoleculeSTM scripts)
# ---------------------------------------------------------------------------

def load_molecule_model(args, device):
    MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
    molecule_model = MegaMolBART_wrapper.model
    molecule_dim = 256
    mol2latent = nn.Linear(molecule_dim, args.SSL_emb_dim)

    input_model_path = os.path.join(args.input_model_dir, "molecule_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)

    input_model_path = os.path.join(args.input_model_dir, "mol2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)

    molecule_model = molecule_model.to(device).eval()
    mol2latent = mol2latent.to(device).eval()
    return MegaMolBART_wrapper, molecule_model, mol2latent


def load_text_model(args, device):
    pretrained_SciBERT_folder = os.path.join(args.dataspace, 'pretrained_SciBERT')
    text_tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased',
        cache_dir=pretrained_SciBERT_folder)
    text_model = AutoModel.from_pretrained(
        'allenai/scibert_scivocab_uncased',
        cache_dir=pretrained_SciBERT_folder)
    text2latent = nn.Linear(768, args.SSL_emb_dim)

    input_model_path = os.path.join(args.input_model_dir, "text_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict)

    input_model_path = os.path.join(args.input_model_dir, "text2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent.load_state_dict(state_dict)

    text_model = text_model.to(device).eval()
    text2latent = text2latent.to(device).eval()
    return text_tokenizer, text_model, text2latent


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_all_embeddings(smiles_list, text_list,
                       MegaMolBART_wrapper, mol2latent,
                       text_tokenizer, text_model, text2latent,
                       args, device):
    mol_reprs, txt_reprs = [], []
    for start in tqdm(range(0, len(smiles_list), args.batch_size),
                      desc="Extracting embeddings"):
        smi_batch  = smiles_list[start: start + args.batch_size]
        text_batch = text_list[start:  start + args.batch_size]

        mol_repr = get_molecule_repr_MoleculeSTM(
            smi_batch, mol2latent=None,
            molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper)
        mol_repr = nn.functional.normalize(mol2latent(mol_repr), dim=-1)
        mol_reprs.append(mol_repr.cpu())

        text_tokens_ids, text_masks = prepare_text_tokens(
            device=device, description=text_batch,
            tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
        text_out  = text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_out["pooler_output"]
        text_repr = nn.functional.normalize(text2latent(text_repr), dim=-1)
        txt_reprs.append(text_repr.cpu())

    return torch.cat(mol_reprs, dim=0), torch.cat(txt_reprs, dim=0)


# ---------------------------------------------------------------------------
# T-choose-one retrieval  (mirrors paper evaluation exactly)
# ---------------------------------------------------------------------------

def retrieval_accuracy(query_repr, key_repr, T_list=(4, 10, 20), seed=42):
    rng = np.random.default_rng(seed)
    N   = query_repr.shape[0]
    sim = (query_repr @ key_repr.T).numpy()   # (N, N) cosine similarity

    results = {}
    for T in T_list:
        if T > N:
            print(f"  Warning: T={T} > N={N}, skipping.")
            results[T] = float("nan")
            continue
        correct = 0
        for i in range(N):
            negs       = rng.choice([j for j in range(N) if j != i], size=T-1, replace=False)
            candidates = np.concatenate([[i], negs])
            if candidates[np.argmax(sim[i, candidates])] == i:
                correct += 1
        results[T] = correct / N
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve CSV path — same relative convention as the original DrugBank script
    if args.task not in TASK_TO_FILE:
        raise ValueError(f"Unknown task '{args.task}'. Choices: {list(TASK_TO_FILE)}")
    csv_path = os.path.join(args.dataspace, "ChEMBL_data", TASK_TO_FILE[args.task])
    print(f"Dataset: {csv_path}")

    # Load pairs
    df = pd.read_csv(csv_path).dropna(subset=["canonical_smiles", "text"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["canonical_smiles", "text"]).reset_index(drop=True)
    smiles_list = df["canonical_smiles"].tolist()
    text_list   = df["text"].tolist()
    print(f"Loaded {len(df)} structure-text pairs.")

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} pairs — need ≥20 for T=20 evaluation.")

    # Load models
    MegaMolBART_wrapper, molecule_model, mol2latent = load_molecule_model(args, device)
    text_tokenizer, text_model, text2latent = load_text_model(args, device)

    # Extract embeddings
    mol_repr, txt_repr = get_all_embeddings(
        smiles_list, text_list,
        MegaMolBART_wrapper, mol2latent,
        text_tokenizer, text_model, text2latent,
        args, device)

    T_list = [4, 10, 20]

    # Direction 1: Given Structure → Retrieve Text
    print("\n=== Given Structure → Retrieve Text ===")
    acc_s2t = retrieval_accuracy(mol_repr, txt_repr, T_list=T_list, seed=args.seed)
    for T, acc in acc_s2t.items():
        print(f"  T={T:>2}  Accuracy = {acc*100:.2f}%")

    # Direction 2: Given Text → Retrieve Structure
    print("\n=== Given Text → Retrieve Structure ===")
    acc_t2s = retrieval_accuracy(txt_repr, mol_repr, T_list=T_list, seed=args.seed)
    for T, acc in acc_t2s.items():
        print(f"  T={T:>2}  Accuracy = {acc*100:.2f}%")

    # Summary  (matches format of Tables 8-10 in the paper)
    print(f"\n{'Task':<35} {'Direction':>20}  " +
          "  ".join(f"T={T}" for T in T_list))
    print("-" * 75)
    print(f"{args.task:<35} {'Given Structure':>20}  " +
          "  ".join(f"{acc_s2t[T]*100:5.2f}%" for T in T_list))
    print(f"{args.task:<35} {'Given Text':>20}  " +
          "  ".join(f"{acc_t2s[T]*100:5.2f}%" for T in T_list))

    # Optional: save CSV
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        rows = []
        for T in T_list:
            rows += [
                {"task": args.task, "molecule_type": args.molecule_type,
                 "direction": "Given Structure", "T": T, "accuracy": acc_s2t[T]},
                {"task": args.task, "molecule_type": args.molecule_type,
                 "direction": "Given Text",      "T": T, "accuracy": acc_t2s[T]},
            ]
        out = os.path.join(args.output_dir, f"results_{args.task}_{args.molecule_type}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Same 3 args as the original DrugBank script ──────────────────────
    parser.add_argument(
        "--task", type=str, default="molecule_mechanism_filtered",
        choices=list(TASK_TO_FILE.keys()),
        help="molecule_mechanism_filtered : PubChemSTM molecules removed (recommended)\n"
             "molecule_mechanism_all      : all positive pairs")
    parser.add_argument(
        "--molecule_type", type=str, default="SMILES", choices=["SMILES"],
        help="Molecule encoder type.")
    parser.add_argument(
        "--input_model_dir", type=str, required=True,
        help="Directory with pretrained MoleculeSTM checkpoint .pth files.")

    # ── Optional overrides (not in original, added for flexibility) ───────
    parser.add_argument(
        "--dataspace", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        help="Root data directory.")
    parser.add_argument(
        "--vocab_path", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MoleculeSTM", "bart_vocab.txt"),
        help="Path to MegaMolBART vocabulary file.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="If set, saves a results CSV here (original script prints only).")

    # ── Model / inference hyperparams ────────────────────────────────────
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--seed",        type=int, default=42)

    args = parser.parse_args()
    main(args)