"""
Track 1: ECFP4 Bit Attribution ("Molecular Lens")

For any drug-text retrieval pair, compute a closed-form linear saliency score for every ECFP4 bit
to the similarity score, then decode those bits back to RDKit substructures and
highlight them on the molecule.

Attribution formula (closed-form linear approximation):
    attr_vec = W_m.weight.T @ bt_j       shape [2048]
    attr_k   = attr_vec[k]   (positive → bit k pushes drug toward text)

This script writes raw Track 1 artifacts. The camera-ready CSVs used by the paper
are synced later by 09b_diagonal_artifact_sync.py.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

os.makedirs("outputs/track1", exist_ok=True)
os.makedirs("outputs/track1/case_studies", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load saved tensors
# ---------------------------------------------------------------------------
print("Loading saved tensors...")

Wm_state  = torch.load("outputs/proj_mol.pt",        map_location="cpu")
Wt_state  = torch.load("outputs/proj_text.pt",       map_location="cpu")
Wm_nd     = torch.load("outputs/proj_mol_nodrug.pt",  map_location="cpu")
Wt_nd     = torch.load("outputs/proj_text_nodrug.pt", map_location="cpu")

Bm = torch.load("outputs/Bm_test.pt",        map_location="cpu")  # [435, 256]
Bt = torch.load("outputs/Bt_test.pt",        map_location="cpu")  # [435, 256]
Bt_nodrug = torch.load("outputs/Bt_test_nodrug.pt", map_location="cpu")  # [435, 256]
Bm_nodrug = torch.load("outputs/Bm_test_nodrug.pt", map_location="cpu")  # [435, 256]

X_mol = np.load("outputs/X_mol_test.npy")   # [435, 2048]  raw ECFP4 bits

df = pd.read_csv("outputs/test_df.csv")
N  = len(df)  # 435

# Extract weight matrices  (nn.Linear state_dict: weight shape [out, in])
W_m  = Wm_state["weight"]   # [256, 2048]
W_t  = Wt_state["weight"]   # [256, 768]
W_m_nd = Wm_nd["weight"]
W_t_nd = Wt_nd["weight"]

print(f"Loaded: W_m {W_m.shape}, W_t {W_t.shape}, Bm {Bm.shape}, Bt {Bt.shape}")
print(f"test_df shape: {df.shape}, columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. Core attribution functions
# ---------------------------------------------------------------------------

def ecfp_attribution(j, W_m_mat, Bt_mat):
    """
    Returns attribution vector in ECFP4 space for text query j.
    attr[k] = how much ECFP4 bit k pushes a drug toward text j.
    Shape: [2048]
    """
    # W_m.T @ bt_j  →  [2048, 256] @ [256]  →  [2048]
    return W_m_mat.T @ Bt_mat[j]


def get_atom_highlights(smiles, attribution_vec, top_k=15):
    """
    Decode top-k ECFP4 bits (by |attribution|) back to atoms.
    Returns (mol, atom_highlights, bit_attrs) where atom_highlights is a
    dict mapping atom_idx → attribution value.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}, {}

    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)

    # Only consider bits that are actually ON in this molecule
    present_bits = set(info.keys())
    attr_present = {k: attribution_vec[k].item() for k in present_bits}

    # Top-k by absolute value among present bits
    sorted_bits = sorted(attr_present.keys(),
                         key=lambda k: abs(attr_present[k]), reverse=True)[:top_k]

    atom_attr = {}   # atom_idx → summed attribution
    for bit in sorted_bits:
        if bit not in info:
            continue
        for (center_atom, radius) in info[bit]:
            if radius == 0:
                # single atom environment
                atom_attr[center_atom] = atom_attr.get(center_atom, 0) + attr_present[bit]
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    for a_idx in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                        atom_attr[a_idx] = atom_attr.get(a_idx, 0) + attr_present[bit]

    return mol, atom_attr, attr_present


def draw_highlighted_molecule(smiles, atom_attr, title="", size=(400, 300)):
    """
    Draw molecule with atoms coloured by attribution value.
    Returns a PIL Image.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_list = list(atom_attr.keys())
    values    = np.array([atom_attr[a] for a in atom_list])

    # Colour: positive = green, negative = red
    if len(values) > 0:
        vmax = max(abs(values.max()), abs(values.min()), 1e-8)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.cm.RdYlGn
        atom_colours = {a: cmap(norm(atom_attr[a]))[:3] for a in atom_list}
    else:
        atom_colours = {}

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareMolForDrawing(mol)

    if atom_colours:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atom_list,
            highlightAtomColors=atom_colours,
            highlightBonds=[],
            highlightBondColors={},
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Convert SVG → PIL via cairosvg if available, else save SVG directly
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg.encode())
        img = Image.open(io.BytesIO(png_data))
    except ImportError:
        # Fallback: use rdkit PNG renderer
        drawer2 = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer2.drawOptions().addAtomIndices = False
        rdMolDraw2D.PrepareMolForDrawing(mol)
        if atom_colours:
            drawer2.DrawMolecule(
                mol,
                highlightAtoms=atom_list,
                highlightAtomColors=atom_colours,
                highlightBonds=[],
                highlightBondColors={},
            )
        else:
            drawer2.DrawMolecule(mol)
        drawer2.FinishDrawing()
        png_data = drawer2.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))

    return img


# ---------------------------------------------------------------------------
# 3. Compute full similarity matrix and retrieval ranks
# ---------------------------------------------------------------------------
print("\nComputing similarity matrices...")

sim_rich   = Bm @ Bt.T          # [435, 435]  drug × text (text_rich)
sim_nodrug = Bm_nodrug @ Bt_nodrug.T

# Recall@K helper
def recall_at_k(sim_matrix, k=1):
    N = sim_matrix.shape[0]
    ranks = sim_matrix.argsort(dim=1, descending=True)
    correct = (ranks[:, :k] == torch.arange(N).unsqueeze(1)).any(dim=1).float()
    return correct.mean().item(), correct

r1_rich,   hits_rich   = recall_at_k(sim_rich,   1)
r5_rich,   hits_r5     = recall_at_k(sim_rich,   5)
r10_rich,  _           = recall_at_k(sim_rich,  10)
r1_nodrug, hits_nodrug = recall_at_k(sim_nodrug, 1)

print(f"  text_rich   R@1={r1_rich:.3f}  R@5={r5_rich:.3f}  R@10={r10_rich:.3f}")
print(f"  text_nodrug R@1={r1_nodrug:.3f}")

# Rank of the correct answer for each drug
def get_ranks(sim_matrix):
    N = sim_matrix.shape[0]
    order = sim_matrix.argsort(dim=1, descending=True)
    ranks = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        ranks[i] = (order[i] == i).nonzero(as_tuple=True)[0][0]
    return ranks  # 0-indexed rank of correct item

ranks_rich   = get_ranks(sim_rich)
ranks_nodrug = get_ranks(sim_nodrug)

# ---------------------------------------------------------------------------
# 4. Analysis A — Per-pair case studies (6-8 correctly retrieved pairs)
# ---------------------------------------------------------------------------
print("\n--- Analysis A: Per-pair case studies ---")

# Pick diverse correctly retrieved pairs spanning different mechanisms
correctly_retrieved = (ranks_rich == 0).nonzero(as_tuple=True)[0].tolist()
print(f"  Correctly retrieved at R@1: {len(correctly_retrieved)}/{N}")

# Sample across different mechanism categories
mech_groups = df.loc[correctly_retrieved].groupby("mechanism_of_action")
selected_indices = []
seen_mechs = set()
for mech, group in sorted(mech_groups, key=lambda x: -len(x[1])):
    if mech not in seen_mechs and len(selected_indices) < 8:
        idx = group.index[0]
        selected_indices.append(idx)
        seen_mechs.add(mech)

print(f"  Selected {len(selected_indices)} case studies:")
for idx in selected_indices:
    row = df.iloc[idx]
    print(f"    [{idx}] {row['pref_name']} | {row['mechanism_of_action']}")

# Generate figures for each case study
for case_num, idx in enumerate(selected_indices):
    row    = df.iloc[idx]
    smiles = row["smiles"]
    drug   = row["pref_name"]
    mech   = row["mechanism_of_action"]

    attr_vec   = ecfp_attribution(idx, W_m, Bt)         # [2048]
    mol, atom_attr, bit_attrs = get_atom_highlights(smiles, attr_vec, top_k=15)
    if mol is None:
        print(f"    Skipping {drug}: invalid SMILES")
        continue

    # Top-20 present bits by attribution
    sorted_bits = sorted(bit_attrs.keys(), key=lambda k: abs(bit_attrs[k]), reverse=True)[:20]
    bit_vals    = [bit_attrs[b] for b in sorted_bits]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4])

    # Left: highlighted molecule
    ax_mol = fig.add_subplot(gs[0])
    mol_img = draw_highlighted_molecule(smiles, atom_attr, title=drug)
    if mol_img is not None:
        ax_mol.imshow(mol_img)
    ax_mol.axis("off")
    ax_mol.set_title(f"{drug}\n({mech})", fontsize=9, wrap=True)

    # Right: attribution bar chart
    ax_bar = fig.add_subplot(gs[1])
    colours = ["forestgreen" if v > 0 else "crimson" for v in bit_vals]
    ax_bar.barh(range(len(sorted_bits)), bit_vals[::-1], color=colours[::-1])
    ax_bar.set_yticks(range(len(sorted_bits)))
    ax_bar.set_yticklabels([f"Bit {b}" for b in sorted_bits[::-1]], fontsize=7)
    ax_bar.set_xlabel("Attribution (W_m.T · b_text)")
    ax_bar.set_title("Top-20 ECFP4 Bit Attributions\n(green=push toward text, red=push away)")
    ax_bar.axvline(0, color="black", linewidth=0.7)

    # Similarity score
    sim_score = sim_rich[idx, idx].item()
    fig.suptitle(
        f"Case {case_num+1}: {drug}  |  Similarity={sim_score:.3f}  |  Rank=1 (correct)",
        fontsize=10, fontweight="bold"
    )

    out_path = f"outputs/track1/case_studies/case_{case_num+1:02d}_{drug.replace(' ', '_')[:20]}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")

# ---------------------------------------------------------------------------
# 5. Analysis B — Aggregate attribution across all 435 test pairs
# ---------------------------------------------------------------------------
print("\n--- Analysis B: Aggregate bit attribution across all test pairs ---")

# For each test pair (i, i), compute attribution and find top-10 bits
bit_top10_counts = np.zeros(2048)
bit_mean_attr    = np.zeros(2048)
bit_sum_attr     = np.zeros(2048)

for i in range(N):
    attr_vec = ecfp_attribution(i, W_m, Bt)   # [2048]
    attr_np  = attr_vec.numpy()
    # Only present bits
    info_i = {}
    mol_i  = Chem.MolFromSmiles(df.iloc[i]["smiles"])
    if mol_i is None:
        continue
    AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048, bitInfo=info_i)
    present = list(info_i.keys())
    if not present:
        continue

    present_arr  = np.array(present)
    present_vals = attr_np[present]
    top10_local  = present_arr[np.argsort(np.abs(present_vals))[-10:]]
    bit_top10_counts[top10_local] += 1
    bit_sum_attr[present] += present_vals
    bit_mean_attr[present] += 1   # count occurrences

# Normalize to get mean attribution per bit
with np.errstate(divide="ignore", invalid="ignore"):
    bit_mean_vals = np.where(bit_mean_attr > 0,
                             bit_sum_attr / bit_mean_attr, 0.0)

top_universal = np.argsort(bit_top10_counts)[-30:][::-1]
print("  Top-30 most frequently top-10 ECFP4 bits (universal importance):")
for rank, bit in enumerate(top_universal[:10]):
    print(f"    #{rank+1:2d}  bit {bit:4d}  freq={bit_top10_counts[bit]:.0f}/{N}")

# Plot top-30 universal bits
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
top20 = top_universal[:20]
ax.barh(range(20), bit_top10_counts[top20][::-1], color="steelblue")
ax.set_yticks(range(20))
ax.set_yticklabels([f"Bit {b}" for b in top20[::-1]], fontsize=8)
ax.set_xlabel("# test pairs where bit is in top-10 attribution")
ax.set_title("Most Universally Important\nECFP4 Bits (all 435 test pairs)")

ax = axes[1]
# Mean attribution of top bits
top20_mean = top_universal[:20]
mean_vals  = bit_mean_vals[top20_mean]
colours    = ["forestgreen" if v > 0 else "crimson" for v in mean_vals[::-1]]
ax.barh(range(20), mean_vals[::-1], color=colours)
ax.set_yticks(range(20))
ax.set_yticklabels([f"Bit {b}" for b in top20_mean[::-1]], fontsize=8)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlabel("Mean attribution value")
ax.set_title("Mean Attribution of Top-20\nUniversal Bits")

plt.suptitle("Aggregate ECFP4 Bit Attribution — 435 Test Pairs", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/track1/aggregate_attribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/track1/aggregate_attribution.png")

# Save aggregate data
agg_df = pd.DataFrame({
    "bit":         top_universal,
    "top10_freq":  bit_top10_counts[top_universal],
    "mean_attr":   bit_mean_vals[top_universal],
})
agg_df.to_csv("outputs/track1/aggregate_attribution.csv", index=False)
print("  Saved: outputs/track1/aggregate_attribution.csv")

# ---------------------------------------------------------------------------
# 6. Analysis C — Leakage attribution comparison (text_rich vs text_nodrug)
# ---------------------------------------------------------------------------
print("\n--- Analysis C: Leakage attribution comparison ---")

# For same drug, compare top-10 bits under text_rich vs text_nodrug projections
attribution_overlap = []

for i in range(N):
    attr_rich   = ecfp_attribution(i, W_m,    Bt)          # [2048]
    attr_nodrug = ecfp_attribution(i, W_m_nd,  Bt_nodrug)  # [2048]

    info_i = {}
    mol_i  = Chem.MolFromSmiles(df.iloc[i]["smiles"])
    if mol_i is None:
        attribution_overlap.append(np.nan)
        continue
    AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048, bitInfo=info_i)
    present = list(info_i.keys())
    if len(present) < 10:
        attribution_overlap.append(np.nan)
        continue

    present_arr = np.array(present)

    top10_rich   = set(present_arr[np.argsort(np.abs(attr_rich.numpy()[present]))[-10:]].tolist())
    top10_nodrug = set(present_arr[np.argsort(np.abs(attr_nodrug.numpy()[present]))[-10:]].tolist())

    jaccard = len(top10_rich & top10_nodrug) / len(top10_rich | top10_nodrug)
    attribution_overlap.append(jaccard)

overlap_arr = np.array([x for x in attribution_overlap if not np.isnan(x)])
print(f"  Mean Jaccard overlap of top-10 attribution bits (rich vs nodrug): {overlap_arr.mean():.3f}")
print(f"  Std: {overlap_arr.std():.3f}")
print(f"  Low overlap (<0.2) pairs: {(overlap_arr < 0.2).sum()}/{len(overlap_arr)} — these are leakage-driven")
print(f"  High overlap (>0.6) pairs: {(overlap_arr > 0.6).sum()}/{len(overlap_arr)} — mechanism-driven")

# Pick a leakage example (low overlap, correctly retrieved only in rich)
leakage_candidates = [
    i for i in range(N)
    if not np.isnan(attribution_overlap[i])
    and attribution_overlap[i] < 0.2
    and hits_rich[i].item() == 1
    and hits_nodrug[i].item() == 0
]
print(f"  Leakage candidates (correct rich, wrong nodrug, low attr overlap): {len(leakage_candidates)}")

if leakage_candidates:
    i = leakage_candidates[0]
    row = df.iloc[i]
    attr_rich_vec   = ecfp_attribution(i, W_m,   Bt)
    attr_nodrug_vec = ecfp_attribution(i, W_m_nd, Bt_nodrug)

    info_i = {}
    mol_i  = Chem.MolFromSmiles(row["smiles"])
    AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048, bitInfo=info_i)
    present = list(info_i.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, attr_vec, label, colour in [
        (axes[0], attr_rich_vec,   "text_rich",   "steelblue"),
        (axes[1], attr_nodrug_vec, "text_nodrug",  "darkorange"),
    ]:
        present_arr  = np.array(present)
        present_vals = attr_vec.numpy()[present]
        top15        = present_arr[np.argsort(np.abs(present_vals))[-15:]]
        top15_vals   = attr_vec.numpy()[top15]
        colours      = ["forestgreen" if v > 0 else "crimson" for v in top15_vals[::-1]]
        ax.barh(range(15), top15_vals[::-1], color=colours)
        ax.set_yticks(range(15))
        ax.set_yticklabels([f"Bit {b}" for b in top15[::-1]], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Attribution")
        ax.set_title(f"{label}\n(Rank @1 = {'correct' if label=='text_rich' else 'wrong'})")

    fig.suptitle(
        f"Leakage Comparison: {row['pref_name']}\n"
        f"Mech: {row['mechanism_of_action']}\n"
        f"Attribution Jaccard overlap = {attribution_overlap[i]:.2f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("outputs/track1/leakage_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: outputs/track1/leakage_comparison.png  (drug: {row['pref_name']})")

# Plot distribution of Jaccard overlaps
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(overlap_arr, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(overlap_arr.mean(), color="red", linestyle="--", label=f"Mean={overlap_arr.mean():.3f}")
ax.set_xlabel("Jaccard Overlap of Top-10 Attribution Bits (rich vs nodrug)")
ax.set_ylabel("# Drug-text Pairs")
ax.set_title("Attribution Stability Under Drug-Name Removal\n(Low = attribution pattern changed = leakage-driven)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/track1/leakage_jaccard_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/track1/leakage_jaccard_dist.png")

# Save per-pair leakage data
leakage_df = df[["molecule_chembl_id", "pref_name", "mechanism_of_action", "action_type"]].copy()
leakage_df["attr_jaccard_overlap"] = attribution_overlap
leakage_df["rank_rich"]   = ranks_rich.numpy()
leakage_df["rank_nodrug"] = ranks_nodrug.numpy()
leakage_df["hit_rich"]    = hits_rich.numpy().astype(int)
leakage_df["hit_nodrug"]  = hits_nodrug.numpy().astype(int)
leakage_df.to_csv("outputs/track1/leakage_per_pair_raw.csv", index=False)
print("  Saved: outputs/track1/leakage_per_pair_raw.csv")

# ---------------------------------------------------------------------------
# 7. Analysis D — Wrong-but-close retrieval analysis
# ---------------------------------------------------------------------------
print("\n--- Analysis D: Wrong-but-close retrieval analysis ---")

# Pairs where model retrieved rank-2 or rank-3
wrong_close = (ranks_rich >= 1) & (ranks_rich <= 2)
wrong_close_idx = wrong_close.nonzero(as_tuple=True)[0].tolist()
print(f"  Rank-2 or rank-3 retrievals: {len(wrong_close_idx)}")

# For each, compare attribution overlap between correct and retrieved drug
close_results = []
for i in wrong_close_idx[:50]:   # limit to first 50
    rank_i     = ranks_rich[i].item()
    # retrieved drug index = the one ranked first
    retrieved_j = sim_rich[i].argsort(descending=True)[0].item()

    attr_correct  = ecfp_attribution(i, W_m, Bt)   # attribution for correct text query i
    attr_retrieved = ecfp_attribution(retrieved_j, W_m, Bt)  # attribution for retrieved drug

    # Get present bits for both molecules
    info_c, info_r = {}, {}
    mol_c = Chem.MolFromSmiles(df.iloc[i]["smiles"])
    mol_r = Chem.MolFromSmiles(df.iloc[retrieved_j]["smiles"])
    if mol_c is None or mol_r is None:
        continue
    AllChem.GetMorganFingerprintAsBitVect(mol_c, radius=2, nBits=2048, bitInfo=info_c)
    AllChem.GetMorganFingerprintAsBitVect(mol_r, radius=2, nBits=2048, bitInfo=info_r)
    present_c = set(info_c.keys())
    present_r = set(info_r.keys())
    shared_bits = present_c & present_r

    # Among shared bits, how similar are their attributions?
    if len(shared_bits) < 5:
        continue

    shared_list = sorted(shared_bits)
    av_c = attr_correct.numpy()[shared_list]
    av_r = attr_retrieved.numpy()[shared_list]
    corr = np.corrcoef(av_c, av_r)[0, 1] if av_c.std() > 1e-8 else 0.0

    close_results.append({
        "drug_idx": i,
        "retrieved_idx": retrieved_j,
        "drug_name": df.iloc[i]["pref_name"],
        "retrieved_name": df.iloc[retrieved_j]["pref_name"],
        "rank": rank_i + 1,
        "shared_bits": len(shared_bits),
        "attr_correlation": corr,
        "moa_correct": df.iloc[i]["mechanism_of_action"],
        "moa_retrieved": df.iloc[retrieved_j]["mechanism_of_action"],
    })

close_df = pd.DataFrame(close_results)
if len(close_df) > 0:
    print(f"  Analysed {len(close_df)} wrong-but-close pairs")
    print(f"  Mean attribution correlation (correct vs retrieved drug): {close_df['attr_correlation'].mean():.3f}")
    print(f"  High correlation (>0.5): {(close_df['attr_correlation'] > 0.5).sum()} — chemically reasonable errors")
    print(f"  Low correlation (<0.1):  {(close_df['attr_correlation'] < 0.1).sum()} — structurally unrelated errors")

    # Show some examples
    print("\n  Top-5 chemically reasonable errors (high attribution correlation):")
    top5 = close_df.nlargest(5, "attr_correlation")
    for _, r in top5.iterrows():
        same_moa = r["moa_correct"] == r["moa_retrieved"]
        print(f"    corr={r['attr_correlation']:.2f}  "
              f"{r['drug_name']} → retrieved {r['retrieved_name']}  "
              f"same_moa={same_moa}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(close_df["attr_correlation"].dropna(), bins=25,
            color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(close_df["attr_correlation"].mean(), color="red", linestyle="--",
               label=f"Mean={close_df['attr_correlation'].mean():.3f}")
    ax.set_xlabel("Attribution Correlation (correct drug vs retrieved drug)")
    ax.set_ylabel("# Pairs")
    ax.set_title("Are Wrong-but-Close Errors Chemically Reasonable?\n"
                 "(high correlation = model confused structurally similar drugs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/track1/wrong_close_attr_corr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/track1/wrong_close_attr_corr.png")

    close_df.to_csv("outputs/track1/wrong_close_analysis_raw.csv", index=False)
    print("  Saved: outputs/track1/wrong_close_analysis_raw.csv")

# ---------------------------------------------------------------------------
# 8. Summary figure — Overview panel
# ---------------------------------------------------------------------------
print("\n--- Generating summary figure ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: R@1 rich vs nodrug bar
ax = axes[0, 0]
ax.bar(["text_rich", "text_nodrug"], [r1_rich, r1_nodrug], color=["steelblue", "darkorange"])
ax.set_ylim(0, 1)
ax.set_ylabel("Recall@1")
ax.set_title("A: Retrieval Performance\n(with vs without drug name)")
for i, v in enumerate([r1_rich, r1_nodrug]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

# Panel B: Top-20 universal bits frequency
ax = axes[0, 1]
top20 = top_universal[:20]
ax.barh(range(20), bit_top10_counts[top20][::-1] / N, color="steelblue")
ax.set_yticks(range(20))
ax.set_yticklabels([f"Bit {b}" for b in top20[::-1]], fontsize=7)
ax.set_xlabel("Fraction of test pairs where bit is in top-10")
ax.set_title("B: Most Universally Important\nECFP4 Bits")

# Panel C: Jaccard overlap distribution
ax = axes[1, 0]
ax.hist(overlap_arr, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(overlap_arr.mean(), color="red", linestyle="--",
           label=f"Mean={overlap_arr.mean():.3f}")
ax.set_xlabel("Jaccard Overlap (rich vs nodrug attribution)")
ax.set_ylabel("# pairs")
ax.set_title("C: Attribution Stability\nUnder Drug-Name Removal")
ax.legend(fontsize=8)

# Panel D: wrong-but-close correlation
ax = axes[1, 1]
if len(close_df) > 0:
    ax.hist(close_df["attr_correlation"].dropna(), bins=20,
            color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(close_df["attr_correlation"].mean(), color="red", linestyle="--",
               label=f"Mean={close_df['attr_correlation'].mean():.3f}")
    ax.set_xlabel("Attribution Correlation (correct vs retrieved)")
    ax.set_ylabel("# pairs")
    ax.set_title("D: Wrong-but-Close Errors\nAre They Chemically Reasonable?")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No rank-2/3 pairs to analyse", ha="center", va="center",
            transform=ax.transAxes)

plt.suptitle("Track 1 — ECFP4 Bit Attribution: MoleculeLens Interpretability",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/track1/summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/track1/summary.png")

# ---------------------------------------------------------------------------
# 9. Print final summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TRACK 1 COMPLETE — ECFP4 Attribution Results")
print("="*60)
print(f"  N test pairs:        {N}")
print(f"  R@1 text_rich:       {r1_rich:.3f}")
print(f"  R@1 text_nodrug:     {r1_nodrug:.3f}")
print(f"  Leakage drop:        {r1_rich - r1_nodrug:.3f}")
print(f"  Attr Jaccard (mean): {overlap_arr.mean():.3f}  ± {overlap_arr.std():.3f}")
if len(close_df) > 0:
    print(f"  Wrong-close corr:    {close_df['attr_correlation'].mean():.3f}")
print(f"\nOutputs in: outputs/track1/")
print("  case_studies/   — {0} highlighted molecule figures".format(len(selected_indices)))
print("  aggregate_attribution.png/.csv")
print("  leakage_comparison.png")
print("  leakage_jaccard_dist.png")
print("  leakage_per_pair_raw.csv")
print("  wrong_close_attr_corr.png")
print("  wrong_close_analysis_raw.csv")
print("  summary.png")
