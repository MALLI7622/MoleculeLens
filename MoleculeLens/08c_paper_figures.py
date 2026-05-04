"""
Generate three publication-quality figures for Section 7 (Mechanistic Interpretability).

Figure 1 — Molecular Lens in Action:
    4 correctly retrieved drug–text pairs, each showing a 2D molecule with
    atoms colour-coded by ECFP4 attribution (green=positive, red=negative)
    alongside a horizontal attribution bar chart.

Figure 2 — Leakage Visible at Substructure Level:
    Left: Jaccard-overlap histogram across all 435 test pairs.
    Middle: attribution bars for a same-model leakage candidate under
            text_rich (correct retrieval) vs text_nodrug (drug name removed
            at inference with the rich model held fixed).
    Right: side-by-side highlighted molecules for the same drug.

Figure 3 — Wrong-but-Close Errors are Chemically Grounded:
    Left: attribution-correlation histogram for rank-2/3 errors.
    Right: molecule pair for a representative high-correlation wrong-close
           confusion with shared bits marked.

Outputs: MoleculeLens-paper/figures/fig_molecular_lens.pdf
                                     fig_leakage_substructure.pdf
                                     fig_wrong_close.pdf
"""

import os, io, textwrap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as pe
from PIL import Image

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

os.makedirs("MoleculeLens-paper/figures", exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CMAP_ATTR = LinearSegmentedColormap.from_list("attr", ["#d62728", "#ffffff", "#2ca02c"])

# ── load everything ─────────────────────────────────────────────────────────
Wm_state = torch.load("outputs/proj_mol.pt", map_location="cpu")
Wt_state = torch.load("outputs/proj_text.pt", map_location="cpu")

Bm = torch.load("outputs/Bm_test.pt", map_location="cpu")
Bt = torch.load("outputs/Bt_test.pt", map_location="cpu")
Z_nodrug = torch.load("outputs/Z_text_test_nodrug.pt", map_location="cpu")

W_m = Wm_state["weight"]  # [256, 2048]
W_t = Wt_state["weight"]
b_t = Wt_state["bias"]
Bt_nodrug_same = F.normalize(Z_nodrug @ W_t.T + b_t, dim=1)

df     = pd.read_csv("outputs/test_df.csv")
ldf    = pd.read_csv("outputs/track1/leakage_per_pair.csv")
wdf    = pd.read_csv("outputs/track1/wrong_close_analysis.csv")

# ── core helpers ─────────────────────────────────────────────────────────────

def attr_vec(j, W_m_mat, Bt_mat):
    """Attribution vector in ECFP4 space for text query j → [2048]."""
    return W_m_mat.T @ Bt_mat[j]


def mol_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
    return mol, info


def top_bits(info, av, k=12):
    """Return top-k present bits ranked by |attribution|."""
    present = list(info.keys())
    arr = np.array(present)
    vals = av.numpy()[present]
    idx  = np.argsort(np.abs(vals))[-k:]
    return arr[idx[::-1]], vals[idx[::-1]]


def atom_colours_from_bits(mol, info, bit_list, av):
    """Build per-atom colour dict (RGB 0-1) from top attribution bits."""
    atom_score = {}
    for bit in bit_list:
        if bit not in info:
            continue
        a_val = av[bit].item()
        for (center, radius) in info[bit]:
            if radius == 0:
                atom_score[center] = atom_score.get(center, 0) + a_val
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
                for bidx in env:
                    bond = mol.GetBondWithIdx(bidx)
                    for aidx in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                        atom_score[aidx] = atom_score.get(aidx, 0) + a_val
    if not atom_score:
        return {}, {}
    vmax  = max(abs(v) for v in atom_score.values()) or 1e-8
    norm  = Normalize(vmin=-vmax, vmax=vmax)
    cmap  = plt.cm.RdYlGn
    colours = {a: cmap(norm(v))[:3] for a, v in atom_score.items()}
    return colours, atom_score


def render_mol(smiles, atom_colours, size=(340, 250)):
    """Render molecule to PIL Image with highlighted atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.padding = 0.15
    atom_list = list(atom_colours.keys())
    if atom_list:
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
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def decode_bit_label(mol, info, bit):
    """Return a short human-readable label for an ECFP4 bit.

    Format: "<prefix><element>-<neighbors> (r=<radius>)"
    where prefix is 'ar.' for aromatic or 'ring.' for non-aromatic ring atoms.
    Example: 'ar.C-CN (r=2)' means aromatic carbon bonded to C and N,
             environment captured up to radius-2 bonds.
    """
    if bit not in info or mol is None:
        return f"bit {bit}"
    center_idx, radius = info[bit][0]
    atom = mol.GetAtomWithIdx(center_idx)
    sym = atom.GetSymbol()
    is_ar = atom.GetIsAromatic()
    in_ring = atom.IsInRing()
    nbrs = sorted(set(
        mol.GetAtomWithIdx(n.GetIdx()).GetSymbol()
        for n in atom.GetNeighbors()
    ))
    nbr_str = "".join(nbrs[:3])
    prefix = "ar." if is_ar else ("ring." if in_ring else "")
    return f"{prefix}{sym}-{nbr_str} (r={radius})"


def render_bit_fragment(smiles, info, bit, size=(82, 62)):
    """Render the ECFP4 circular environment of one bit as a PIL Image.

    Re-creates mol from SMILES with explicit 2D coords so the drawer works
    correctly with RDKit 2020. Center atom = deep blue; environment = light blue.
    """
    if bit not in info:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    center_idx, radius = info[bit][0]
    env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx))
    env_atoms = {center_idx}
    for bidx in env_bonds:
        b = mol.GetBondWithIdx(bidx)
        env_atoms.add(b.GetBeginAtomIdx())
        env_atoms.add(b.GetEndAtomIdx())
    env_atoms = list(env_atoms)
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2          # must be int in RDKit 2020
        opts.addAtomIndices = False
        center_col = {center_idx: (0.10, 0.45, 0.85)}
        env_col    = {a: (0.65, 0.80, 0.95) for a in env_atoms if a != center_idx}
        drawer.DrawMolecule(mol,
            highlightAtoms=env_atoms,
            highlightAtomColors={**center_col, **env_col},
            highlightBonds=env_bonds,
            highlightBondColors={})
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Molecular Lens in Action
# 4 drugs × (molecule image | attribution bars)
# ═══════════════════════════════════════════════════════════════════════════

CASES = [
    (247, "Mafenide acetate",    "Bacterial DHPS\ninhibitor",         "sulfonamide\npharmacophore"),
    (281, "Carbamazepine",       "Sodium channel\nα-subunit blocker", "dibenzazepine\nscaffold"),
    (202, "Albuterol",           "β₂-adrenergic\nreceptor agonist",  "catecholamine\npharmacophore"),
    (222, "Acyclovir sodium",    "Herpesvirus DNA\npolymerase inh.",  "guanine base +\nacyclic sugar"),
]

K_BITS = 6   # show 6 bits — readable without crowding

fig1 = plt.figure(figsize=(14, 9))
outer = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.60, wspace=0.38)

for panel_idx, (drug_idx, name, mech, expected) in enumerate(CASES):
    row, col = divmod(panel_idx, 2)
    # Three inner columns: molecule | fragment thumbnails | attribution bars
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[row, col],
        width_ratios=[1.05, 0.72, 1.10], wspace=0.06
    )
    ax_mol  = fig1.add_subplot(inner[0])
    ax_frag = fig1.add_subplot(inner[1])
    ax_bar  = fig1.add_subplot(inner[2])

    smiles = df.iloc[drug_idx]["smiles"]
    av     = attr_vec(drug_idx, W_m, Bt)
    mol, info = mol_info(smiles)

    bits, vals = top_bits(info, av, k=K_BITS)
    atom_cols, _ = atom_colours_from_bits(mol, info, bits, av)
    img = render_mol(smiles, atom_cols, size=(310, 240))

    # ── molecule panel ────────────────────────────────────────────────
    if img:
        ax_mol.imshow(img)
    ax_mol.axis("off")
    sim_score = (Bm[drug_idx] @ Bt[drug_idx]).item()
    ax_mol.set_title(
        f"({chr(65+panel_idx)}) {name}\n{mech}",
        fontsize=8.5, fontweight="bold", pad=3, loc="left"
    )

    # ── fragment thumbnail column ─────────────────────────────────────
    # Each row shows a tiny rendering of the ECFP4 circular environment
    # (blue center atom, light-blue environment) so the reviewer sees
    # exactly which substructure the bit index refers to.
    ax_frag.axis("off")
    ax_frag.set_xlim(0, 1)
    ax_frag.set_ylim(-0.5, K_BITS - 0.5)
    ax_frag.set_title("Substructure\n(center atom = blue)",
                      fontsize=6.2, pad=3, color="#444444")

    bits_rev = bits[::-1]
    vals_rev = vals[::-1]

    for i, bit in enumerate(bits_rev):
        frag_img = render_bit_fragment(smiles, info, bit, size=(82, 60))
        y_pos = K_BITS - 1 - i  # top-to-bottom so top bar aligns with top image
        if frag_img is not None:
            img_arr = np.array(frag_img)
            # imshow with extent places the image at exact data coordinates
            ax_frag.imshow(img_arr,
                           extent=[0.04, 0.96, y_pos - 0.40, y_pos + 0.40],
                           aspect='auto', zorder=3)
            # light border around each thumbnail
            from matplotlib.patches import Rectangle
            rect = Rectangle((0.04, y_pos - 0.40), 0.92, 0.80,
                              linewidth=0.5, edgecolor="#b0b8c8",
                              facecolor="none", zorder=4)
            ax_frag.add_patch(rect)

    # ── attribution bar chart ─────────────────────────────────────────
    # Y-axis labels use decoded chemical names so the reviewer understands
    # what each bit represents without chemistry expertise:
    #   "ar.C-CN (r=2)" = aromatic Carbon bonded to C,N; environment
    #   extends 2 bonds from center. Green bar = pushes toward text match.
    colours = [CMAP_ATTR(0.85) if v > 0 else CMAP_ATTR(0.15) for v in vals_rev]
    ax_bar.barh(range(K_BITS), vals_rev, color=colours, height=0.60, edgecolor="none")
    ax_bar.axvline(0, color="#333333", lw=0.7, zorder=3)
    ax_bar.set_xlabel("Attribution score", fontsize=7)
    ax_bar.set_title(f"Top-{K_BITS} substructure attributions\n(sim={sim_score:.3f})",
                     fontsize=7.5, pad=3)

    decoded = [decode_bit_label(mol, info, b) for b in bits_rev]
    ax_bar.set_yticks(range(K_BITS))
    ax_bar.set_yticklabels(decoded, fontsize=6.2)

    # Expected pharmacophore annotation
    ax_bar.text(0.97, 0.03, f"Expected:\n{expected}",
                transform=ax_bar.transAxes, fontsize=6,
                ha="right", va="bottom", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.5))

# shared colorbar legend
cbar_ax = fig1.add_axes([0.30, 0.01, 0.40, 0.018])
sm = plt.cm.ScalarMappable(cmap=CMAP_ATTR, norm=Normalize(vmin=-1, vmax=1))
sm.set_array([])
cb = fig1.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["pushes away\nfrom text", "neutral", "pushes toward\ntext"], fontsize=6.5)
cb.ax.tick_params(size=0)
cbar_ax.set_title("ECFP4 bit attribution", fontsize=7, pad=2)


fig1.savefig("MoleculeLens-paper/figures/fig_molecular_lens.pdf",
             bbox_inches="tight", dpi=300)
fig1.savefig("MoleculeLens-paper/figures/fig_molecular_lens.png",
             bbox_inches="tight", dpi=300)
plt.close(fig1)
print("Saved: fig_molecular_lens.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Leakage Visible at Substructure Level
# Left: Jaccard histogram   Middle+Right: representative same-model leakage case
# ═══════════════════════════════════════════════════════════════════════════

# Prefer a peptide-hormone antagonist case if present; otherwise use the
# lowest-overlap same-model leakage candidate from the canonical CSV.
candidate_df = ldf[
    (ldf["attr_jaccard_overlap"] < 0.2)
    & (ldf["hit_rich"] == 1)
    & (ldf["hit_nodrug"] == 0)
].copy()
preferred_names = ["CETRORELIX ACETATE", "ITRACONAZOLE", "CEFTRIAXONE SODIUM"]
i_lk = None
for name in preferred_names:
    match = candidate_df[candidate_df["pref_name"].str.upper() == name]
    if len(match) > 0:
        i_lk = int(match.index[0])
        break
if i_lk is None:
    i_lk = int(candidate_df.sort_values("attr_jaccard_overlap").index[0])

smiles_lk  = df.iloc[i_lk]["smiles"]
name_lk    = df.iloc[i_lk]["pref_name"].title()
mech_lk    = df.iloc[i_lk]["mechanism_of_action"]
jac_lk     = ldf.iloc[i_lk]["attr_jaccard_overlap"]

av_rich   = attr_vec(i_lk, W_m, Bt)
av_nodrug = attr_vec(i_lk, W_m, Bt_nodrug_same)
mol_lk, info_lk = mol_info(smiles_lk)

bits_r,  vals_r  = top_bits(info_lk, av_rich,   k=10)
bits_nd, vals_nd = top_bits(info_lk, av_nodrug, k=10)
atom_cols_r,  _ = atom_colours_from_bits(mol_lk, info_lk, bits_r,  av_rich)
atom_cols_nd, _ = atom_colours_from_bits(mol_lk, info_lk, bits_nd, av_nodrug)
img_r  = render_mol(smiles_lk, atom_cols_r,  size=(300, 230))
img_nd = render_mol(smiles_lk, atom_cols_nd, size=(300, 230))

fig2 = plt.figure(figsize=(13, 4.5))
gs2 = gridspec.GridSpec(1, 5, figure=fig2, wspace=0.45,
                        width_ratios=[1.4, 0.05, 1.6, 0.05, 1.6])

# ── Panel A: Jaccard histogram ───────────────────────────────────────────
ax_hist = fig2.add_subplot(gs2[0])
jac_vals = ldf["attr_jaccard_overlap"].dropna().values
ax_hist.hist(jac_vals, bins=28, color="#4c78a8", edgecolor="white", lw=0.4, alpha=0.9)
ax_hist.axvline(jac_vals.mean(), color="#e45756", lw=1.5, ls="--",
                label=f"mean = {jac_vals.mean():.3f}")
ax_hist.axvspan(0, 0.2, alpha=0.12, color="#e45756", label="leakage zone (<0.2)")
n_leakage = (jac_vals < 0.2).sum()
ax_hist.text(0.10, ax_hist.get_ylim()[1]*0.88 if ax_hist.get_ylim()[1] > 0 else 30,
             f"{n_leakage}/{len(jac_vals)}\npairs", ha="center", fontsize=7,
             color="#e45756", fontweight="bold")
ax_hist.set_xlabel("Jaccard overlap of top-10\nattribution bits (rich vs nodrug)")
ax_hist.set_ylabel("# test pairs")
ax_hist.set_title("(A) Same-model attribution stability\nunder drug-name removal", fontweight="bold")
ax_hist.legend(fontsize=6.5, frameon=False)

# ── Panels B & C: attribution bars rich vs nodrug ───────────────────────
for col_idx, (bits, vals, label, correct_str, bar_col) in enumerate([
    (bits_r,  vals_r,  "text_rich\n(Rank 1 ✓ correct)", "correct",  "#2ca02c"),
    (bits_nd, vals_nd, "same-model nodrug\n(Rank 1 ✗ wrong)", "wrong", "#d62728"),
]):
    ax = fig2.add_subplot(gs2[2 + col_idx * 2])
    colours_b = [CMAP_ATTR(0.85) if v > 0 else CMAP_ATTR(0.15) for v in vals]
    ax.barh(range(len(bits)), vals[::-1], color=colours_b[::-1],
            height=0.65, edgecolor="none")
    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels([f"bit {b}" for b in bits[::-1]], fontsize=6.5)
    ax.axvline(0, color="#333333", lw=0.7, zorder=3)
    ax.set_xlabel("Attribution")
    cond_char = "B" if col_idx == 0 else "C"
    ax.set_title(f"({cond_char}) {name_lk}  —  {label}", fontweight="bold",
                 fontsize=8, color=bar_col)
    # embed molecule image
    inset = ax.inset_axes([0.55, 0.45, 0.43, 0.52])
    img_to_show = img_r if col_idx == 0 else img_nd
    if img_to_show:
        inset.imshow(img_to_show)
    inset.axis("off")
    inset.set_title("atoms coloured\nby attribution", fontsize=5.5, pad=1)

fig2.savefig("MoleculeLens-paper/figures/fig_leakage_substructure.pdf",
             bbox_inches="tight", dpi=300)
fig2.savefig("MoleculeLens-paper/figures/fig_leakage_substructure.png",
             bbox_inches="tight", dpi=300)
plt.close(fig2)
print("Saved: fig_leakage_substructure.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Wrong-but-Close Errors are Chemically Grounded
# Left: correlation histogram   Right: representative high-correlation pair
# ═══════════════════════════════════════════════════════════════════════════

fig3 = plt.figure(figsize=(12, 4.2))
gs3 = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.35, width_ratios=[1.1, 1.9])

# ── Panel A: correlation histogram ──────────────────────────────────────
ax_corr = fig3.add_subplot(gs3[0])
corr_vals = wdf["attr_correlation"].dropna().values
ax_corr.hist(corr_vals, bins=22, color="#9467bd", edgecolor="white", lw=0.4, alpha=0.9)
ax_corr.axvline(corr_vals.mean(), color="#e45756", lw=1.5, ls="--",
                label=f"mean = {corr_vals.mean():.3f}")
ax_corr.axvspan(0.5, 1.0, alpha=0.10, color="#2ca02c",
                label=f"chemically reasonable\n(ρ > 0.5):  {(corr_vals>0.5).sum()}/{len(corr_vals)}")
ax_corr.set_xlabel("Attribution correlation\n(correct drug vs retrieved drug)")
ax_corr.set_ylabel("# rank-2/3 error pairs")
ax_corr.set_title("(A) Are rank-2/3 errors\nchemically reasonable?", fontweight="bold")
ax_corr.legend(fontsize=6.5, frameon=False)

# ── Panel B: representative high-correlation pair ───────────────────────
wrow = wdf.nlargest(1, "attr_correlation").iloc[0]

i_c  = wrow["drug_idx"]
i_r  = wrow["retrieved_idx"]
s_c  = df.iloc[i_c]["smiles"]
s_r  = df.iloc[i_r]["smiles"]
n_c  = df.iloc[i_c]["pref_name"].title()
n_r  = df.iloc[i_r]["pref_name"].title()
corr_val = wrow["attr_correlation"]
same_moa = wrow["moa_correct"] == wrow["moa_retrieved"]

av_c = attr_vec(i_c, W_m, Bt)
av_r = attr_vec(i_r, W_m, Bt)
mol_c, info_c = mol_info(s_c)
mol_r, info_r = mol_info(s_r)
shared_bits   = set(info_c.keys()) & set(info_r.keys())

# colour shared bits by attribution on each drug
ac_c, _ = atom_colours_from_bits(mol_c, info_c, list(shared_bits)[:12], av_c)
ac_r, _ = atom_colours_from_bits(mol_r, info_r, list(shared_bits)[:12], av_r)
img_c   = render_mol(s_c, ac_c, size=(290, 220))
img_r   = render_mol(s_r, ac_r, size=(290, 220))

inner3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs3[1],
                                          width_ratios=[1, 0.18, 1], wspace=0.0)
ax_c   = fig3.add_subplot(inner3[0])
ax_mid = fig3.add_subplot(inner3[1])
ax_r   = fig3.add_subplot(inner3[2])

for ax, img, label, rank_label in [
    (ax_c, img_c, n_c, "correct (rank 2)"),
    (ax_r, img_r, n_r, "retrieved (rank 1)"),
]:
    if img:
        ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{label}\n({rank_label})", fontsize=8, fontweight="bold",
                 pad=3)

# central annotation
ax_mid.axis("off")
ax_mid.text(0.5, 0.55, f"ρ = {corr_val:.2f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2ca02c",
            transform=ax_mid.transAxes)
ax_mid.text(0.5, 0.38, f"{len(shared_bits)}\nshared\nbits", ha="center", va="center",
            fontsize=7, color="#555555", transform=ax_mid.transAxes)
ax_mid.text(0.5, 0.18, "same\nMOA" if same_moa else "diff.\nMOA",
            ha="center", va="center", fontsize=7,
            color="#2ca02c" if same_moa else "#e45756",
            transform=ax_mid.transAxes)

fig3.savefig("MoleculeLens-paper/figures/fig_wrong_close.pdf",
             bbox_inches="tight", dpi=300)
fig3.savefig("MoleculeLens-paper/figures/fig_wrong_close.png",
             bbox_inches="tight", dpi=300)
plt.close(fig3)
print("Saved: fig_wrong_close.pdf/.png")

print("\nAll three paper figures saved to MoleculeLens-paper/figures/")
