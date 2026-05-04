"""
MoleculeLens architecture diagram — compact horizontal layout for NeurIPS.
Two-row left-to-right flow (molecule top, text bottom) converging at shared
embedding space on the right.
"""

import os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

os.makedirs("MoleculeLens-paper/figures", exist_ok=True)

# ── Canvas ────────────────────────────────────────────────────────────────
FW, FH = 15.0, 6.2
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Palette ───────────────────────────────────────────────────────────────
C = dict(
    input  = "#FFCDD2",
    frozen = "#B3E5FC",
    vec    = "#EEEEEE",
    proj   = "#FFF9C4",
    norm   = "#C8E6C9",
    embed  = "#B2EBF2",
    shared = "#E1BEE7",
    loss   = "#FFCCBC",
    attr   = "#FFE0B2",
    border = "#37474F",
    fz_    = "#01579B",
    tr_    = "#BF360C",
    attr_  = "#E65100",
    lens_  = "#1B5E20",
    grp    = "#FFFDE7",
)

# ── Geometry ──────────────────────────────────────────────────────────────
ROW_MOL = 4.50
ROW_TXT = 1.95
BH      = 0.74   # box height

# Column x-centres  (content starts at x=1.90 to leave left-label room)
X_IN   = 2.10
X_ENC  = 4.20
X_VEC  = 6.30
X_PROJ = 8.20
X_NORM = 10.10
X_EMB  = 11.80
X_SHR  = 13.55
X_LOSS = 13.55

BW_IN  = 1.72
BW_ENC = 2.10
BW_VEC = 1.72
BW_PRJ = 1.95
BW_NRM = 1.42
BW_EMB = 1.85
BW_SHR = 2.30
BW_LOS = 2.10

# ── Helpers ───────────────────────────────────────────────────────────────

def rbox(cx, cy, w, h, fc, label, sub="", fs=8.5, lw=1.3,
         ec=C["border"], bold=False, tc="#1C1C1C"):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.07",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, zorder=3)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + 0.13, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4)
        ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                fontsize=fs - 1.6, color="#616161", zorder=4, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4)
    return cx + w/2, cx - w/2, cy + h/2, cy - h/2

def arrow(x0, y0, x1, y1, clr=C["border"], lw=1.5, ls="-", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=clr, lw=lw,
                                mutation_scale=11, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def badge(cx, cy, txt, fc, tc, fs=6.0):
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", fc=fc, ec=tc, lw=0.8))

def dlabel(x, y, txt):
    ax.text(x, y, txt, ha="center", va="center", fontsize=6.0,
            color="#9E9E9E", style="italic", zorder=6)

# ═══════════════════════════════════════════════════════════════════════
# ROW LABELS  (horizontal, left margin)
# ═══════════════════════════════════════════════════════════════════════
for y, lbl, clr in [(ROW_MOL, "Molecule\nBranch", "#0277BD"),
                    (ROW_TXT, "Text\nBranch",      "#1565C0")]:
    ax.text(0.18, y, lbl, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            multialignment="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.30", fc=clr, ec=clr, lw=0))

# ═══════════════════════════════════════════════════════════════════════
# MOLECULE ROW  (top)
# ═══════════════════════════════════════════════════════════════════════
r,l,t,b = rbox(X_IN,   ROW_MOL, BW_IN,  BH, C["input"],
               "Drug SMILES\n/ InChI", bold=True, fs=8.0)
badge(r - 0.30, t - 0.19, "INPUT", C["input"], C["border"])
MIN_r = r

r,l,t,b = rbox(X_ENC,  ROW_MOL, BW_ENC, BH, C["frozen"],
               "RDKit ECFP4", sub="radius=2, 2048-bit", fs=8.5)
badge(r - 0.40, t - 0.19, "❄ frozen", C["frozen"], C["fz_"])
MEN_r = r

r,l,t,b = rbox(X_VEC,  ROW_MOL, BW_VEC, BH, C["vec"],
               r"$x_i \in \mathbb{R}^{2048}$", fs=8.2)
MVE_r = r

r,l,t,b = rbox(X_PROJ, ROW_MOL, BW_PRJ, BH, C["proj"],
               r"$W_m$: Linear", sub="2048→256", bold=True, fs=8.5)
MPR_r, MPR_l, MPR_t = r, l, t

r,l,t,b = rbox(X_NORM, ROW_MOL, BW_NRM, BH, C["norm"],
               r"$\ell_2$ Norm", fs=8.5)
MNO_r = r

r,l,t,b = rbox(X_EMB,  ROW_MOL, BW_EMB, BH, C["embed"],
               r"$\hat{b}_i^{(m)} \in \mathbb{R}^{256}$",
               sub="unit-norm emb.", bold=True, fs=8.5)
MEM_r, MEM_cx, MEM_t, MEM_b = r, X_EMB, t, b

# ═══════════════════════════════════════════════════════════════════════
# TEXT ROW  (bottom)
# ═══════════════════════════════════════════════════════════════════════
r,l,t,b = rbox(X_IN,   ROW_TXT, BW_IN,  BH, C["input"],
               "Mechanism\nText", bold=True, fs=8.0)
badge(r - 0.30, t - 0.19, "INPUT", C["input"], C["border"])
TIN_r = r

r,l,t,b = rbox(X_ENC,  ROW_TXT, BW_ENC, BH + 0.12, C["frozen"],
               "S-Biomed-RoBERTa", sub="12 layers | 768-d", fs=8.5)
badge(r - 0.40, t - 0.19, "❄ frozen", C["frozen"], C["fz_"])
TEN_r, TEN_l, TEN_t, TEN_b = r, l, t, b

r,l,t,b = rbox(X_VEC,  ROW_TXT, BW_VEC, BH, C["vec"],
               r"$z_i \in \mathbb{R}^{768}$", sub="CLS pool", fs=8.2)
TVE_r = r

r,l,t,b = rbox(X_PROJ, ROW_TXT, BW_PRJ, BH, C["proj"],
               r"$W_t$: Linear", sub="768→256", bold=True, fs=8.5)
TPR_r = r

r,l,t,b = rbox(X_NORM, ROW_TXT, BW_NRM, BH, C["norm"],
               r"$\ell_2$ Norm", fs=8.5)
TNO_r = r

r,l,t,b = rbox(X_EMB,  ROW_TXT, BW_EMB, BH, C["embed"],
               r"$\hat{b}_i^{(t)} \in \mathbb{R}^{256}$",
               sub="unit-norm emb.", bold=True, fs=8.5)
TEM_r, TEM_cx, TEM_t, TEM_b = r, X_EMB, t, b

# ═══════════════════════════════════════════════════════════════════════
# SHARED SPACE  +  LOSS
# ═══════════════════════════════════════════════════════════════════════
SHR_cy = (ROW_MOL + ROW_TXT) / 2
r,l,t,b = rbox(X_SHR, SHR_cy, BW_SHR, 1.10, C["shared"],
               r"Shared $256$-d Space",
               sub=r"$s_{ij}=\hat{b}_i^{(m)}\!\cdot\!\hat{b}_j^{(t)}$",
               bold=True, fs=9.0)
SHR_t, SHR_b, SHR_l = t, b, l

LOSS_cy = SHR_cy + 1.65
r,l,t,b = rbox(X_LOSS, LOSS_cy, BW_LOS, 0.84, C["loss"],
               "Symmetric InfoNCE",
               sub="+ same-target wt. + margin",
               bold=True, fs=8.5)
LOS_b = b

# ═══════════════════════════════════════════════════════════════════════
# TRAINABLE GROUP OUTLINE
# ═══════════════════════════════════════════════════════════════════════
gx0 = X_PROJ - BW_PRJ/2 - 0.14
gx1 = X_NORM + BW_NRM/2 + 0.14
gy0 = ROW_TXT - BH/2 - 0.18
gy1 = ROW_MOL + BH/2 + 0.18
grp = FancyBboxPatch((gx0, gy0), gx1-gx0, gy1-gy0,
                     boxstyle="round,pad=0.10",
                     facecolor=C["grp"], edgecolor="#F57F17",
                     linewidth=1.1, linestyle="--", zorder=1, alpha=0.55)
ax.add_patch(grp)
ax.text(gx0 + 0.12, gy1 - 0.20, "trainable only",
        fontsize=6.5, color="#E65100", fontweight="bold", zorder=5)

# ═══════════════════════════════════════════════════════════════════════
# ARROWS — molecule row
# ═══════════════════════════════════════════════════════════════════════
GAP = 0.06
for x0, x1, y in [
    (MIN_r, X_ENC - BW_ENC/2, ROW_MOL),
    (MEN_r, X_VEC - BW_VEC/2, ROW_MOL),
    (MVE_r, X_PROJ- BW_PRJ/2, ROW_MOL),
    (MPR_r, X_NORM- BW_NRM/2, ROW_MOL),
    (MNO_r, X_EMB - BW_EMB/2, ROW_MOL),
]:
    arrow(x0+GAP, y, x1-GAP, y)

for x, y, txt in [
    ((MEN_r + X_VEC - BW_VEC/2)/2, ROW_MOL + 0.32, "2048-d"),
    ((MVE_r + X_PROJ- BW_PRJ/2)/2, ROW_MOL + 0.32, "2048-d"),
    ((MPR_r + X_NORM- BW_NRM/2)/2, ROW_MOL + 0.32, "256-d"),
    ((MNO_r + X_EMB - BW_EMB/2)/2, ROW_MOL + 0.32, "256-d"),
]:
    dlabel(x, y, txt)

# ── text row ──────────────────────────────────────────────────────────
for x0, x1, y in [
    (TIN_r, X_ENC - BW_ENC/2, ROW_TXT),
    (TEN_r, X_VEC - BW_VEC/2, ROW_TXT),
    (TVE_r, X_PROJ- BW_PRJ/2, ROW_TXT),
    (TPR_r, X_NORM- BW_NRM/2, ROW_TXT),
    (TNO_r, X_EMB - BW_EMB/2, ROW_TXT),
]:
    arrow(x0+GAP, y, x1-GAP, y)

for x, y, txt in [
    ((TEN_r + X_VEC - BW_VEC/2)/2, ROW_TXT - 0.32, "768-d"),
    ((TVE_r + X_PROJ- BW_PRJ/2)/2, ROW_TXT - 0.32, "768-d"),
    ((TPR_r + X_NORM- BW_NRM/2)/2, ROW_TXT - 0.32, "256-d"),
    ((TNO_r + X_EMB - BW_EMB/2)/2, ROW_TXT - 0.32, "256-d"),
]:
    dlabel(x, y, txt)

# ── converge to shared space ──────────────────────────────────────────
arrow(MEM_r+GAP, ROW_MOL, SHR_l-GAP, SHR_cy+0.30, rad=-0.22, lw=1.8)
arrow(TEM_r+GAP, ROW_TXT, SHR_l-GAP, SHR_cy-0.30, rad= 0.22, lw=1.8)

# ── shared → loss ────────────────────────────────────────────────────
arrow(X_SHR, SHR_t+GAP, X_LOSS, LOS_b-GAP, lw=1.8)

# ═══════════════════════════════════════════════════════════════════════
# ATTRIBUTION BACK-PATH  (arched annotation above molecule row)
# ═══════════════════════════════════════════════════════════════════════
AY = ROW_MOL + 1.22  # horizontal guideline above molecule row
arc_x0 = X_PROJ      # start above W_m
arc_x1 = X_SHR - BW_SHR/2  # end at left edge of shared space

# dashed horizontal line
ax.plot([arc_x0, arc_x1], [AY, AY], color=C["attr_"],
        lw=1.1, ls="--", zorder=4)
# down-arrow to W_m
ax.annotate("", xy=(arc_x0, MPR_t + 0.04),
            xytext=(arc_x0, AY - 0.06),
            arrowprops=dict(arrowstyle="-|>", color=C["attr_"], lw=1.1,
                            mutation_scale=9, linestyle="dashed"), zorder=5)
# left-arrow head at shared side
ax.annotate("", xy=(arc_x1 + 0.04, AY),
            xytext=(arc_x1 + 0.30, AY),
            arrowprops=dict(arrowstyle="<|-", color=C["attr_"], lw=1.1,
                            mutation_scale=9, linestyle="dashed"), zorder=5)

# formula text
ax.text((arc_x0 + arc_x1) / 2, AY + 0.06,
        r"$\mathrm{attr}_k = [W_m^\top\,\hat{b}_j^{(t)}]_k$"
        r"  — closed-form ECFP4 saliency",
        ha="center", va="bottom", fontsize=7.8, color=C["attr_"],
        fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.22", fc=C["attr"],
                  ec=C["attr_"], lw=0.9, alpha=0.92))

# ═══════════════════════════════════════════════════════════════════════
# LOGIT LENS BRACKET  (below text encoder, pointing up to W_t)
# ═══════════════════════════════════════════════════════════════════════
LL_y  = ROW_TXT - 1.05   # y of annotation line below text row
ll_x0 = TEN_l + 0.08
ll_x1 = X_PROJ - BW_PRJ/2

# horizontal dashed line
ax.plot([ll_x0, ll_x1], [LL_y, LL_y], color=C["lens_"],
        lw=1.1, ls="--", zorder=4)
# up-arrow from encoder bottom
ax.annotate("", xy=(ll_x0, TEN_b - 0.04),
            xytext=(ll_x0, LL_y + 0.06),
            arrowprops=dict(arrowstyle="-|>", color=C["lens_"], lw=1.1,
                            mutation_scale=9, linestyle="dashed"), zorder=5)
# right-arrow head toward W_t
ax.annotate("", xy=(ll_x1 - 0.04, LL_y),
            xytext=(ll_x1 - 0.30, LL_y),
            arrowprops=dict(arrowstyle="-|>", color=C["lens_"], lw=1.1,
                            mutation_scale=9, linestyle="dashed"), zorder=5)

# label
ax.text((ll_x0 + ll_x1) / 2, LL_y - 0.06,
        r"logit lens: apply $W_t$ to CLS at each layer $\ell$",
        ha="center", va="top", fontsize=7.5, color=C["lens_"],
        fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.20", fc="#F1F8E9",
                  ec=C["lens_"], lw=0.8, alpha=0.92))

# ═══════════════════════════════════════════════════════════════════════
# COMPACT INLINE LEGEND  (bottom strip)
# ═══════════════════════════════════════════════════════════════════════
LGND = [
    (C["input"],  "Input data"),
    (C["frozen"], "Frozen encoder"),
    (C["vec"],    "Intermediate"),
    (C["proj"],   "Trainable proj."),
    (C["norm"],   r"$\ell_2$ Norm"),
    (C["embed"],  "Unit-norm emb."),
    (C["shared"], "Shared space"),
    (C["loss"],   "InfoNCE loss"),
    (C["attr"],   "Saliency path"),
    ("#F1F8E9",   "Logit lens"),
]
LY = 0.30
sep_y = LY + 0.42
ax.plot([0.35, FW - 0.35], [sep_y, sep_y], color="#BDBDBD", lw=0.6)
item_w = FW / len(LGND)
for k, (col, lbl) in enumerate(LGND):
    bx = item_w * k + 0.50
    by = LY
    ec_col = C["lens_"] if "ogit" in lbl else (C["attr_"] if "aliency" in lbl else C["border"])
    ax.add_patch(FancyBboxPatch((bx, by - 0.12), 0.26, 0.24,
                                boxstyle="round,pad=0.02",
                                fc=col, ec=ec_col, lw=0.6, zorder=5))
    ax.text(bx + 0.35, by + 0.01, lbl, ha="left", va="center",
            fontsize=6.6, color="#333333", zorder=6)

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
fig.tight_layout(pad=0.4)
fig.savefig("MoleculeLens-paper/figures/fig_architecture.pdf",
            bbox_inches="tight", format="pdf")
fig.savefig("MoleculeLens-paper/figures/fig_architecture.png",
            bbox_inches="tight", dpi=220)
shutil.copy("MoleculeLens-paper/figures/fig_architecture.pdf",
            "/home/cheriearjun/figures/fig_architecture.pdf")
shutil.copy("MoleculeLens-paper/figures/fig_architecture.png",
            "/home/cheriearjun/figures/fig_architecture.png")
plt.close(fig)
print("Saved: fig_architecture.pdf + fig_architecture.png")
