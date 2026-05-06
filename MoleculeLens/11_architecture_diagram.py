"""Generate a polished MoleculeLens strategy/architecture figure for the paper.

The current paper figure is built from this script.  The renderer below uses a
compact narrative layout inspired by clean architecture diagrams: paired inputs,
frozen encoders, small trainable bridges, shared retrieval space, and explanation
outputs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.patheffects as pe


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = SCRIPT_DIR / "MoleculeLens-paper" / "figures"
ROOT_FIG_DIR = REPO_ROOT / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)


COL = {
    "ink": "#16212f",
    "muted": "#5d6877",
    "line": "#435166",
    "soft_line": "#cbd5e1",
    "paper": "#ffffff",
    "panel": "#f7fafc",
    "mol": "#47b56b",
    "mol_dark": "#16843a",
    "text": "#5aa9e6",
    "text_dark": "#1f76b5",
    "frozen": "#e8f3fb",
    "bridge": "#fff1bf",
    "bridge_edge": "#d7a309",
    "shared": "#f0e8ff",
    "shared_edge": "#8a63d2",
    "loss": "#ffe2d6",
    "loss_edge": "#dd6b45",
    "sal": "#ffefcf",
    "lens": "#e9f8eb",
    "red": "#e96262",
    "green": "#65bb68",
    "yellow": "#f2c94c",
}


def shadow_patch(patch):
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.4, -1.4), shadow_rgbFace=(0, 0, 0), alpha=0.12),
        pe.Normal(),
    ])
    return patch


def rbox(ax, x, y, w, h, label="", sub="", fc="#fff", ec=None, lw=1.15,
         radius=0.13, fs=8.2, bold=False, color=None, z=3):
    ec = ec or COL["soft_line"]
    color = color or COL["ink"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(shadow_patch(patch))
    if label:
        ax.text(
            x + w / 2,
            y + h * (0.58 if sub else 0.50),
            label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=color,
            zorder=z + 2,
            linespacing=1.05,
        )
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            sub,
            ha="center",
            va="center",
            fontsize=fs - 1.6,
            color=COL["muted"],
            zorder=z + 2,
            linespacing=1.05,
        )
    return patch


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.35, rad=0.0, style="-|>",
          ls="-", z=6, ms=10):
    color = color or COL["line"]
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def stage_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=12.2,
            fontweight="bold", color=COL["ink"], zorder=10)


def badge(ax, x, y, text, fc, ec, tc=None, fs=6.6):
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fs,
        fontweight="bold",
        color=tc or ec,
        zorder=11,
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08", fc=fc, ec=ec, lw=0.9),
    )


def lock_icon(ax, x, y, scale=1.0, color=None):
    color = color or COL["text_dark"]
    body_w, body_h = 0.11 * scale, 0.085 * scale
    body = FancyBboxPatch(
        (x - body_w / 2, y - body_h / 2 - 0.015 * scale),
        body_w,
        body_h,
        boxstyle=f"round,pad=0.004,rounding_size={0.018 * scale}",
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=0.8,
        zorder=12,
    )
    ax.add_patch(body)
    ax.add_patch(Arc((x, y + 0.012 * scale), 0.10 * scale, 0.11 * scale,
                     theta1=0, theta2=180, color=color, lw=0.8, zorder=12))


def document_card(ax, x, y, w, h, title, lines, accent):
    rbox(ax, x, y, w, h, fc="#ffffff", ec=accent, lw=1.25, radius=0.12)
    ax.text(x + 0.16, y + h - 0.18, title, ha="left", va="top",
            fontsize=8.8, fontweight="bold", color=accent, zorder=8)
    ty = y + h - 0.45
    for line in lines:
        ax.text(x + 0.16, ty, line, ha="left", va="top", fontsize=7.8,
                color=COL["ink"], zorder=8)
        ty -= 0.25


def token_stack(ax, x, y, colors, w=0.32, h=0.34, dx=0.24, dy=0.08):
    for layer in range(3):
        off = layer * dy
        alpha = 0.34 + 0.18 * layer
        for i, c in enumerate(colors):
            patch = FancyBboxPatch(
                (x + i * dx + off, y + off), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=c,
                edgecolor=c,
                linewidth=0.8,
                alpha=alpha,
                zorder=3 + layer,
            )
            ax.add_patch(patch)


ARCH_MOL_SMILES = "NC(=O)N1c2ccccc2C=Cc2ccccc21"


def trim_molecule_image(img, pad=6):
    try:
        from PIL import Image, ImageChops
    except Exception:
        return img
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bbox = ImageChops.difference(rgba, white).getbbox()
    if bbox is None:
        return rgba
    left, upper, right, lower = bbox
    left = max(left - pad, 0)
    upper = max(upper - pad, 0)
    right = min(right + pad, rgba.width)
    lower = min(lower + pad, rgba.height)
    cropped = rgba.crop((left, upper, right, lower))
    transparent = []
    for r, g, b, a in cropped.getdata():
        if r > 245 and g > 245 and b > 245:
            transparent.append((255, 255, 255, 0))
        else:
            transparent.append((18, 24, 32, 255))
    cropped.putdata(transparent)
    return cropped


def rdkit_molecule_image(smiles=ARCH_MOL_SMILES, size=(520, 240)):
    try:
        import io
        from PIL import Image
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = 0.08
    opts.bondLineWidth = 2
    opts.addAtomIndices = False
    if hasattr(opts, "useBWAtomPalette"):
        opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return trim_molecule_image(Image.open(io.BytesIO(drawer.GetDrawingText())))


def molecule_icon(ax, cx, cy, scale=1.0):
    pts = [
        (-0.55, 0.02), (-0.28, 0.18), (0.00, 0.05),
        (0.27, 0.22), (0.54, 0.05), (0.23, -0.18), (-0.08, -0.15),
    ]
    pts = [(cx + px * scale, cy + py * scale) for px, py in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#475569", lw=1.4,
                solid_capstyle="round", zorder=9)
    for i, (px, py) in enumerate(pts):
        col = [COL["green"], "#11a579", "#111827", COL["yellow"], COL["green"], "#f97316", "#2563eb"][i]
        ax.add_patch(Circle((px, py), 0.055 * scale, fc=col, ec="white", lw=0.5, zorder=10))


def draw_structure(ax, x0, y0, x1, y1, z=9):
    img = rdkit_molecule_image()
    if img is not None:
        ax.imshow(img, extent=(x0, x1, y0, y1), zorder=z, aspect="auto")
    else:
        molecule_icon(ax, (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) * 1.4)


def drug_structure_card(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#ffffff", ec=COL["mol_dark"], lw=1.25, radius=0.12)
    ax.text(x + 0.16, y + h - 0.18, "Drug structure", ha="left", va="top",
            fontsize=8.8, fontweight="bold", color=COL["mol_dark"], zorder=8)
    ax.text(x + 0.16, y + h - 0.42, "SMILES -> ECFP4", ha="left", va="top",
            fontsize=7.8, color=COL["ink"], zorder=8)
    draw_structure(ax, x + 0.30, y + 0.04, x + w - 0.13, y + 0.57)


def mini_bar_chart(ax, x, y, w, h):
    vals = [0.92, 0.72, 0.55, -0.30, 0.42]
    labels = ["508", "193", "759", "1204", "1178"]
    zero = x + w * 0.36
    ax.plot([zero, zero], [y + 0.02, y + h - 0.02],
            color="#334155", lw=0.9, zorder=8)
    for i, val in enumerate(vals):
        yy = y + h - 0.08 - i * (h - 0.12) / len(vals)
        bar_h = 0.052
        if val >= 0:
            ax.add_patch(FancyBboxPatch((zero, yy - bar_h / 2), val * w * 0.52, bar_h,
                                         boxstyle="round,pad=0.006,rounding_size=0.015",
                                         fc=COL["green"], ec="none", zorder=8))
        else:
            ax.add_patch(FancyBboxPatch((zero + val * w * 0.52, yy - bar_h / 2),
                                         -val * w * 0.52, bar_h,
                                         boxstyle="round,pad=0.006,rounding_size=0.015",
                                         fc=COL["red"], ec="none", zorder=8))
        ax.text(x, yy, f"bit {labels[i]}", ha="left", va="center",
                fontsize=6.4, color=COL["muted"], zorder=9)


def mini_line_chart(ax, x, y, w, h):
    for frac in [0.33, 0.66]:
        ax.plot([x, x + w], [y + frac * h, y + frac * h],
                color="#d7e1ea", lw=0.55, zorder=7)
    ax.plot([x, x], [y, y + h], color="#7d8a99", lw=0.95, zorder=8)
    ax.plot([x, x + w], [y, y], color="#7d8a99", lw=0.95, zorder=8)
    xs = [x + 0.10 * w, x + 0.30 * w, x + 0.50 * w, x + 0.70 * w, x + 0.92 * w]
    rich = [y + 0.08 * h, y + 0.09 * h, y + 0.11 * h, y + 0.22 * h, y + 0.92 * h]
    nodrug = [y + 0.06 * h, y + 0.07 * h, y + 0.09 * h, y + 0.17 * h, y + 0.55 * h]
    ax.plot(xs, rich, color="#2f73c9", lw=1.55, marker="o", ms=2.8, zorder=9)
    ax.plot(xs, nodrug, color="#e15d4f", lw=1.25, ls="--", marker="s", ms=2.5, zorder=9)
    ax.text(x + w * 0.60, y + h * 0.84, "layer 12", fontsize=6.2,
            color=COL["muted"], zorder=9)


def mini_retrieve_list(ax, x, y, w, h):
    rows = [0.92, 0.78, 0.88]
    for i, width_frac in enumerate(rows):
        yy = y + h - 0.10 - i * 0.14
        ax.add_patch(FancyBboxPatch((x, yy - 0.035), w * width_frac, 0.07,
                                    boxstyle="round,pad=0.012,rounding_size=0.025",
                                    fc="#e7edf3", ec="none", zorder=8))
        ax.add_patch(FancyBboxPatch((x, yy - 0.035), w * 0.18, 0.07,
                                    boxstyle="round,pad=0.012,rounding_size=0.025",
                                    fc="#cfdbe6", ec="none", zorder=9))


def render():
    fig, ax = plt.subplots(figsize=(14.8, 5.00))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 5.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Subtle background bands make the flow readable after NeurIPS downscaling.
    for x, w, fc in [
        (0.22, 2.35, "#fbfdff"),
        (2.80, 2.55, "#f8fbfe"),
        (5.70, 2.20, "#fffdfa"),
        (8.15, 3.05, "#fbf8ff"),
        (11.55, 2.70, "#fcfdfb"),
    ]:
        ax.add_patch(FancyBboxPatch((x, 0.58), w, 4.42,
                                    boxstyle="round,pad=0.03,rounding_size=0.18",
                                    fc=fc, ec="#eef2f7", lw=0.7, zorder=0))

    stage_label(ax, 1.38, 4.92, "Paired Dataset")
    stage_label(ax, 4.12, 4.92, "Frozen Encoders")
    stage_label(ax, 6.76, 4.92, "Trainable Bridges")
    stage_label(ax, 9.92, 4.92, "Shared Retrieval Space")
    stage_label(ax, 12.92, 4.92, "Audit Outputs")

    # Inputs.
    drug_structure_card(ax, 0.55, 3.18, 1.88, 1.15)
    document_card(ax, 0.55, 1.48, 1.88, 1.10, "Mechanism text",
                  ["sodium channel", "beta-2 agonist", "DNA polymerase"], COL["text_dark"])
    ax.plot([1.49, 1.49], [2.67, 3.14], color=COL["soft_line"], lw=1.3, zorder=2)
    ax.text(1.49, 2.91, "matched pair", ha="center", va="center",
            fontsize=6.6, color=COL["muted"], zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#e2e8f0", lw=0.7))

    # Frozen encoders.
    rbox(ax, 3.28, 3.43, 1.76, 0.85, "ECFP4 / RDKit", "2048-bit fingerprint",
         fc=COL["frozen"], ec=COL["text"], fs=8.7, bold=True)
    badge(ax, 4.72, 4.18, "frozen", "#ffffff", COL["text_dark"], fs=6.2)
    lock_icon(ax, 4.38, 4.18, scale=0.70, color=COL["text_dark"])
    rbox(ax, 3.28, 1.60, 1.76, 0.95, "S-Biomed-RoBERTa", "CLS from each layer",
         fc=COL["frozen"], ec=COL["text"], fs=8.5, bold=True)
    badge(ax, 4.72, 2.43, "frozen", "#ffffff", COL["text_dark"], fs=6.2)
    lock_icon(ax, 4.38, 2.43, scale=0.70, color=COL["text_dark"])

    # Embedding/token stacks.
    token_stack(ax, 5.48, 3.55, [COL["green"], COL["green"], "#f5c15d", COL["green"], COL["green"]])
    token_stack(ax, 5.48, 1.78, [COL["text"], COL["text"], COL["text"], "#f6a6a6", COL["text"]])
    ax.text(6.08, 4.28, "molecule bits", ha="center", va="center", fontsize=6.7, color=COL["muted"])
    ax.text(6.08, 4.12, "feature map", ha="center", va="center",
            fontsize=5.6, color=COL["muted"], style="italic")
    ax.text(6.09, 2.51, "text states", ha="center", va="center", fontsize=6.7, color=COL["muted"])
    ax.text(6.09, 2.35, "feature map", ha="center", va="center",
            fontsize=5.6, color=COL["muted"], style="italic")

    # Trainable bridges.
    trainable_group = FancyBboxPatch(
        (6.72, 1.43), 1.08, 2.90,
        boxstyle="round,pad=0.045,rounding_size=0.12",
        fc="#fff8d8", ec=COL["bridge_edge"], lw=0.9,
        linestyle="--", alpha=0.55, zorder=2,
    )
    ax.add_patch(trainable_group)
    ax.text(7.26, 4.45, "trainable\nprojection heads",
            ha="center", va="center", fontsize=6.4,
            color="#9a6700", fontweight="bold", linespacing=0.9, zorder=8)
    rbox(ax, 6.88, 3.42, 0.86, 0.82, r"$W_m$", "2048 -> 256",
         fc=COL["bridge"], ec=COL["bridge_edge"], fs=9.5, bold=True)
    rbox(ax, 6.88, 1.62, 0.86, 0.82, r"$W_t$", "768 -> 256",
         fc=COL["bridge"], ec=COL["bridge_edge"], fs=9.5, bold=True)

    # Unit vectors and shared space.
    rbox(ax, 7.98, 3.46, 0.86, 0.58, r"$\hat b^{(m)}$", "unit",
         fc="#e9fbef", ec=COL["mol"], fs=7.8, bold=True)
    rbox(ax, 7.98, 1.66, 0.86, 0.58, r"$\hat b^{(t)}$", "unit",
         fc="#eaf5ff", ec=COL["text"], fs=7.8, bold=True)

    # Keep normalized vectors outside the retrieval space, then enter the
    # geometric space as a separate zone.
    shared_x, shared_y, shared_w, shared_h = 9.12, 1.50, 1.86, 2.58
    shared_cx = shared_x + shared_w / 2
    shared_right = shared_x + shared_w
    shared = FancyBboxPatch((shared_x, shared_y), shared_w, shared_h,
                            boxstyle="round,pad=0.045,rounding_size=0.18",
                            fc="#f8f4ff", ec=COL["shared_edge"], lw=1.2, zorder=3)
    ax.add_patch(shadow_patch(shared))
    ax.text(shared_cx, 3.78, "256-d joint space", ha="center", va="center",
            fontsize=8.4, fontweight="bold", color=COL["ink"], zorder=9)
    ax.text(shared_cx, 3.50, "cosine / dot-product retrieval", ha="center", va="center",
            fontsize=6.1, color=COL["muted"], zorder=9)
    legend_y = 3.31
    for x0, color, label in [
        (shared_x + 0.26, COL["mol"], "molecule"),
        (shared_x + 0.88, COL["text"], "text"),
        (shared_x + 1.30, COL["yellow"], "anchor"),
    ]:
        ax.add_patch(Circle((x0, legend_y), 0.045, fc=color, ec="white", lw=0.8, zorder=9))
        ax.text(x0 + 0.08, legend_y, label, ha="left", va="center",
                fontsize=5.7, color=COL["muted"], zorder=9)

    inner = FancyBboxPatch((shared_x + 0.18, 1.78), shared_w - 0.36, 1.42,
                           boxstyle="round,pad=0.02,rounding_size=0.10",
                           fc="#ffffff", ec="#e3d7f4", lw=0.55, alpha=0.62, zorder=4)
    ax.add_patch(inner)
    for y_grid in [2.15, 2.55, 2.95]:
        ax.plot([shared_x + 0.27, shared_right - 0.27], [y_grid, y_grid],
                color="#d9ccef", lw=0.55, alpha=0.50, zorder=5)
    for x_grid in [shared_cx - 0.56, shared_cx, shared_cx + 0.56]:
        ax.plot([x_grid, x_grid], [1.88, 3.10],
                color="#d9ccef", lw=0.55, alpha=0.50, zorder=5)
    ax.plot([shared_x + 0.32, shared_right - 0.30], [2.08, 2.08],
            color="#c8b6e6", lw=0.75, alpha=0.70, zorder=5)
    ax.plot([shared_x + 0.32, shared_x + 0.32], [1.92, 3.07],
            color="#c8b6e6", lw=0.75, alpha=0.70, zorder=5)
    dots = [
        (shared_cx - 0.58, 2.36, COL["mol"], 0.115),
        (shared_cx - 0.12, 2.52, COL["mol"], 0.115),
        (shared_cx + 0.50, 2.34, COL["mol"], 0.115),
        (shared_cx - 0.48, 2.96, COL["text"], 0.115),
        (shared_cx + 0.02, 3.12, COL["text"], 0.115),
        (shared_cx + 0.58, 2.88, COL["text"], 0.115),
        (shared_cx - 0.02, 2.72, COL["yellow"], 0.135),
    ]
    for x0, y0, x1, y1 in [
        (shared_cx - 0.58, 2.36, shared_cx - 0.48, 2.96),
        (shared_cx - 0.12, 2.52, shared_cx + 0.02, 3.12),
        (shared_cx + 0.50, 2.34, shared_cx + 0.58, 2.88),
    ]:
        ax.plot([x0, x1], [y0, y1], color="#7d5cc8", lw=1.05,
                alpha=0.42, linestyle="--", zorder=6)
    for dx, dy, c, rad in dots:
        ax.add_patch(Circle((dx, dy), rad, fc=c, ec="white", lw=1.2, zorder=9))
    ax.text(shared_cx, 1.93, r"$s_{ij}=\hat b_i^{(m)}\cdot\hat b_j^{(t)}$",
            ha="center", va="center", fontsize=8.7, color=COL["ink"], zorder=10,
            bbox=dict(boxstyle="round,pad=0.20,rounding_size=0.07",
                      fc="#fffefe", ec="#cdbbed", lw=0.75, alpha=0.98))

    # Loss.
    loss_box = rbox(ax, shared_x - 0.08, 4.23, shared_w + 0.16, 0.52,
                    "CLOOB / InfoLOOB objective",
                    "align matched pairs; repel impostors",
                    fc="#fff5ec", ec=COL["loss_edge"], lw=1.55, fs=7.6, bold=True)
    loss_box.set_linestyle("--")
    arrow(ax, shared_cx, 4.23, shared_cx, 4.08,
          color=COL["loss_edge"], lw=1.15, ls="--", ms=7, z=8)
    ax.text(shared_cx + 0.28, 4.13, "supervises alignment", ha="left", va="center",
            fontsize=5.4, color=COL["loss_edge"], fontweight="bold", zorder=10)
    arrow(ax, shared_x - 0.04, 4.48, 7.68, 4.48, color=COL["loss_edge"],
          lw=1.0, ls="--", rad=0.0, ms=8)

    # Audit outputs: one clean, vertically aligned stack.
    out_x, out_w = 12.12, 1.60
    explain_y, retrieve_y, probe_y = 3.55, 2.48, 1.40
    explain_h = retrieve_h = probe_h = 0.86

    rbox(ax, out_x, explain_y, out_w, explain_h,
         fc="#ffffff", ec="#e0a923", fs=7.4, bold=True)
    ax.add_patch(FancyBboxPatch((out_x + 0.08, explain_y + 0.15), 0.055, explain_h - 0.30,
                                boxstyle="round,pad=0.004,rounding_size=0.025",
                                fc="#e0a923", ec="none", zorder=8))
    ax.text(out_x + out_w - 0.14, explain_y + explain_h - 0.10, "explain",
            ha="right", va="center", fontsize=6.7, color="#9a6700",
            fontweight="bold", zorder=9)
    ax.text(out_x + out_w / 2, explain_y + explain_h - 0.28, "ECFP4 saliency",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=COL["ink"], zorder=9)
    mini_bar_chart(ax, out_x + 0.24, explain_y + 0.17, 1.16, 0.41)
    ax.text(out_x + out_w / 2, explain_y + 0.08, "bit scores",
            ha="center", va="center", fontsize=6.4, color=COL["muted"], zorder=9)

    rbox(ax, out_x, retrieve_y, out_w, retrieve_h,
         fc="#ffffff", ec=COL["mol"], fs=7.2, bold=True)
    ax.add_patch(FancyBboxPatch((out_x + 0.08, retrieve_y + 0.15), 0.055, retrieve_h - 0.30,
                                boxstyle="round,pad=0.004,rounding_size=0.025",
                                fc=COL["mol"], ec="none", zorder=8))
    ax.text(out_x + out_w - 0.14, retrieve_y + retrieve_h - 0.10, "retrieve",
            ha="right", va="center", fontsize=6.7, color=COL["mol_dark"],
            fontweight="bold", zorder=9)
    ax.text(out_x + out_w / 2, retrieve_y + retrieve_h - 0.27, "Ranked retrieval",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=COL["ink"], zorder=9)
    mini_retrieve_list(ax, out_x + 0.28, retrieve_y + 0.26, 1.05, 0.37)
    ax.text(out_x + out_w / 2, retrieve_y + 0.13, "top mechanisms",
            ha="center", va="center", fontsize=6.4, color=COL["muted"], zorder=9)

    rbox(ax, out_x, probe_y, out_w, probe_h,
         fc="#ffffff", ec=COL["text"], fs=7.2, bold=True)
    ax.add_patch(FancyBboxPatch((out_x + 0.08, probe_y + 0.15), 0.055, probe_h - 0.30,
                                boxstyle="round,pad=0.004,rounding_size=0.025",
                                fc=COL["text"], ec="none", zorder=8))
    ax.text(out_x + out_w - 0.14, probe_y + probe_h - 0.10, "probe",
            ha="right", va="center", fontsize=6.7, color=COL["text_dark"],
            fontweight="bold", zorder=9)
    ax.text(out_x + out_w / 2, probe_y + probe_h - 0.27, "RoBERTa logit lens",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=COL["ink"], zorder=9)
    mini_line_chart(ax, out_x + 0.30, probe_y + 0.19, 1.00, 0.34)
    ax.text(out_x + out_w / 2, probe_y + 0.10, "layer-wise retrieval",
            ha="center", va="center", fontsize=6.4, color=COL["muted"], zorder=9)

    # Main arrows.
    arrow(ax, 2.48, 3.82, 3.20, 3.82)
    arrow(ax, 2.48, 2.05, 3.20, 2.05)
    arrow(ax, 5.10, 3.82, 5.48, 3.82)
    arrow(ax, 5.10, 2.05, 5.48, 2.05)
    arrow(ax, 6.62, 3.82, 6.83, 3.82)
    arrow(ax, 6.62, 2.05, 6.83, 2.05)
    arrow(ax, 7.74, 3.82, 7.94, 3.75)
    arrow(ax, 7.74, 2.05, 7.94, 1.94)
    arrow(ax, 8.86, 3.73, shared_x - 0.02, 3.24, rad=-0.10, lw=1.55)
    arrow(ax, 8.86, 1.94, shared_x - 0.02, 2.33, rad=0.10, lw=1.55)
    arrow(ax, shared_right + 0.03, 3.12, out_x - 0.06, explain_y + explain_h / 2,
          rad=0.12, lw=1.75, ms=11)
    arrow(ax, shared_right + 0.03, 2.78, out_x - 0.06, retrieve_y + retrieve_h / 2,
          rad=0.00, lw=1.75, ms=11)
    arrow(ax, shared_right + 0.03, 2.34, out_x - 0.06, probe_y + probe_h / 2,
          rad=-0.12, lw=1.75, ms=11)

    # Explicit legend for the visual encoding.
    legend_box = FancyBboxPatch(
        (0.62, 0.34), 13.10, 0.34,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        fc="#ffffff", ec="#dbe4ee", lw=0.7, zorder=2,
    )
    ax.add_patch(legend_box)
    ax.text(0.82, 0.51, "Legend", ha="left", va="center",
            fontsize=6.3, fontweight="bold", color=COL["ink"], zorder=8)
    legend = [
        (COL["mol"], "molecule branch"),
        (COL["text"], "text branch"),
        (COL["frozen"], "frozen encoder"),
        (COL["bridge"], "trainable bridge"),
        (COL["shared"], "shared space"),
        (COL["loss"], "loss objective"),
    ]
    legend_left = 1.55
    legend_step = 2.02
    for idx, (col, label) in enumerate(legend):
        x = legend_left + idx * legend_step
        swatch = FancyBboxPatch(
            (x, 0.44), 0.14, 0.14,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc="#fff5ec" if label == "loss objective" else col,
            ec=COL["loss_edge"] if label == "loss objective" else "#94a3b8",
            lw=0.8 if label == "loss objective" else 0.4,
            zorder=8,
        )
        if label == "loss objective":
            swatch.set_linestyle("--")
        ax.add_patch(swatch)
        ax.text(x + 0.22, 0.51, label, ha="left", va="center",
                fontsize=6.0, color=COL["muted"], zorder=8)

    fig.tight_layout(pad=0.02)
    paper_pdf = PAPER_FIG_DIR / "fig_architecture.pdf"
    paper_png = PAPER_FIG_DIR / "fig_architecture.png"
    root_pdf = ROOT_FIG_DIR / "fig_architecture.pdf"
    root_png = ROOT_FIG_DIR / "fig_architecture.png"
    fig.savefig(paper_pdf, bbox_inches="tight", pad_inches=0.015, format="pdf")
    fig.savefig(paper_png, bbox_inches="tight", pad_inches=0.015, dpi=240)
    shutil.copy2(paper_pdf, root_pdf)
    shutil.copy2(paper_png, root_png)
    plt.close(fig)
    print(f"Saved: {paper_pdf}")
    print(f"Saved: {root_pdf}")


if __name__ == "__main__":
    render()
    raise SystemExit

r'''
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
'''
