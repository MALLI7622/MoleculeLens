# MoleculeLens Paper Repo

This repository contains the LaTeX source for the paper draft.

## Push Changes

Run these commands after editing any files:

```bash
cd /home/cheriearjun/MoleculeLens-paper
git add .
git commit -m "Describe your change here"
git push origin main
```

## Check Status

Use this before committing if you want to see what changed:

```bash
cd /home/cheriearjun/MoleculeLens-paper
git status
```

## Build PDF Locally

Compile the paper after making changes:

```bash
cd /home/cheriearjun/MoleculeLens-paper
source ~/.bashrc
latexmk -pdf -interaction=nonstopmode neurips_2026.tex
```

Force a clean rebuild if the PDF does not refresh:

```bash
cd /home/cheriearjun/MoleculeLens-paper
source ~/.bashrc
latexmk -C
latexmk -pdf -interaction=nonstopmode neurips_2026.tex
```

## Show Author Names In PDF

If you want to see your real author names locally, use preprint mode in `neurips_2026.tex`:

```tex
\usepackage[preprint]{neurips_2026}
```

The default line:

```tex
\usepackage{neurips_2026}
```

keeps the paper anonymous for submission.

## Current Main Files

- `neurips_2026.tex` — main paper source
- `refs.bib` — bibliography
- `checklist.tex` — NeurIPS checklist
- `neurips_2026.sty` — style file
