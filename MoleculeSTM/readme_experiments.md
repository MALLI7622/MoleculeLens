# TODO — Next Steps (NeurIPS Workshop Revision)

> Advisor recommendation: add baselines to strengthen the drug-name leakage finding and scaffold-split generalisation story.
> Priority order below matches highest ROI for the paper.

---

## TODO 1 — KV-PLM on ChEMBL *(in progress — start here)*

**Goal:** Show that drug-name leakage is a field-wide problem, not just a MoleculeSTM artifact.
Run KV-PLM zero-shot on ChEMBL, both with and without drug names in the text, and add it to Table 1 and Table 2 of the paper.

**Status:** KV-PLM checkpoint already downloaded to `data/pretrained_KV-PLM/ckpt_ret01.pt`.
Script to write: `scripts/downstream_01_retrieval_ChEMBL_KV-PLM.py`
(adapt `scripts/downstream_01_retrieval_Description_Pharmacodynamics_KV-PLM.py` + the ChEMBL CSV loading from `scripts/downstream_01_retrieval_ChEMBL.py`)

**Key things the script needs to do:**
- Load `BigModel` (SciBERT BERT wrapped, same for both SMILES and text — that's the KV-PLM trick)
- Load `data/ChEMBL_data/ChEMBL_retrieval_test_all.csv` (`canonical_smiles`, `text` columns)
- Embed all SMILES and all texts through the same `BigModel`
- Run T-choose-one in both directions (S→T and T→S) at T ∈ {4, 10, 20}
- Run again with drug names stripped from `text` (name-leakage ablation)
- Output format matching `downstream_01_retrieval_ChEMBL.py`

**Expected result:** KV-PLM's Recall@1 / T-choose-one should collapse similarly to MoleculeSTM's 84.3% drop when drug names are removed — because it also uses a BERT tokenizer that sees drug name strings.

---

## TODO 2 — BM25 + ECFP4 Tanimoto Non-Neural Baselines

**Goal:** Give reviewers the expected lower bounds and quantify leakage via keyword matching.
~50 lines of code each; no GPU needed; no pretrained weights.

**BM25 (text-only retrieval):**
- Use `rank_bm25` (`pip install rank-bm25`) on the mechanism text descriptions
- Query = each text; corpus = all texts; retrieve top-k
- Run with and without drug names → the BM25 score should mirror MoleculeSTM's collapse, proving leakage is just keyword matching
- Metrics: Recall@{1,5,10}, MRR, T-choose-one at T ∈ {4,10,20}

**ECFP4 Tanimoto (structure-only retrieval):**
- Use RDKit `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)` + `DataStructs.TanimotoSimilarity`
- Query = each molecule; corpus = all molecules; rank by Tanimoto
- Gives the structural ceiling — how well pure fingerprint similarity correlates with mechanism text
- Expected: moderate within-scaffold, poor cross-scaffold (consistent with scaffold-split design)

**Script to write:** `scripts/downstream_01_retrieval_ChEMBL_baselines.py`
Outputs both baselines in the same metric format as the KV-PLM and MoleculeSTM scripts.

---

## TODO 3 (Optional) — Text2Mol

**Goal:** Engage the EMNLP/ACL retrieval community; show domain specificity of ChEMBL vs general ChEBI-20 descriptions.

- Paper: Edwards et al., EMNLP 2021 — https://aclanthology.org/2021.emnlp-main.47/
- Code: https://github.com/cnedwards/text2mol
- Model: GCN + SciBERT pretrained on ChEBI-20 (general chemical descriptions, not bioassay text)
- Expected: lower performance than Thin Bridges on ChEMBL — illustrates why in-domain training matters
- Requires: PyTorch Geometric (already installed), ChEBI-20 → ChEMBL format adapter (~0.5 day)

---

# MoleculeSTM Retrieval Experiments

All experiments use the pretrained MoleculeSTM SMILES model.

## Environment Setup

The following env vars must be set before running any experiment (paths are hardcoded to `/workspace/` in defaults, so pass `--vocab_path` explicitly):

```bash
conda activate MoleculeSTM
export LD_LIBRARY_PATH=/home/cheriearjun/miniconda3/envs/MoleculeSTM/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/cheriearjun/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism:/home/cheriearjun/apex:/home/cheriearjun/apex/build/lib.linux-x86_64-cpython-37:$PYTHONPATH
cd /home/cheriearjun/MoleculeSTM
```

- `LD_LIBRARY_PATH`: needed for `libcusparse.so.11` (torch_sparse)
- `PYTHONPATH`: provides `megatron`, `apex`, and `fused_layer_norm_cuda` (compiled CUDA extension)

## Experiment 1: Description Retrieval (removed PubChem)

```bash
conda activate MoleculeSTM
export LD_LIBRARY_PATH=/home/cheriearjun/miniconda3/envs/MoleculeSTM/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/cheriearjun/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism:/home/cheriearjun/apex:/home/cheriearjun/apex/build/lib.linux-x86_64-cpython-37:$PYTHONPATH
cd /home/cheriearjun/MoleculeSTM
python scripts/downstream_01_retrieval_Description_Pharmacodynamics.py \
    --task=molecule_description_removed_PubChem \
    --molecule_type=SMILES \
    --input_model_dir=data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=MoleculeSTM/bart_vocab.txt
```

### Results

| Mode | T=4 Acc (%) | T=10 Acc (%) | T=20 Acc (%) |
|------|------------|-------------|-------------|
| Given Text → Retrieve Structure (zero-shot) | 97.04 | 93.16 | 89.78 |

## Experiment 2: Pharmacodynamics Retrieval (removed PubChem)

```bash
python scripts/downstream_01_retrieval_Description_Pharmacodynamics.py \
    --task=molecule_pharmacodynamics_removed_PubChem \
    --molecule_type=SMILES \
    --input_model_dir=data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=MoleculeSTM/bart_vocab.txt
```

### Results

| Mode | T=4 Acc (%) | T=10 Acc (%) | T=20 Acc (%) |
|------|------------|-------------|-------------|
| Given Text → Retrieve Structure (zero-shot) | 87.51 | 78.82 | 71.45 |

## Experiment 3: ATC Retrieval

```bash
python scripts/downstream_01_retrieval_ATC.py \
    --molecule_type=SMILES \
    --input_model_dir=data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=MoleculeSTM/bart_vocab.txt
```

### Results (ATC Level 5)

| Mode | T=4 Acc (%) | T=10 Acc (%) | T=20 Acc (%) |
|------|------------|-------------|-------------|
| Given Text → Retrieve Structure (zero-shot) | 70.41 | 56.44 | 45.20 |

## Experiment 4: ChEMBL Retrieval

Run from `/home/cheriearjun/MoleculeSTM/scripts`:

```bash
cd scripts
python downstream_01_retrieval_ChEMBL.py \
    --task=molecule_mechanism_all \
    --molecule_type=SMILES \
    --input_model_dir=../data/pretrained_MoleculeSTM/SMILES \
    --vocab_path=../MoleculeSTM/bart_vocab.txt
```

### Results (task: molecule_mechanism_all)

| Direction | T=4 Acc (%) | T=10 Acc (%) | T=20 Acc (%) |
|-----------|------------|-------------|-------------|
| Given Structure → Retrieve Text | 29.87 | 13.87 | 7.77 |
| Given Text → Retrieve Structure | 25.13 | 10.53 | 5.07 |

---

# MoleculeLens Experiments

The following experiments are run from `/workspace/MoleculeLens`.

## Step 1: Download Data

```bash
cd /workspace/MoleculeLens
python 01_download_data.py
```

## Step 2: Preprocess and Baselines

```bash
python 02_preprocess_and_baselines.py
```

## Step 3: Train Scaffold Split

```bash
python 03_train_scaffold_split.py
```

## Step 3b: Train Global Bridge

```bash
python 03b_train_global_bridge.py
```

## Step 4: Results Visualisation

Open and run the Jupyter notebook:

```
04_results_visualisation.ipynb
```

## Step 5: Compare MoleculeSTM vs Thin Bridges

```bash
python 05_compare_moleculestm_vs_thinbridges.py \
    --moleculestm_dir /workspace/MoleculeSTM \
    --bridge_outdir /workspace/MoleculeLens/outputs \
    --outdir /workspace/MoleculeLens/outputs/comparison
```
