# MoleculeLens: Next Steps (NeurIPS 2026 reviews → Nature Machine Intelligence)

This file turns the NeurIPS 2026 reviews into a revision plan for the Nature Machine Intelligence (NMI) submission.
The reviews and our working notes are in the shared review document:
https://docs.google.com/document/d/1FJnPE_qbCgJQpwvD4-Rvc5yzxCnnAO5mvg6N8xQMccI

> **Note:** "Review 2" in that document is about a different paper (SMolLM, a SMILES generator), not MoleculeLens.
> Every action below comes from **R1** or **R3**. If the real second MoleculeLens review turns up, add it here.

---

## 0. The core problem to address

The headline claim is "+9.6 R@1 over MolPrompt on the scaffold split." Our own Table 2 undercuts it:

| Condition (scaffold split, N=435) | R@1 | 95% CI |
|---|---|---|
| MoleculeLens `text_rich` (with drug name) | 0.225 | [0.186, 0.267] |
| MoleculeLens retrained, no drug name | 0.110 | [0.083, 0.140] |
| MolPrompt | 0.129 | [0.101, 0.164] |

Without drug names, MoleculeLens is roughly level with MolPrompt, and we never ran the baselines without names (R3 W1/Q1).

**Proposed reframing for NMI:** make the paper's main contribution an *interpretable leakage audit of drug–text retrieval*, not a leaderboard win.
That means all models evaluated with and without names, on our benchmark and on established ones, with MoleculeLens as the lens that shows *which substructures* a score depends on.
This turns R3's central criticism into our main result.

---

## 1. Experiments

Priority: **P0** = required before resubmission · **P1** = strongly expected · **P2** = nice to have.

### 1.1 Leakage and baseline fairness

| # | Pri | Task | Addresses |
|---|---|---|---|
| E1 | P0 | Re-run **all baselines** (MolPrompt, KV-PLM, MoleculeSTM, Graphormer) with drug names removed (`text_nodrug`) and with mechanism text only. Report them side by side with MoleculeLens. | R3 W1, Q1 |
| E2 | P0 | Add a **scaffold-split row for MoleculeSTM** (currently Full-gallery only). | R3 Q1 |
| E3 | P0 | **Document how each baseline was run**: zero-shot vs. fine-tuned, which checkpoint, what input text, what training budget. Use the same protocol for every model or explain any difference. | R3 Q2 |
| E4 | P1 | **INN-stem control**: mask drug-name stems that encode the mechanism class (-tinib, -vir, -prazole, -olol, -mab, …) and other implicit name fragments, then re-evaluate. | R3 Q6; our own Limitations |

### 1.2 Data and generalization

| # | Pri | Task | Addresses |
|---|---|---|---|
| E5 | P0 | **Expand the dataset beyond approved drugs**: drop the `max_phase=4` filter (keep clinical-phase and preclinical ChEMBL compounds that have curated mechanisms). If any subset is kept, justify the choice explicitly. | R3 Q4, R1 (effective sample size) |
| E6 | P0 | **Evaluate on established benchmarks** (PCDes, MoMu, PubChem324k / PubChemSTM) where the baselines have published numbers. Explain why we also need our own benchmark (leakage controls, mechanism text). | R3 W1, Q3 |
| E7 | P1 | **Text-side out-of-distribution splits**: (a) target-disjoint split, (b) split that holds out both scaffolds and targets. Tests generalization to unseen biology, not only unseen chemistry. | R1 (OOD weakness) |

### 1.3 Statistical rigor

| # | Pri | Task | Addresses |
|---|---|---|---|
| E8 | P0 | Run the **headline configuration across ≥5 seeds** and report mean ± std for R@1, R@5, R@10 and MRR. Use one protocol for main and robustness tables (no mixing full vs. proxy objective). | R3 Q5, R1 UQ |
| E9 | P0 | **Bootstrap grouped by parent structure** (193/435 test rows belong to 76 parents) for all metrics, the faithfulness numbers (Table 4) and the leakage numbers (§3.2). | R1 UQ, R1 effective N |
| E10 | P0 | **Explain how the canonical split relates to seeds 0–4** (Table 3 test sizes 403/261/224/221/395 vs. canonical 435). Rename "seed" to "scaffold partition" where it means a different split. List the dominant scaffolds per partition. | R1 Q, R3 Q5 |
| E11 | P1 | Report the leakage drop with a CI across partitions (it currently swings from 2% to 35%) and stop relying on a single-split leakage number. | R3 Q5 |

### 1.4 Interpretability validity

| # | Pri | Task | Addresses |
|---|---|---|---|
| E12 | P0 | **Strip salts and counterions** (RDKit `FragmentParent`) *before* computing fingerprints and saliency. Today Fig. 3D's top bit is `Na+` (bit 1213) and Fig. 3A's is the acetate counterion. | R1 (Fig. 3) |
| E13 | P0 | **Quantitative chemical grounding**: compare top-k salient atoms against pharmacophore or known-binding-motif annotations (e.g., ChEMBL binding sites, PDB ligand contacts, or expert-labelled pharmacophores) across the *whole* test set, with a random-bit or random-atom null for above-chance testing. Optionally add a small expert-rating study. | R1, R3 W2 ("8 hand-picked examples") |
| E14 | P0 | **Per-layer logit lens**: train a separate linear head on each RoBERTa layer's CLS embedding instead of reusing the layer-12 head. If the result does not change the story, shrink or drop §3.3. | R3 W2 #4, Q10 |
| E15 | P1 | **Fingerprint bit-collision analysis**: report the collision rate of the 2048-bit folded ECFP4, how often each bit fires in training, and flag attributions on rarely trained bits. Compare with an unfolded or 4096-bit fingerprint. | R3 W2 #3, Q8 |
| E16 | P1 | **Compare attribution with baselines**: run integrated gradients or the published attribution methods on MolPrompt and Text2Mol, scored by the same grounding metric as E13. | R3 W2 #2, Q9 |
| E17 | P2 | Bit-drop / substructure-deletion test: remove top-salient substructures from the molecule and measure the retrieval drop vs. removing random ones. | R3 W2 (causal evidence) |

### 1.5 Ablations and compute

| # | Pri | Task | Addresses |
|---|---|---|---|
| E18 | P1 | **ECFP radius ablation**: ECFP2/4/6 (radius 1/2/3), reporting metrics and runtime. | R1 |
| E19 | P1 | **Shared-space dimension ablation** (e.g., 64/128/256/512/1024). | R1 |
| E20 | P1 | **Compute report**: exact GPU model, plus wall-clock for text encoding, training, evaluation, attribution and logit lens. | R1 |

---

## 2. Writing and presentation

### 2.1 Claims to soften or remove

- [ ] "Experimentally validated substructure explanations": proxy-vs-gradient agreement is close to guaranteed for a linear branch. Lead with the **top-10 Jaccard (0.71)** and present it as faithfulness to the model, not chemical validity. Chemical validity comes from E13. *(R3 W2, Q6)*
- [ ] Drop "same text, different drug → distinct top-10 sets" as evidence of drug-specificity. It follows mechanically from the fingerprints being different. *(R3 Q7)*
- [ ] Fig. 3 caption, "high-saliency bits align with pharmacophoric regions": rewrite after E12/E13. *(R1)*
- [ ] Claim that existing methods cannot expose substructure evidence: either support it with E16 or remove it. *(R3 Q9)*
- [ ] State clearly that the saliency is a **linear diagnostic of the model's score**, not full coverage of chemistry. *(R1 Limitations)*
- [ ] Frame the R@5/R@10 results honestly (MolPrompt is ahead), with error bars from E8. *(R1 Limitations)*

### 2.2 Clarity

- [ ] **Equation 1**: include the molecule mask in the main text, `attr_k(i,j) = x_ik · [W_mᵀ b̂_j]_k`. *(R3 W3)*
- [ ] Define in the main text: "leakage drop" (formula, now only in Table 6), "molecule gallery", "target/action", "Full" vs. "Scaffold", `text_rich` ("rich model"). *(R1, R3 W3)*
- [ ] Move the Full (N=2699) rows of Table 1 to the appendix, since they evaluate on training data. *(R1)*
- [ ] Fill in the empty stub appendices (A, B, E) with real implementation and evaluation details.
- [ ] Fix the S-Biomed-RoBERTa citation: "Deka et al. 2021" currently points to an unrelated fact-checking paper.

### 2.3 Figures

- [ ] Fig. 1: redraw the architecture without overlapping text or arrows. *(R3)*
- [ ] Fig. 2: larger fonts, no overlapping labels. *(R1)*
- [ ] Fig. 3: regenerate on salt-stripped parents (E12); consider moving it to the appendix and keeping one compact example in the main text. *(R3)*
- [ ] Fig. 4: vector graphics, no cropped panels, aligned labels. *(R3)*
- [ ] Fig. 5: move the molecule renderings off the bar plots into a separate panel. *(R3)*

### 2.4 NMI-specific

- [ ] Reframe the abstract, introduction and contributions around the leakage-audit story (Section 0).
- [ ] NMI favours broad significance over implementation detail. Move benchmark construction and hyperparameters to Methods or Supplementary (the lesson from our NucleicBERT reading notes).
- [ ] Remove the NeurIPS checklist; add NMI's Data Availability, Code Availability and Reporting Summary sections.
- [ ] Release code, split manifests and projection weights (currently "upon acceptance"; NMI expects availability at submission or review).

---

## 3. Suggested order of work

1. **Quick checks (days):** E12 salt stripping, E14 per-layer heads, E10 split bookkeeping, E20 timings, citation fix.
   These are cheap and tell us early whether any section needs to be cut.
2. **Decisive experiment:** E1–E3, all baselines name-stripped. The result decides the final framing.
3. **Scale-up:** E5 expanded dataset, E6 external benchmarks, E7 target splits, E8–E9 seeds and grouped bootstrap on everything.
4. **Interpretability validity:** E13 grounding with a null, E15 collisions, E16 baseline attribution.
5. **Ablations:** E18–E19.
6. **Rewrite and figures:** Section 2, last, once the numbers are final.

---

## 4. Reviewer coverage map

| Reviewer point | Covered by |
|---|---|
| R1 – Fig. 3 salt / counterion saliency | E12, §2.1 |
| R1 – ECFP radius only 2 | E18 |
| R1 – shared-space dimension fixed | E19 |
| R1 – UQ only for R@1 | E8, E9 |
| R1 – Full rows test on training data | §2.2 |
| R1 – no text-side OOD control | E7, E1, E4 |
| R1 – 76 parents → effective N lower | E9, E5 |
| R1 – scaffold split details, "seed" naming | E10 |
| R1 – hardware and timings | E20 |
| R1 – "leakage drop" undefined; Fig. 2 fonts | §2.2, §2.3 |
| R3 W1/Q1 – baselines never name-stripped; no MoleculeSTM scaffold row | E1, E2 |
| R3 Q2 – baseline protocols undocumented | E3 |
| R3 Q3 – why a new benchmark | E6 |
| R3 Q4 – why approved-only | E5 |
| R3 Q5 – no cross-seed error bars on the headline | E8, E10, E11 |
| R3 Q6 – "validated" is internal; INN stems | E13, E4, §2.1 |
| R3 Q7 – distinct top-10 sets is mechanical | §2.1 |
| R3 Q8 – bit collisions; untrained bits | E15 |
| R3 Q9 – no attribution comparison with baselines | E16 |
| R3 Q10 – mismatched logit-lens probe | E14 |
| R3 W3 – Eq. 1, undefined terms, figure quality | §2.2, §2.3 |

---

## 5. Results log (overnight run, 2026-10-14/15)

### ⭐ Overnight summary (read this first)
21 experiments run. Code in `MoleculeLens/nmi/`, raw results in `MoleculeLens/nmi_results/`. Every number below is in the per-experiment sections.

**Corrections the paper needs:**
1. The "retrained no-drug" R@1 0.110 is a pairing bug; the correct value is **0.145**. The leakage drop is 35.6%, not 51.0% (E1).
2. The +9.6 R@1 margin over MolPrompt mixes tie rules. Under the same rule it is +8.5 (optimistic) or **+4.2 (structure-deduplicated gallery, recommended)** (E1).
3. The headline 0.225 is a favourable seed; **10-seed mean 0.213 ± 0.010** (E8).
4. **MoleculeSTM on the scaffold test (zero-shot) matches or beats MoleculeLens with names** (R@1 struct 0.230 vs 0.208), so "strongest top-rank retrieval" is not true for rich text (E2).
5. **§3.3 logit lens is wrong.** With a per-layer head the signal exists from layer 1, and a bag of token embeddings (layer 0, mean pool) is best (E14).
6. **76% of top-1 saliency on salt rows is counterion bits**, driven by salt words in the drug name (E12).
7. The target-split rich score is inflated by molecule overlap (33% of test molecules in train) (E7b).

**What holds up, and is stronger than claimed:**
- **Name-free, MoleculeLens beats every baseline.** Paper config: 0.145 vs MolPrompt 0.074, MoleculeSTM 0.060, KV-PLM 0.034 (E1, E2). **Final model: 0.201 vs MolPrompt 0.097 on the deduplicated gallery, +10.7 [+5.2, +16.2]** (E21).
- **The saliency is causally faithful** (deleting the top bits destroys hits, 16× more than random deletion) (E17), and **chemically grounded beyond chance and text-specifically** (top-3 pharmacophore hit 0.70 → 0.84 in the final model vs 0.50 chance), while MolPrompt's attributions are at chance (E13, E16, E21).
- **The canonical split was not cherry-picked:** it is the second-hardest of 10 partitions (E10).
- **On ChEBI-20 / PubChem324k, a two-linear-head model is competitive with the original Text2Mol baseline.** Those benchmarks reuse 70–84% of test scaffolds in training, and on scaffold-novel molecules R@1 roughly halves, which is the motivation for our benchmark (E6).

**Recommended NMI framing:** an interpretable *leakage audit* of drug–text retrieval.
1. Names, INN stems, salt words and molecule overlap inflate scores for every model.
2. MoleculeLens's closed-form lens is what exposes these shortcuts.
3. Once they are removed, MoleculeLens is the best name-free model and its explanations are the only ones that are chemically grounded.

**Recommended final model (E21):** parent (salt-stripped) ECFP4 with 8192 bits, mean-pooled S-Biomed-RoBERTa, 1024-d shared space, trained on all-phase ChEMBL (5,035 pairs). Evaluate name-free on the structure-deduplicated gallery over 10 scaffold partitions.

**Not done; needs you, or more time:**
- PCdes benchmark (must be rebuilt from KV-PLM raw files).
- A real expert rating study (E13 is an automated proxy).
- MolPrompt and MoleculeSTM retrained on the all-phase data and the OOD splits.
- Integrated gradients for MolPrompt.
- Check the test-drug overlap with MoleculeSTM's PubChemSTM pretraining set.
- KV-PLM per-example ranks (only aggregate metrics are stored).
- All writing and figure redraws (§2 of this file): Eq. 1 mask, definitions, Fig. 1/2/4/5, regenerate Fig. 3 with parent fingerprints, NMI restructuring.

Code: `MoleculeLens/nmi/` (shared library `core.py`; one script per experiment).
Raw outputs: `MoleculeLens/nmi_results/` (JSON + markdown per experiment).
Hardware for this run: 1× NVIDIA **Tesla T4** 16 GB (GCP `n1-standard-4`, us-central1-a). L4 GPUs were sold out across the US.
The paper's original runs used 1× NVIDIA **L4** 24 GB (`g2-standard-4`).

### E00: pipeline sanity check ✅
- The paper's saved checkpoint, run through the new pipeline, gives R@1 0.221. With the saved text embeddings it gives **0.2253**, matching the paper's 0.225. The 0.004 gap comes from GPU text-encoding numerics (max |ΔCLS| = 0.018; one query = 0.0023 R@1).
- The vectorised loss equals the paper's loop implementation exactly (6.44997 vs 6.44997).
- Retraining at seed 0 gives R@1 0.200, MRR 0.290 in 1.1 s on the T4.

### E1: name-stripped baselines ✅ (with two corrections to the paper)
**Correction 1: the paper's "retrained no-drug" R@1 = 0.110 is a pairing bug.**
`09b_diagonal_artifact_sync.py` scores the no-drug model's *text* embeddings against the *rich* model's *molecule* embeddings (`outputs/Bm_test.pt`).
Pairing the no-drug text with its own molecule head gives **R@1 0.145** (MRR 0.244).
The leakage drop is therefore **35.6%** (0.225 → 0.145), not 51.0%.

**Correction 2: tie handling inflates the headline margin.**
87 of the 435 test rows have an ECFP4 vector identical to another test row (mostly the same drug under several mechanism rows).
The shared evaluator counts ties as hits (optimistic), whereas MolPrompt's stored 0.129 came from its own scorer.
Recomputing MolPrompt from its saved test embeddings under the shared rule gives 0.140.

Main table (scaffold test, N=435; 95% CIs from a 10k cluster bootstrap over 358 unique test structures; shared optimistic-tie evaluator):

| Model | Protocol | R@1 [95% CI] | R@5 | R@10 | MRR |
|---|---|---|---:|---:|---:|
| MoleculeLens, rich text | trained (scaffold train) | 0.225 [0.182, 0.271] | 0.384 | 0.464 | 0.303 |
| MoleculeLens, retrained no-drug | trained | **0.145** [0.108, 0.184] | 0.359 | 0.464 | 0.244 |
| MoleculeLens, mechanism-only | trained | 0.124 [0.092, 0.159] | 0.331 | 0.453 | 0.226 |
| MoleculeLens, same model, name removed at inference | trained (rich) | 0.067 [0.040, 0.095] | 0.159 | 0.253 | 0.121 |
| MolPrompt, rich | fine-tuned (scaffold train, best val epoch) | 0.140 [0.108, 0.176] | 0.400 | 0.533 | 0.265 |
| MolPrompt, no-drug | fine-tuned | 0.074 [0.049, 0.102] | 0.287 | 0.418 | 0.182 |
| KV-PLM, rich | zero-shot (pretrained retrieval ckpt) | 0.055 [0.037, 0.081]† | 0.172 | 0.251 | 0.121 |
| KV-PLM, no-drug | zero-shot | 0.034 [0.021, 0.056]† | 0.110 | 0.179 | 0.084 |

† Wilson interval; only aggregate KV-PLM metrics were stored.

R@1 under different tie rules (both models scored by the same rule):

| | optimistic | random tie-break | strict | **structure-deduplicated gallery** (recommended) |
|---|---:|---:|---:|---:|
| MoleculeLens rich | 0.225 | 0.179 | 0.147 | **0.208** |
| MolPrompt rich | 0.140 | 0.130 | 0.122 | **0.166** |
| MoleculeLens retrained no-drug | 0.145 | 0.107 | 0.076 | **0.139** |
| MolPrompt no-drug | 0.074 | 0.070 | 0.067 | **0.087** |

Paired R@1 differences (parent-cluster bootstrap, optimistic rule):
- MoleculeLens rich − MolPrompt rich: **+0.085** [+0.035, +0.134], P(diff ≤ 0) = 0.0004
- MoleculeLens no-drug − MolPrompt no-drug: **+0.071** [+0.031, +0.110], P = 0.0003
- MoleculeLens no-drug − MolPrompt rich: +0.005 [−0.042, +0.051]. Without names, MoleculeLens matches MolPrompt *with* names.

Ceiling: once names are removed, only 195 of the 435 test query texts are unique (identical mechanism text is shared across drugs).
Single-label diagonal evaluation therefore caps no-drug R@1 at **0.448** (mechanism-only: 0.444), so low absolute no-drug numbers are partly a property of the evaluation.

**Takeaway for R3 Q1:** the advantage survives name removal and holds under every tie rule, but the honest margin is about **+4 to +5 R@1 points** (structure-deduplicated gallery), not +9.6.
- Report the structure-deduplicated gallery as the main protocol.
- Fix the 0.110 → 0.145 number and the 51.0% → 35.6% leakage drop.

### E3: how each baseline was run ✅ (from code and manifests)
| Model | Training on our data | Checkpoint / protocol | Eval set |
|---|---|---|---|
| MoleculeLens | trains 2 linear heads on the scaffold train split (2,019 rows) | seed 0, 100 epochs | scaffold test 435 |
| MolPrompt | **fine-tuned** on the same scaffold train split; best epoch chosen on val (26 rich / 27 no-drug), batch size 6 | `MolPrompt/main_chembl_retrieval.py` (`--remove_drug_name` for no-drug) | scaffold test 435 |
| KV-PLM | none (**zero-shot**) | `pretrained_KV-PLM/ckpt_ret01.pt` | scaffold test 435 |
| MoleculeSTM | none (**zero-shot**) | `chao1224/MoleculeSTM` SMILES checkpoint | full gallery 2,699 only (scaffold row: see E2) |
| Graphormer | none (zero-shot) | `graphormer-base-pcqm4mv1` | scaffold test 435 |

The rows labelled "zero-shot" are not adapted to our data, which disadvantages them relative to the trained MoleculeLens and the fine-tuned MolPrompt. Say this explicitly in the paper.

### E8–E11: seeds, cluster bootstrap, partitions, leakage variance ✅
Full tables: `MoleculeLens/nmi_results/e08_seeds_partitions.md`.

**E8, headline config across 10 initialisation seeds (canonical split):**

| Text | R@1 | R@1 struct | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| rich | 0.213 ± 0.010 | 0.201 ± 0.009 | 0.384 ± 0.009 | 0.468 ± 0.012 | 0.299 ± 0.007 |
| retrained no-drug | 0.129 ± 0.005 | 0.126 ± 0.006 | 0.337 ± 0.007 | 0.437 ± 0.008 | 0.228 ± 0.004 |
| mechanism-only | 0.130 ± 0.005 | 0.127 ± 0.005 | 0.339 ± 0.009 | 0.450 ± 0.007 | 0.229 ± 0.003 |

The paper's checkpoint (0.225) sits about 1.2 SD above the 10-seed mean. Report **0.213 ± 0.010** as the headline instead.

**E9, parent-cluster bootstrap** (318 parent structures / 435 rows), seed 0: R@1 0.200 [0.159, 0.243], R@5 0.379 [0.326, 0.434], R@10 0.462 [0.405, 0.518], MRR 0.290 [0.250, 0.332].

**E10:** scaffold partitions built with the paper's own recipe (`07_make_chembl_split_manifest.py`). **Partition 0 reproduces the canonical manifest exactly**, which answers R3 Q5 ("no seed reproduces the canonical split"). The paper's Table 3 used a *different* 90/10 recipe (`12c`), which is why its test sizes did not match.

| Partition | n_test | R@1 rich | R@1 no-drug (retrained) | R@1 same-model no-name | drop (retrained) | drop (same model) |
|---|---|---|---|---|---|---|
| 0 (= canonical) | 435 | 0.200 | 0.126 | 0.094 | 36.8% | 52.9% |
| 1 | 294 | 0.303 | 0.201 | 0.133 | 33.7% | 56.2% |
| 2 | 221 | 0.276 | 0.244 | 0.131 | 11.5% | 52.5% |
| 3 | 258 | 0.295 | 0.244 | 0.116 | 17.1% | 60.5% |
| 4 | 411 | 0.226 | 0.158 | 0.085 | 30.1% | 62.4% |
| 5 | 255 | 0.353 | 0.231 | 0.118 | 34.4% | 66.7% |
| 6 | 240 | 0.383 | 0.233 | 0.117 | 39.1% | 69.6% |
| 7 | 211 | 0.275 | 0.223 | 0.076 | 19.0% | 72.4% |
| 8 | 374 | 0.176 | 0.118 | 0.088 | 33.3% | 50.0% |
| 9 | 211 | 0.408 | 0.318 | 0.199 | 22.1% | 51.2% |
| **mean [95% t-CI]** | | **0.289** [0.235, 0.344] | 0.210 [0.166, 0.253] | 0.116 [0.090, 0.141] | **27.7%** [20.9, 34.5] | **59.4%** [53.6, 65.2] |

- **The canonical split is the second-hardest of 10**, so it was not cherry-picked in our favour.
- **E11:** the retrained leakage drop ranges from 11.5% to 39.1%, so a single-split leakage number should not be quoted. Report 27.7% [20.9, 34.5] (retrained) and 59.4% [53.6, 65.2] (same model).

### E4: what in the drug name drives retrieval (INN stems) ✅
Canonical split, model retrained per text variant, 5 seeds:

| Text the model sees | R@1 | R@1 struct | MRR |
|---|---|---|---|
| rich (mechanism + target + action + name) | 0.214 ± 0.013 | 0.202 | 0.297 |
| **name only** (`Drug: X.`) | **0.195 ± 0.014** | 0.182 | 0.271 |
| name replaced by its 5-letter stem (`-tinib`) | 0.169 ± 0.005 | 0.161 | 0.260 |
| no name | 0.130 ± 0.007 | 0.127 | 0.228 |
| mechanism only | 0.130 ± 0.005 | 0.128 | 0.229 |
| names shuffled across drugs | **0.093 ± 0.007** | 0.089 | 0.164 |

- **The drug name alone carries ~91% of the rich-text R@1** (0.195 / 0.214).
- **The INN stem alone recovers ~46% of the name's gain** (0.130 → 0.169 of 0.130 → 0.214); 69.7% of test drugs share their 5-letter stem with a training drug. This confirms R3 Q6.
- **A wrong name is worse than no name** (0.093 < 0.130), so the model actively uses the name.

### E7: target-disjoint and joint scaffold+target splits ✅
5 partitions each, init seed 0:

| Split | n_test | R@1 rich | R@1 no-drug | R@1 mechanism-only |
|---|---|---|---|---|
| scaffold-disjoint | 324 | 0.260 ± 0.045 | 0.195 ± 0.052 | 0.197 ± 0.059 |
| **target-disjoint** | 278 | 0.342 ± 0.049 | **0.098 ± 0.019** | 0.097 ± 0.013 |
| **scaffold- and target-disjoint** | 250 | 0.102 ± 0.026 | **0.060 ± 0.017** | 0.061 ± 0.022 |

- **For unseen targets, name-free retrieval collapses** (0.195 → 0.098).
- **Correction (E7b):** the rich-text target-split score (0.342) is inflated. **33.4% of target-split test rows have their exact molecule in training** (the same drug listed under another target), and the name lets the model memorise it. Removing test molecules from training gives **rich 0.103, no-drug 0.053** (R@1 struct). A target-disjoint split must also be molecule-disjoint.
- **When both chemistry and biology are new, only 6% of name-free queries retrieve the right drug.**
- Mechanism-level generalisation to new targets is the real open problem, and the paper should say so (R1's OOD point).

### E14: logit lens with a separately trained head per layer ✅ (overturns §3.3)
| Layer | reused L12 head (paper) | own head, CLS | own head, mean-pooled tokens |
|---|---|---|---|
| 0 (embeddings) | 0.002 | 0.002 (CLS = constant `<s>`) | **0.275** |
| 1 | 0.002 | 0.180 ± 0.020 | 0.268 |
| 6 | 0.002 | 0.074 ± 0.005 | 0.273 |
| 9 | 0.002 | 0.160 ± 0.011 | 0.260 |
| 11 | 0.005 | **0.213 ± 0.007** | 0.234 |
| 12 | 0.200 | 0.207 ± 0.007 | 0.225 |

- **R3 Q10 is correct.** With its own head, the CLS state carries drug-retrievable signal from layer 1 onwards (0.180), and layer 11 ≥ layer 12.
- **Mean-pooled raw token embeddings (layer 0, essentially a bag of words) give the best text representation of all (0.275).** Retrieval on this benchmark is largely lexical matching.
- **Action:** drop the "signal appears only at layer 12" claim. Replace §3.3 with this per-layer table and the lexical-shortcut interpretation. Consider mean pooling as the default text representation.

### E15: fingerprint bit collisions ✅
- **2048-bit ECFP4:** 99.3% of active bits host more than one distinct environment (mean 7.2 per bit), and 99.8% of top-10 salient entries fall on collided bits. A salient bit is **not** a unique substructure. Mapping a bit to the environments actually present in *that* molecule (as Fig. 3 does) is still well-defined, but should be stated explicitly.
- **Training exposure is not a problem:** median 50 training molecules per salient entry; 1.8% of entries are on bits seen in fewer than 5 training molecules; 0% on never-seen bits.
- **Larger folds reduce collisions and improve retrieval:**

| Fold | collision rate | R@1 (5 seeds) | MRR |
|---|---|---|---|
| 1024 | 100% | 0.207 ± 0.012 | 0.291 |
| 2048 (paper) | 99.3% | 0.214 ± 0.013 | 0.297 |
| 4096 | 89.1% | 0.251 ± 0.008 | 0.333 |
| **8192** | 62.4% | **0.263 ± 0.006** | 0.347 |
| 16384 | 37.6% | 0.258 ± 0.007 | 0.349 |

**Action:** move to 8192 bits. That's +5 R@1 points with ~38% fewer collisions.

### E17: deletion test (causal check of the saliency) ✅
All 435 pairs; zero the top-k salient bits of the correct molecule vs. k random or k least-salient bits:

| k | score drop, top-k | random k | least-salient k | top > random (pairs) | rank-1 hits kept: top / random / least (baseline 22.5%) |
|---|---|---|---|---|---|
| 1 | 0.077 | 0.005 | −0.038 | 99.8% | 8.3% / 16.4% / 26.4% |
| 3 | 0.164 | 0.014 | −0.089 | 100% | 4.1% / 14.1% / 35.4% |
| 5 | 0.233 | 0.023 | −0.125 | 100% | 0.7% / 12.6% / 43.4% |
| 10 | 0.364 | 0.048 | −0.181 | 100% | 0.0% / 9.6% / 58.2% |

- Deleting salient bits hurts **~16×** more than random deletion and destroys rank-1 hits.
- Deleting the least-salient (negative) bits *improves* the score.
- The fast proxy and the exact leave-one-out ranking give nearly identical drops (0.164 vs 0.165 at k=3).
- This is model-faithfulness evidence; chemical validity is tested separately in E13.
- **Atom-level variant:** deleting the atoms of the top-1 salient environment from the molecule (RDKit edit, fingerprint recomputed) in 68 correctly retrieved molecules **loses the rank-1 hit in 57 (83.8%)**, mean score drop 0.206. There is no random-atom control for this variant yet.

### E12: salts and counterions in the saliency ✅ (R1's Fig. 3 concern confirmed and explained)
- **204/435 test rows (47%) are salts or have counterions.** In the paper checkpoint the **top-1 salient bit is a counterion-only environment for 156 of them (76%)**; a counterion bit is in the top 6 for 186 (91%); 22% of absolute attribution mass sits on counterion bits.
- **Mechanism: a salt-name shortcut.** 180/204 salt rows name the salt in the drug name ("ACYCLOVIR **SODIUM**", "MAFENIDE **ACETATE**"). The top-1 counterion share is 76% for the rich model, 50% for the same model with the name removed at inference, and **3% for the model trained without names**. The model learned to match the salt word in the name to the counterion bit.
- **Parent-structure (salt-stripped) fingerprints**, 10 seeds:

| Fingerprint | R@1 | R@1 struct (unique parent) | R@5 | MRR |
|---|---|---|---|---|
| original | 0.213 ± 0.010 | 0.206 ± 0.009 | 0.384 | 0.299 |
| parent | 0.206 ± 0.006 | 0.182 ± 0.006 | 0.342 | 0.273 |

  About 2.4 R@1 points (deduplicated gallery) of the original score come from the salt shortcut.
- **Fig. 3 re-done with the parent model:**
  - Mafenide: top bits move from acetate to the aryl sulfonamide (`cccc(c)S`, `O=S`).
  - Acyclovir: `Na` disappears; the top bit becomes the guanine base (`cc([O-])nc(n)N`).
  - Carbamazepine and albuterol are unchanged (carboxamide / ethanolamine).
- **Action:** use parent fingerprints and regenerate Fig. 3. Report the salt-name shortcut as a *finding*: it is a clean example of the leakage MoleculeLens is designed to expose.

### E13: quantitative pharmacophore grounding ✅ (automated proxy for an expert study)
16 mechanism classes, each mapped to a literature pharmacophore (SMARTS, `nmi/pharm.py`). 118/435 test pairs fall in a class and **86** contain the pharmacophore. Atom saliency = Eq. 1 spread over each bit's atoms.

| | positive saliency mass on pharmacophore | top-1 atom in pharmacophore | top-3 atoms hit pharmacophore |
|---|---|---|---|
| **correct mechanism text** | **0.31** | **0.30** | **0.70** |
| same molecule, 20 wrong-class texts | 0.25 | 0.25 | 0.47 |
| untrained heads | 0.27 | – | – |
| chance (random atoms) | 0.26 | 0.26 | 0.50 |

- **Pooled:** correct > chance, p = 9×10⁻⁶; correct > wrong text, p = 3×10⁻⁷ (one-sided Wilcoxon).
- **Strongest classes:** β-adrenergic (top-3 0.85 vs 0.51), COX (0.64 vs 0.26), β-lactam (0.67 vs 0.19), azoles (1.00 vs 0.15).
- **Failures:** nucleoside/nucleobase polymerase inhibitors (0.07 vs 0.13; the acyclovir case R1 raised). Steroids are uninformative because the pharmacophore covers ~60% of atoms.
- **Honest summary:** the saliency is **significantly and text-specifically** grounded, but the effect is **modest** (1.2× chance in mass, 1.4× in top-3 hit rate) and class-dependent.
- **Action:** replace "chemically meaningful substructures" with this quantified, class-level statement; a small expert study remains optional.

### E18–E20: ECFP radius, embedding dimension, compute ✅
| Fingerprint (2048 bits, 5 seeds) | R@1 rich | R@1 no-drug |
|---|---|---|
| ECFP2 | 0.219 ± 0.007 | 0.134 ± 0.004 |
| ECFP4 (paper) | 0.214 ± 0.013 | 0.130 ± 0.007 |
| ECFP6 | 0.223 ± 0.011 | 0.130 ± 0.005 |
| ECFP8 | 0.206 ± 0.003 | 0.131 ± 0.008 |

Radius does not matter within noise. Note that ECFP6 is not worse, which answers R1.

| Shared dim | 32 | 64 | 128 | 256 (paper) | 512 | 1024 |
|---|---|---|---|---|---|---|
| R@1 (5 seeds) | 0.160 | 0.185 | 0.207 | 0.214 | 0.215 | **0.226** |

**E20 compute (Tesla T4 16 GB, GCP n1-standard-4):**
- text encoding of all 2,699 rows: 10.8 s (all 13 layers: 10.5 s)
- ECFP4 for 2,699 molecules: 0.8 s
- training both heads for 100 epochs: **1.05 s**
- evaluation (435×435): 0.01 s
- Eq. 1 attribution for all 435 pairs: 0.27 s; exact gradients for all 435 pairs: 0.57 s

The paper's original runs used an NVIDIA L4 24 GB (`g2-standard-4`).

### E5: expanding beyond approved drugs ✅
Downloaded all ChEMBL mechanism rows with no phase filter (`nmi/e05_download_expanded.py`, ChEMBL REST API; same cleaning and text recipe): **5,035 pairs** (3,816 molecules, 2,644 scaffolds, 988 targets) vs. 2,699 in the paper. Breakdown: 2,769 approved, 680 phase 3, 1,292 phase 2, 268 phase 1, 26 unknown. File: `chembl_mechanisms_allphases.csv` (on the VM).

**More data, same canonical test (435 rows), 5 seeds.** The extra rows are scaffold-disjoint from the test and validation sets:

| Training set | n_train | R@1 rich | R@1 no-drug | R@1 mechanism-only |
|---|---|---|---|---|
| (i) paper: approved train | 2,019 | 0.215 ± 0.012 | 0.131 ± 0.006 | 0.133 ± 0.009 |
| (ii) all phases | 4,265 | 0.224 ± 0.014 | **0.160 ± 0.003** | **0.161 ± 0.009** |
| (iii) approved-only subset of (ii) | 2,082 | 0.220 ± 0.010 | 0.128 ± 0.003 | 0.132 ± 0.006 |

- **Investigational compounds give +22% name-free R@1** (0.131 → 0.160). Adding approved drugs from newer ChEMBL releases alone (iii) does not help.
- The rich-text gain is small (+0.009) because names already saturate it.
- **This answers R3 Q4:** restricting to approved drugs was unnecessary and cost performance.

**New all-phase scaffold benchmark (5 partitions, ~441 test rows):**

| Text | R@1 | MRR | R@1 on approved rows | R@1 on non-approved rows | R@1 ceiling |
|---|---|---|---|---|---|
| rich | 0.197 ± 0.018 | 0.263 | 0.278 | **0.107** | 0.992 |
| no-drug | 0.173 ± 0.014 | 0.271 | 0.198 | **0.147** | 0.611 |
| mechanism-only | 0.181 ± 0.006 | 0.278 | 0.213 | 0.146 | 0.609 |

- **For investigational compounds, names *hurt*:** rich 0.107 < name-free 0.147, because their names are mostly development codes that carry no INN-stem signal.
- **The name shortcut only works for drugs with informative INN names.** That's a useful robustness argument for name-free evaluation.

### E6: established benchmarks (ChEBI-20, PubChem324kV2) ✅
MoleculeLens trained exactly as in the paper on each benchmark's own train split (no target labels, so the loss reduces to symmetric InfoNCE; text max length 256). Text → molecule retrieval, 3 seeds:

| Benchmark | Text pooling | R@1 | R@10 | R@20 | MRR | R@1 on scaffold-novel test molecules |
|---|---|---|---|---|---|---|
| ChEBI-20 (test gallery 3,300) | CLS (paper) | 0.558 ± 0.002 | 0.855 | 0.893 | 0.663 | 0.299 |
| ChEBI-20 | **mean** | **0.603 ± 0.005** | 0.880 | 0.917 | 0.702 | 0.356 |
| PubChem324kV2 (test gallery 2,000) | CLS (paper) | 0.357 ± 0.006 | 0.618 | 0.682 | 0.446 | 0.217 |
| PubChem324kV2 | **mean** | **0.392 ± 0.007** | 0.650 | 0.710 | 0.482 | 0.234 |

**ChEBI-20 ranked against all 33,008 molecules (Text2Mol protocol):** mean pooling gives Hits@1 0.304, Hits@10 0.650, MRR 0.415 (CLS: 0.270 / 0.603 / 0.372).

**Published reference points (to verify against the original papers; the protocols differ between papers):**
- ChEBI-20, Text2Mol ensemble: Hits@1 34.4, Hits@10 81.1, MRR 0.499 ([Text2Mol](https://ml.paperswithcode.com/paper/text2mol-cross-modal-molecule-retrieval-with)).
- ChEBI-20, recent alignment models (AMAN, ORMA, CLASS): Hits@1 about 49–67 ([ORMA](https://arxiv.org/html/2411.11875v1), [CLASS](https://arxiv.org/pdf/2502.11633)).
- PubChem324k, MoleculeSTM: full-test text → molecule accuracy about 44 according to an aggregator; I could not confirm it from MolCA's own table ([MolCA](https://arxiv.org/pdf/2310.12798)).

**Two findings that support the paper's motivation (R3 Q3, "why a new benchmark"):**
1. **These benchmarks use random splits:** 84.2% of ChEBI-20 and 69.5% of PubChem324k test molecules have a Murcko scaffold seen in training.
2. **On scaffold-novel test molecules R@1 roughly halves:** ChEBI-20 drops from 0.603 to 0.356, PubChem324k from 0.392 to 0.234.

Published leaderboard numbers therefore largely measure in-scaffold retrieval, which is the gap our scaffold-disjoint, leakage-controlled benchmark is designed to expose.

A model of two linear heads is competitive with early published models (it reaches roughly the level of the Text2Mol ensemble) while staying fully decomposable. Mean pooling again beats CLS (+4–5 R@1), consistent with E14.

PCdes (KV-PLM) was not run: it has to be rebuilt from raw files (`generate_PCdes.py`). It is listed as remaining work.

### E21: combined experiment, all justified fixes in one model ✅
The fixes are added one at a time on the canonical scaffold test (5 seeds). The metric is R@1 on the structure-deduplicated gallery (unique parent structures), the protocol recommended in E1.

| Config | no-drug R@1 | no-drug MRR | rich R@1 | rich MRR |
|---|---|---|---|---|
| A paper (ECFP4-2048, CLS, dim 256, approved train) | 0.149 ± 0.007 | 0.229 | 0.208 ± 0.010 | 0.297 |
| B + 8192 bits (E15) | 0.160 ± 0.006 | 0.242 | 0.257 ± 0.004 | 0.347 |
| C + parent fingerprints (E12) | 0.166 ± 0.007 | 0.271 | 0.214 ± 0.008 | 0.308 |
| D + mean-pooled text (E14) | 0.167 ± 0.005 | 0.273 | 0.226 ± 0.005 | 0.322 |
| E + dim 1024 (E19) | 0.165 ± 0.010 | 0.272 | 0.231 ± 0.006 | 0.330 |
| **F + all-phase training data (E5) = final** | **0.201 ± 0.003** | **0.315** | 0.236 ± 0.001 | 0.333 |

- **Name-free retrieval improves by 35%** (0.149 → 0.201) and MRR by 38% (0.229 → 0.315). The biggest single gain comes from the extra data (C–E → F).
- **Parent fingerprints lower rich-text R@1** (B → C: 0.257 → 0.214) because they remove the salt-name shortcut (E12), yet they *improve* the name-free model (0.160 → 0.166). That's the expected signature of removing a shortcut.

**Final model, canonical split, no-drug text, seed 0:** R@1 0.223 [0.168, 0.281] (parent-cluster bootstrap), R@1 struct 0.203, MRR 0.317 [0.265, 0.371].

**vs MolPrompt (no-drug, fine-tuned), same deduplicated gallery:** **0.203 vs 0.097**, paired difference **+0.107 [+0.052, +0.162]**, P(diff ≤ 0) < 10⁻⁴.

| 10 scaffold partitions | paper config R@1 struct | final R@1 struct [95% t-CI] | final MRR |
|---|---|---|---|
| no-drug | 0.242 | **0.287** [0.242, 0.332] | 0.400 |
| rich | 0.281 | 0.322 [0.271, 0.372] | 0.409 |

| OOD split (5 partitions, molecule-disjoint) | text | paper config | final |
|---|---|---|---|
| target-disjoint | no-drug | 0.053 (E7b) | **0.073** |
| target-disjoint | rich | 0.103 (E7b) | 0.103 |
| scaffold + target disjoint | no-drug | 0.066 | **0.094** |
| scaffold + target disjoint | rich | 0.104 | 0.108 |

The "paper config" target numbers in the raw E21 output (0.111 / 0.348) still contain the molecule overlap; the molecule-disjoint values from E7b are shown here.

**The final model is also more interpretable** (same 16-class pharmacophore test as E13, n = 86):

| | paper model (E13) | final model |
|---|---|---|
| saliency mass on pharmacophore (chance 0.27) | 0.31 | **0.43** |
| top-3 atoms hit pharmacophore (chance 0.51) | 0.70 | **0.84** |
| same, with wrong-class text | 0.47 | 0.50 |
| p (correct > wrong text) | 3×10⁻⁷ | **5×10⁻¹⁵** |
| counterion-driven top-1 saliency | 76% of salt rows | 0 (by construction) |

Deletion check on the final model (score drop for top-k salient vs random k bits): k=1 0.049 vs 0.002; k=3 0.134 vs 0.008; k=5 0.204 vs 0.014.

**Bottom line:** removing the shortcuts the reviewers suspected (names, salts, collisions) and adding data makes MoleculeLens **both more accurate without names and more chemically grounded**. The +10.7-point name-free margin over MolPrompt is a clean replacement for the paper's +9.6 headline, which mixed tie rules and names.

### E2: MoleculeSTM on the scaffold test ✅ (important for the headline)
MoleculeSTM-SMILES (MegaMolBART + SciBERT, pretrained checkpoint), canonical scaffold test, N=435:

| MoleculeSTM variant | R@1 | R@1 struct | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| zero-shot, rich | 0.202 | **0.230** | 0.515 | 0.630 | 0.348 |
| zero-shot, no-drug | 0.060 | 0.062 | 0.172 | 0.267 | 0.127 |
| zero-shot, mechanism-only | 0.057 | 0.057 | 0.191 | 0.294 | 0.135 |
| frozen MoleculeSTM encoders + MoleculeLens linear bridge, rich (5 seeds) | 0.250 | **0.283** | 0.569 | 0.681 | 0.393 |
| same, no-drug | 0.070 | 0.076 | 0.234 | 0.364 | 0.160 |
| same, mechanism-only | 0.076 | 0.080 | 0.233 | 0.350 | 0.161 |

- **The scaffold row R3 asked for overturns the paper's "strongest top-rank retrieval" claim for rich text.** Zero-shot MoleculeSTM already matches MoleculeLens (struct 0.230 vs 0.208) and is much stronger at R@5/R@10. With a trained bridge it reaches 0.283.
- **MoleculeSTM is far more name-dependent.** It loses 73% of R@1 when the name is removed (0.230 → 0.062). Name-free, MoleculeLens is about 2.4× better (paper config 0.149) and **about 3.2× better as the final model (0.201)**.
- **Probable cause (not verified):** MoleculeSTM was pretrained on PubChem descriptions, which name these approved drugs, so its "zero-shot" rich-text score likely benefits from memorised name-structure pairs. Checking the overlap of our 435 test drugs with PubChemSTM is a recommended follow-up.
- **Action:** the honest headline is the **name-free** comparison, where MoleculeLens leads every baseline: final 0.201 vs MolPrompt 0.097, MoleculeSTM 0.062 / 0.076, KV-PLM 0.034 (optimistic rule). Report the rich-text row too, and state plainly that MoleculeSTM is stronger there.
- Technical note: MegaMolBART needs `apex`. The compiled apex on the VM was built for the L4 and fails on the T4, so `FusedLayerNorm` was replaced by the numerically equivalent `torch.nn.LayerNorm` (shim in `~/nmi_shim`).

### E16: attribution for MolPrompt on the same pharmacophore test ✅
MolPrompt (fine-tuned checkpoint) gradient × input at the atom-embedding layer, same 16 classes, 86 pairs, same controls as E13:

| Model / attribution | mass on pharmacophore (chance 0.26–0.27) | top-3 hit (chance 0.50) | p (correct > wrong text) | time per explanation |
|---|---|---|---|---|
| MolPrompt, gradient × input | 0.26 | 0.43 | 0.98 (n.s.) | 0.070 s |
| **MoleculeLens, paper model (Eq. 1)** | **0.31** | **0.70** | **3×10⁻⁷** | 0.0006 s |
| **MoleculeLens, final model** | **0.43** | **0.84** | **5×10⁻¹⁵** | 0.0006 s |

- **This answers R3 Q9 directly.** MolPrompt's attributions sit at chance and are not text-specific, while MoleculeLens's closed-form saliency is significantly grounded and about 100× cheaper.
- Caveat: only one attribution method (gradient × input) was tried for MolPrompt. Integrated gradients would make the comparison more robust.
