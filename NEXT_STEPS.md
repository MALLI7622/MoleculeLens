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
