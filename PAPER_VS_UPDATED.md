# MoleculeLens: NeurIPS 2026 submission vs. updated results

Every claim in `neurips_2026.pdf` that the NMI experiments re-measured, with the paper's number, the updated number and the experiment that produced it. Full details: `NEXT_STEPS.md` §5; raw outputs: `MoleculeLens/nmi_results/`.

Unless stated otherwise: canonical scaffold test (435 pairs), text→molecule retrieval.
- **Optimistic rule:** the paper's evaluator. A tie with an identical fingerprint counts as a hit.
- **Dedup gallery** (recommended): one gallery entry per unique salt-stripped parent structure, so identical fingerprints can't inflate R@1 ("R@1 struct" in the results).

Status: ✅ holds · ⚠️ number changes · ❌ claim does not hold · ➕ new result.

## 1. Main retrieval results (Table 1, §3.1)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| MoleculeLens rich-text R@1 | 0.225 (one checkpoint) | **0.213 ± 0.010** (10 seeds); dedup gallery 0.201 ± 0.009 | ⚠️ | E8 |
| MoleculeLens rich-text MRR | 0.304 | 0.299 ± 0.007 (10 seeds) | ⚠️ | E8 |
| MolPrompt rich-text R@1 | 0.129 (MolPrompt's own scorer) | 0.140 (paper's optimistic rule); 0.166 (dedup gallery) | ⚠️ | E1 |
| R@1 margin over MolPrompt | +9.6 points | +8.5 [+3.5, +13.4] (same rule for both); **+4.2 (dedup gallery)** | ⚠️ | E1 |
| KV-PLM scaffold R@1 | 0.055 | 0.055; no-drug 0.034 (zero-shot, never trained on our data) | ✅ | E1, E3 |
| MoleculeSTM on the scaffold test | not reported (full gallery only, 0.089) | zero-shot R@1 0.202, **dedup 0.230, above MoleculeLens's 0.208**; with a MoleculeLens bridge 0.283 | ❌ for "strongest R@1" | E2 |
| "Strongest top-rank retrieval among baselines" | claimed for rich text | false with names (MoleculeSTM is higher). **True without names**: MoleculeLens 0.139 vs MolPrompt 0.087, MoleculeSTM 0.062 (dedup) | ❌ / ✅ name-free | E1, E2 |

## 2. Leakage controls (Table 2, §3.1, Fig. 2)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| Retrained no-drug R@1 | 0.110 | **0.145**. The paper's 0.110 scored no-drug text against the *rich* model's molecule embeddings (pairing bug in `09b`). 10-seed mean 0.129 ± 0.005 | ⚠️ bug | E1, E8 |
| Retrained leakage drop | 51.0% | **35.6%** on the canonical split; 27.7% [20.9, 34.5] over 10 scaffold partitions | ⚠️ | E1, E10–E11 |
| Same-model no-name R@1 (drop) | 0.067 (70.4%) | 0.067 (70.4%) on the canonical split; 59.4% [53.6, 65.2] over 10 partitions | ✅ / ⚠️ | E1, E11 |
| Mechanism-only R@1 / MRR | 0.124 / 0.226 | 0.124 / 0.226; 10-seed mean 0.130 ± 0.005 | ✅ | E1, E8 |
| Multi-seed leakage mean (Fig. 2C) | 22.9% | 27.7% [20.9, 34.5], using the paper's own split recipe (Fig. 2C used a different 90/10 recipe) | ⚠️ | E10–E11 |
| No-drug R@1 ceiling | not reported | 0.448. Only 195 of 435 no-drug texts are unique, so low absolute no-drug scores are partly the evaluation | ➕ | E1 |
| What in the name drives retrieval | not reported | Rich 0.214, name only 0.195, INN stem only 0.169, no name 0.130, shuffled names 0.093. 69.7% of test stems occur in training names | ➕ | E4 |
| Salt-name shortcut | not reported | 204/435 test rows are salts. Top-1 saliency is a counterion for 76% of them (rich model) vs 3% (trained without names). About 2.4 points of dedup-gallery R@1 come from it | ➕ | E12 |

## 3. Robustness (Table 3)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| Multi-split mean R@1 | 0.277 ± 0.081 (5 splits, different recipe) | 0.289 [0.235, 0.344] (10 partitions, paper's own recipe) | ✅ | E10 |
| Is the canonical split favourable? | not addressed | No: partition 0 *is* the canonical split, and it is the 2nd-hardest of 10 | ➕ | E10 |
| Text-encoder pooling, fingerprint size, radius, dim | fixed (CLS, 2048 bits, r=2, 256-d) | 2048 bits: 99.3% of bits collide; 8192 bits gives rich R@1 0.263. Mean pooling beats CLS. Radius makes no difference. 1024-d helps slightly | ➕ | E14, E15, E18–E19 |

## 4. Interpretability (Table 4, Figs. 3–5, §3.2)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| Saliency vs exact gradient / bit-drop (Pearson) | 0.905 / 0.947 | not re-measured | – | – |
| Saliency is causally faithful | correlational only | Deleting the top-k salient bits drops the score ~16× more than deleting random bits (k=3: 0.164 vs 0.014). Top-5 deletion keeps 0.7% of rank-1 hits vs 12.6% for random | ✅ stronger | E17 |
| "High-saliency bits align with pharmacophores" | qualitative (4 examples) | Quantified on 86 pairs over 16 classes: top-3 atoms hit the pharmacophore 0.70 vs 0.50 chance and 0.47 with a wrong mechanism text (p = 3×10⁻⁷). Fails for nucleosides | ✅ modest | E13 |
| …for the final model | – | top-3 hit **0.84** vs 0.51 chance, 0.50 wrong text (p = 5×10⁻¹⁵) | ✅ | E21 |
| …for MolPrompt (gradient × input) | – | at chance: 0.43 vs 0.50 (p = 0.93) | ➕ | E16 |
| Fig. 3 examples (mafenide acetate, acyclovir sodium) | top bits shown as pharmacophore evidence | Top bits were the **counterions** (acetate, Na⁺). With salt-stripped fingerprints they move to the aryl sulfonamide and the guanine base. Fig. 3 must be regenerated | ❌ figure | E12 |
| Same-model Jaccard 0.356, near-miss 30/43 | as reported | not re-measured | – | – |

## 5. Text-side logit lens (§3.3, Fig. 6, Table 5)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| Retrieval signal appears only at layer 12 (layers 0–11 at chance) | claimed | **False.** That came from reusing the layer-12 head on other layers. With its own head per layer: layer 1 0.180, layer 11 0.213 ≥ layer 12 0.207. Mean-pooled layer-0 token embeddings (essentially bag of words) score best, 0.275 | ❌ | E14 |

## 6. Data and scope (§3, §4)

| Claim / number | Paper | Updated | Status | Source |
|---|---|---|---|---|
| Dataset | 2,699 approved-drug pairs | 5,035 pairs over all phases (2,769 approved, 680 phase 3, 1,292 phase 2, 268 phase 1, 26 unknown). This is every ChEMBL_37 mechanism annotation with a structure; the excluded 2,148 rows are mostly biologics with no SMILES | ➕ | E5, E5b |
| "Does not establish generalization to investigational compounds" | stated limitation | Adding non-approved compounds to training raises name-free R@1 0.131 → 0.160 (+22%). For investigational test compounds, names *hurt* (rich 0.107 < no-drug 0.147) | ➕ | E5 |
| Which phases to train on | – | **All of them.** Name-free R@1 (dedup, 5 seeds): approved only 0.171, phase ≥3 0.173, all phases **0.201**. Restricting to late-stage compounds gives up nearly the whole gain, so the paper should use every phase and say why | ➕ | E22b |
| Target-disjoint / joint splits | not reported | Target-split rich 0.342 is inflated (33% of test molecules appear in train). Molecule-disjoint: rich 0.103, no-drug 0.053; joint 0.102 / 0.060 | ➕ | E7, E7b |
| External benchmarks | not reported | ChEBI-20 R@1 0.603 (MRR 0.702); PubChem324kV2 0.392. 70–84% of their test scaffolds appear in training; scaffold-novel R@1 roughly halves | ➕ | E6 |

## 7. Final model and the name-free headline

Final configuration (E21): salt-stripped ECFP4 at 8,192 bits, mean-pooled S-Biomed-RoBERTa, 1024-d shared space, trained on the all-phase pool.

| Name-free R@1 (dedup gallery) | Paper config | Final config | Source |
|---|---|---|---|
| MoleculeLens, canonical test (5 seeds) | 0.149 | **0.201 ± 0.003** | E21, E22b |
| MoleculeLens, 10 scaffold partitions | 0.242 | **0.287** [0.242, 0.332] | E21 |
| Pharmacophore top-3 hit | 0.70 | **0.84** | E13, E21 |

### Training-set size, and comparing fairly

The final model trains on 4,265 pairs; MolPrompt's published run used the paper's 2,019, so the overnight summary's +10.7 mixed an architecture gain with a data gain. E22c separates them by training both models on exactly the same rows.

Note: MolPrompt's Graphormer encoder runs out of memory on the T4 for molecules above 150 atoms, so the equal-data runs drop 27 training rows (4,265 → 4,238). That costs MoleculeLens about 1 point (0.201 → 0.191 ± 0.006), and both models are trained on the identical capped set.

**Equal training data** (E22c; both models trained on exactly the same rows, no-drug text, dedup gallery, paired parent-cluster bootstrap):

| Training data | n_train | MoleculeLens | MolPrompt | difference [95% CI] | P(diff ≤ 0) |
|---|---|---|---|---|---|
| paper data | 2,019 | 0.162 | 0.097 | **+0.066** [+0.017, +0.113] | 0.0033 |
| all-phase, ≤150 atoms | 4,238 | 0.199 | 0.133 | **+0.066** [+0.014, +0.118] | 0.0070 |

MolPrompt gains from the extra data much as MoleculeLens does (0.097 → 0.133), and the margin is **+6.6 points in both regimes**. So the two effects are separable and should be reported separately: the architecture is worth about +6.6 points, the data about +3.7. **Quote +6.6 [+1.4, +11.8], not the +10.7 from the overnight summary**, which compared the final model against a MolPrompt trained on half as much data.

**Headline for the NMI version:** an interpretable *leakage audit*. Names, INN stems, salt words and molecule overlap inflate every model's score. MoleculeLens's closed-form lens exposes these shortcuts. Without them it is the strongest name-free model, and the only one whose explanations are chemically grounded.
