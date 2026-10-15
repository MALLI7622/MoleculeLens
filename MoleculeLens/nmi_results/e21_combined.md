**Step-wise ablation, canonical scaffold test, 5 seeds; R@1 on the structure-deduplicated (unique parent) gallery:**

| Config | no-drug R@1 struct | no-drug MRR | rich R@1 struct | rich MRR |
|---|---|---|---|---|
| A paper | 0.149 ± 0.007 | 0.229 | 0.208 ± 0.010 | 0.297 |
| B +8192 bits | 0.160 ± 0.006 | 0.242 | 0.257 ± 0.004 | 0.347 |
| C +parent fp | 0.166 ± 0.007 | 0.271 | 0.214 ± 0.008 | 0.308 |
| D +mean-pool text | 0.167 ± 0.005 | 0.273 | 0.226 ± 0.005 | 0.322 |
| E +dim 1024 | 0.165 ± 0.010 | 0.272 | 0.231 ± 0.006 | 0.330 |
| F +all-phase train | 0.201 ± 0.003 | 0.315 | 0.236 ± 0.001 | 0.333 |

**Final model, canonical split, no-drug text (seed 0):** R@1 0.223 [0.168, 0.281], R@1 struct 0.203, MRR 0.317 [0.265, 0.371].
**vs MolPrompt no-drug** (same deduplicated gallery): 0.203 vs 0.097, diff +0.107 [+0.052, +0.162], P(diff<=0) = 0.0000.

| 10 scaffold partitions | paper config R@1 struct | final R@1 struct [95% t-CI] | final MRR |
|---|---|---|---|
| nodrug | 0.242 | 0.287 [0.242, 0.332] | 0.400 |
| rich | 0.281 | 0.322 [0.271, 0.372] | 0.409 |

| OOD split (5 partitions) | text | paper config R@1 struct | final R@1 struct |
|---|---|---|---|
| target | nodrug | 0.111 | 0.073 |
| target | rich | 0.348 | 0.103 |
| joint | nodrug | 0.066 | 0.094 |
| joint | rich | 0.104 | 0.108 |

**Final model interpretability:** pharmacophore mass 0.43 (wrong text 0.26, chance 0.27); top-3 hit 0.84 (wrong 0.50, chance 0.51); n=86; p(>chance)=4.5e-15, p(>wrong text)=5.0e-15. Counterion saliency: 0 by construction (parent fingerprints).
Deletion (score drop, top-k salient vs random k): k=1: 0.049 vs 0.002, k=3: 0.134 vs 0.008, k=5: 0.204 vs 0.014
