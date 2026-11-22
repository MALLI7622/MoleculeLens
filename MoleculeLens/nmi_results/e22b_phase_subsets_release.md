**Which phases to train on: canonical scaffold test (435), final config, 5 seeds; R@1 on the structure-deduplicated gallery:**

| Training set | n_train | no-drug R@1 struct | no-drug MRR | rich R@1 struct | rich MRR |
|---|---|---|---|---|---|
| paper approved train (2,019) | 2,019 | 0.165 ± 0.010 | 0.272 | 0.231 ± 0.006 | 0.330 |
| approved only (phase 4) | 2,082 | 0.171 ± 0.009 | 0.278 | 0.223 ± 0.012 | 0.324 |
| phase >= 3 | 2,724 | 0.173 ± 0.009 | 0.288 | 0.213 ± 0.009 | 0.316 |
| all phases (E21 final) | 4,265 | 0.201 ± 0.003 | 0.315 | 0.236 ± 0.001 | 0.333 |
| all phases, <= 150 atoms (MolPrompt E22 train) | 4,238 | 0.191 ± 0.006 | 0.303 | 0.230 ± 0.007 | 0.323 |

**10 scaffold partitions (seed 0), R@1 struct [95% t-CI]:**

| Training set | no-drug | rich |
|---|---|---|
| approved only (phase 4) | 0.256 [0.208, 0.303] | 0.304 [0.257, 0.351] |
| phase >= 3 | 0.263 [0.217, 0.310] | 0.306 [0.258, 0.354] |
| all phases (E21 final) | 0.287 [0.242, 0.332] | 0.322 [0.271, 0.372] |

Released weights (MoleculeLens/nmi_models/): moleculelens_final_nodrug_canonical.pt, moleculelens_final_nodrug_alldata.pt, moleculelens_final_rich_canonical.pt, moleculelens_final_rich_alldata.pt
