**Equal training data, no-drug text, canonical test; R@1 on the structure-deduplicated gallery (paired parent-cluster bootstrap):**

| Training data | n_train | MoleculeLens | MolPrompt | difference [95% CI] | P(diff <= 0) | MolPrompt best epoch |
|---|---|---|---|---|---|---|
| paper data | 2,019 | 0.162 | 0.097 | +0.066 [+0.017, +0.113] | 0.0033 | 27 |
| all-phase, <= 150 atoms | 4,238 | 0.199 | 0.133 | +0.066 [+0.014, +0.118] | 0.0070 | 27 |
