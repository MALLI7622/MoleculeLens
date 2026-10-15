**E5a - all-phase ChEMBL mechanism set:**

| Subset | rows | molecules | scaffolds | targets | with drug name |
|---|---|---|---|---|---|
| all | 5035 | 3816 | 2644 | 988 | 100% |
| approved (4) | 2769 | 2120 | 1229 | 612 | 100% |
| phase 3 | 680 | 517 | 459 | 348 | 100% |
| phase 2 | 1292 | 984 | 921 | 519 | 100% |
| phase 1 | 268 | 174 | 170 | 126 | 100% |
| unknown | 26 | 21 | 21 | 22 | 100% |

**E5b - same canonical test (435), more training data (5 seeds):**

| Training set | n_train | R@1 rich | R@1 struct rich | MRR rich | R@1 no-drug | R@1 mech-only |
|---|---|---|---|---|---|---|
| (i) canonical approved train | 2019 | 0.215 ± 0.012 | 0.202 | 0.297 | 0.131 ± 0.006 | 0.133 ± 0.009 |
| (ii) all-phase, scaffold-disjoint | 4265 | 0.224 ± 0.014 | 0.211 | 0.309 | 0.160 ± 0.003 | 0.161 ± 0.009 |
| (iii) approved-only subset of (ii) | 2082 | 0.220 ± 0.010 | 0.206 | 0.301 | 0.128 ± 0.003 | 0.132 ± 0.006 |

**E5c - all-phase scaffold benchmark (5 partitions):**

| Text | n_test | R@1 | R@1 struct | MRR | R@1 approved rows | R@1 non-approved rows | R@1 ceiling |
|---|---|---|---|---|---|---|---|
| rich | 441 | 0.197 ± 0.018 | 0.196 | 0.263 | 0.278 | 0.107 | 0.992 |
| nodrug | 441 | 0.173 ± 0.014 | 0.173 | 0.271 | 0.198 | 0.147 | 0.611 |
| mech | 441 | 0.181 ± 0.006 | 0.182 | 0.278 | 0.213 | 0.146 | 0.609 |
