**PCdes** (KV-PLM files; 15,000 pairs, split 10,500 / 1,500 / 3,000 exactly as `Ret/preprocess.py`). 73.3% of test molecules have a Murcko scaffold seen in training (2199 seen / 801 novel). Unparseable SMILES: 5. Sentence protocol: 12,405 sentences over 3,000 molecules.

Text → molecule, 3 seeds. R@k and MRR are over the full 3,000-molecule gallery; **Acc(64)** is KV-PLM's own metric, top-1 within randomly sampled 64-pair mini-batches, which is what the published 'Acc' columns report (their Recall@20 is the full-gallery number).

| Config | Protocol | Acc(64) | R@1 | R@5 | R@10 | R@20 | MRR | R@1 scaffold-novel |
|---|---|---|---|---|---|---|---|---|
| paper (2048, cls, 256) | paragraph | 63.18 | 0.292 ± 0.004 | 0.473 | 0.542 | 60.50 | 0.377 | 0.192 |
| paper (2048, cls, 256) | sentence | 58.92 | 0.244 ± 0.004 | 0.419 | 0.497 | 56.34 | 0.328 | 0.164 |
| final (parent 8192, mean, 1024) | paragraph | 59.95 | 0.304 ± 0.002 | 0.468 | 0.526 | 57.96 | 0.381 | 0.193 |
| final (parent 8192, mean, 1024) | sentence | 56.84 | 0.273 ± 0.001 | 0.430 | 0.492 | 54.70 | 0.348 | 0.177 |

Published PCdes text→molecule numbers, all **fine-tuned on the PCdes training set** (KV-PLM Table 1 / MoMu Fig. 3a; Acc = 64-pair mini-batch, Rec@20 = full 3k gallery):

| Model | sentence Acc(64) | sentence R@20 | paragraph Acc(64) | paragraph R@20 |
|---|---|---|---|---|
| Sci-BERT | 50.12 | 68.02 | 61.75 | 60.77 |
| KV-PLM | 54.22 | 71.80 | 64.95 | 64.27 |
| KV-PLM* | 55.61 | 74.77 | 77.03 | 75.47 |
| MoMu-S | 55.44 | 76.92 | 80.22 | 79.02 |
| MoMu-K | 54.94 | 78.29 | 81.45 | 80.62 |

**Names:** 75% of test descriptions open with a pronoun or article ('it is a ...'), so PCdes was built with the molecule name removed. Not verified against a real name list — an observation, not a measurement.
