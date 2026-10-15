| Model | Protocol | R@1 [95% CI, parent-cluster] | R@5 | R@10 | MRR |
|---|---|---|---:|---:|---:|
| MoleculeLens rich | trained on scaffold train split | 0.225 [0.182, 0.271] | 0.384 | 0.464 | 0.303 |
| MoleculeLens same-model no-name | trained (rich) | 0.067 [0.040, 0.095] | 0.159 | 0.253 | 0.121 |
| MoleculeLens retrained no-drug | trained on scaffold train split | 0.145 [0.108, 0.184] | 0.359 | 0.464 | 0.244 |
| [paper] retrained no-drug (mismatched) | - | 0.110 [0.079, 0.145] | 0.285 | 0.370 | 0.198 |
| MoleculeLens mechanism-only | trained on scaffold train split | 0.124 [0.092, 0.159] | 0.331 | 0.453 | 0.226 |
| MolPrompt rich | fine-tuned on scaffold train split | 0.140 [0.108, 0.176] | 0.400 | 0.533 | 0.265 |
| MolPrompt no-drug | fine-tuned on scaffold train split | 0.074 [0.049, 0.102] | 0.287 | 0.418 | 0.182 |
| KV-PLM rich | zero-shot (pretrained retrieval ckpt) | 0.055 [0.037, 0.081] | 0.172 | 0.251 | 0.121 |
| KV-PLM no-drug | zero-shot (pretrained retrieval ckpt) | 0.034 [0.021, 0.056] | 0.110 | 0.179 | 0.084 |

R@1 ceilings (identical query texts): rich: 0.995 (433 unique), nodrug: 0.448 (195 unique), mech: 0.444 (193 unique)

Paired R@1 differences (parent-cluster bootstrap):
- ML rich - MolPrompt rich: +0.085 [+0.035, +0.134], P(diff<=0)=0.0004
- ML retrained no-drug - MolPrompt no-drug: +0.071 [+0.031, +0.110], P(diff<=0)=0.0003
- ML retrained no-drug - MolPrompt rich: +0.005 [-0.042, +0.051], P(diff<=0)=0.4475

Tie handling - R@1 under: optimistic (shared eval) | random tie-break | strict | structure-deduplicated gallery (unique SMILES, random tie-break):
- MoleculeLens rich: 0.225 | 0.179 | 0.147 | 0.208
- MoleculeLens same-model no-name: 0.067 | 0.048 | 0.037 | 0.059
- MoleculeLens retrained no-drug: 0.145 | 0.107 | 0.076 | 0.139
- [paper] retrained no-drug (mismatched): 0.110 | 0.093 | 0.078 | 0.110
- MoleculeLens mechanism-only: 0.124 | 0.095 | 0.071 | 0.120
- MolPrompt rich: 0.140 | 0.130 | 0.122 | 0.166
- MolPrompt no-drug: 0.074 | 0.070 | 0.067 | 0.087
- 87 test rows share an identical ECFP4 vector with another test row
