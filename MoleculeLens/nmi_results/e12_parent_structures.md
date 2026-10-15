Share of salt rows whose top-1 salient bit is counterion-only: rich model 76%, same rich model with the name removed at inference 50%, no-drug model 3%. 180/204 salt rows have the salt named in the drug name.

Paper checkpoint: 204/435 test rows contain a salt or counterion; top-1 salient bit is counterion-only in 156 of them, a counterion-only bit is in the top 6 for 186; mean |attribution| mass on counterion-only bits 22.4%.

| Fingerprint (10 seeds) | R@1 | R@1 struct (unique SMILES) | R@1 struct (unique parent) | R@5 | MRR |
|---|---|---|---|---|---|
| original | 0.213 ± 0.010 | 0.201 ± 0.009 | 0.206 ± 0.009 | 0.384 ± 0.009 | 0.299 ± 0.007 |
| parent | 0.206 ± 0.006 | 0.160 ± 0.006 | 0.182 ± 0.006 | 0.342 ± 0.008 | 0.273 ± 0.004 |

| Fig. 3 case | expected | paper model top-3 envs | hits pharmacophore | parent model top-3 envs | hits pharmacophore |
|---|---|---|---|---|---|
| MAFENIDE ACETATE | sulfonamide | CC(=O)O, CC, S | True | cccc(c)S, S, O=S | True |
| CARBAMAZEPINE | dibenzazepine carboxamide | cc(c)N(C(N)=O)c(c)c, cN(c)C(N)=O, CN | True | CN(c)c, cN(c)C(N)=O, cc(c)N(C(N)=O)c(c)c | True |
| ALBUTEROL | catecholamine-like ethanolamine | CCN, CC(c)O, cccc(c)O | True | CC(C)(C)N, CC(c)O, CNC | True |
| ACYCLOVIR SODIUM | guanine (purine) base | Na, cc(n)n(cn)CO, COCCO | False | cc([O-])nc(n)N, COCCO, cn(c)C | False |
