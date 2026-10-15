**ChEBI-20** (train 26407 / test 3300; 84.2% of test molecules have a Murcko scaffold seen in training; 520 scaffold-novel test molecules):

| MoleculeLens text pooling (3 seeds) | R@1 | R@10 | R@20 | MRR | R@1 struct | R@1 scaffold-novel subset |
|---|---|---|---|---|---|---|
| cls | 0.558 ± 0.002 | 0.855 | 0.893 | 0.663 | 0.514 | 0.299 |
| mean | 0.603 ± 0.005 | 0.880 | 0.917 | 0.702 | 0.556 | 0.356 |

ChEBI-20 cls, ranked against all 33008 molecules (Text2Mol protocol): Hits@1 0.270, Hits@10 0.603, MRR 0.372

ChEBI-20 mean, ranked against all 33008 molecules (Text2Mol protocol): Hits@1 0.304, Hits@10 0.650, MRR 0.415

**PubChem324kV2** (train 12000 / test 2000; 69.5% of test molecules have a Murcko scaffold seen in training; 609 scaffold-novel test molecules):

| MoleculeLens text pooling (3 seeds) | R@1 | R@10 | R@20 | MRR | R@1 struct | R@1 scaffold-novel subset |
|---|---|---|---|---|---|---|
| cls | 0.357 ± 0.006 | 0.618 | 0.682 | 0.446 | 0.337 | 0.217 |
| mean | 0.392 ± 0.007 | 0.650 | 0.710 | 0.482 | 0.371 | 0.234 |

