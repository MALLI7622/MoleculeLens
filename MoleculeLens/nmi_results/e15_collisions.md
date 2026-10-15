2048-bit ECFP4 over the 2,699 molecules: 2033/2048 active bits (99.3%) host more than one distinct environment (mean 7.17 environments per bit).
Bits appearing in a test top-10 saliency list: 1189 distinct, 99.8% collided; 99.8% of top-10 (pair, bit) entries fall on collided bits.
Training exposure: median 50 training molecules per salient entry; 1.8% of entries on bits seen in <5 training molecules, 0.0% on bits never seen in training (3 of 2048 bits never fire in training).

| Fold size | collision rate | R@1 | R@1 struct | R@5 | MRR |
|---|---|---|---|---|---|
| 1024 | 100.0% | 0.207 ± 0.012 | 0.195 ± 0.011 | 0.374 ± 0.008 | 0.291 ± 0.007 |
| 2048 | 99.3% | 0.214 ± 0.013 | 0.202 ± 0.011 | 0.379 ± 0.006 | 0.297 ± 0.009 |
| 4096 | 89.1% | 0.251 ± 0.008 | 0.240 ± 0.007 | 0.414 ± 0.011 | 0.333 ± 0.006 |
| 8192 | 62.4% | 0.263 ± 0.006 | 0.252 ± 0.006 | 0.437 ± 0.013 | 0.347 ± 0.005 |
| 16384 | 37.6% | 0.258 ± 0.007 | 0.244 ± 0.007 | 0.438 ± 0.014 | 0.349 ± 0.007 |
