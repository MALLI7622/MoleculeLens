| k bits zeroed | score drop: top-k salient (proxy) | top-k (exact) | random k | least-salient k | top>random (pairs) | rank-1 kept: top / random / least |
|---|---|---|---|---|---|---|
| 1 | 0.0770 | 0.0787 | 0.0049 | -0.0384 | 99.8% | 8.3% / 16.4% / 26.4% |
| 3 | 0.1642 | 0.1645 | 0.0138 | -0.0890 | 100.0% | 4.1% / 14.1% / 35.4% |
| 5 | 0.2328 | 0.2299 | 0.0230 | -0.1247 | 100.0% | 0.7% / 12.6% / 43.4% |
| 10 | 0.3641 | 0.3488 | 0.0480 | -0.1811 | 100.0% | 0.0% / 9.6% / 58.2% |

Atom-level: deleting the atoms of the top-1 salient environment in 68 correctly retrieved molecules loses the rank-1 hit in 57 (83.8%); mean score drop 0.2062.
