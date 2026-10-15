| Text variant (retrained, 5 seeds) | R@1 | R@1 struct | R@5 | MRR |
|---|---|---|---|---|
| rich | 0.214 ± 0.013 | 0.202 ± 0.011 | 0.379 ± 0.006 | 0.297 ± 0.009 |
| nodrug | 0.130 ± 0.007 | 0.127 ± 0.007 | 0.333 ± 0.007 | 0.228 ± 0.005 |
| stem | 0.169 ± 0.005 | 0.161 ± 0.004 | 0.365 ± 0.011 | 0.260 ± 0.006 |
| shuffled | 0.093 ± 0.007 | 0.089 ± 0.007 | 0.233 ± 0.008 | 0.164 ± 0.006 |
| nameonly | 0.195 ± 0.014 | 0.182 ± 0.011 | 0.335 ± 0.011 | 0.271 ± 0.012 |
| mech | 0.130 ± 0.005 | 0.128 ± 0.005 | 0.334 ± 0.003 | 0.229 ± 0.003 |

Same rich model (seed 0), text changed at inference: rich: R@1 0.200 (struct 0.190), nodrug: R@1 0.094 (struct 0.089), stem: R@1 0.097 (struct 0.087), shuffled: R@1 0.060 (struct 0.057)

Test drugs whose 5-letter name stem also appears among training drug names: 69.7%
