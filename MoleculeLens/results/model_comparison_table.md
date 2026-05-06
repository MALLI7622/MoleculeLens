# MoleculeLens Camera-Ready Comparison

Primary retrieval comparison should be based on full-gallery retrieval metrics
(`Recall@1`, `MRR`, `Recall@5`, `Recall@10`).

## Primary Retrieval Comparison

| Model | Eval Set | Recall@1 | MRR | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|
| MoleculeSTM (zero-shot) | Full gallery from earlier head-to-head run (N=2699) | 0.089 | 0.171 | 0.243 | 0.337 |
| Thin Bridges (Global) | Full gallery from earlier head-to-head run (N=2699; train=test) | 0.747 | 0.849 | 0.985 | 0.997 |
| MoleculeLens (ours) | Held-out scaffold test from saved outputs (N=435) | 0.225 | 0.304 | 0.384 | 0.462 |
| MolPrompt (Scaffold) | Held-out shared/scaffold split from saved NPZ (N=435) | 0.129 | 0.256 | 0.400 | 0.531 |
| KV-PLM (Scaffold) | Held-out aligned scaffold test (N=435) | 0.055 | 0.121 | 0.172 | 0.251 |

## Drug-Name Leakage Ablation (Recall@1)

| Model | Eval Set | With Drug Name | No Drug Name | Abs Drop | % Drop |
|---|---|---:|---:|---:|---:|
| MoleculeSTM (zero-shot) | Full gallery from earlier head-to-head run (N=2699) | 0.089 | 0.014 | 0.074 | 83.8% |
| Thin Bridges (Global) | Full gallery from earlier head-to-head run (N=2699; train=test) | 0.747 | 0.145 | 0.601 | 80.5% |
| MoleculeLens (ours) | Held-out scaffold test from saved outputs (N=435) | 0.225 | 0.110 | 0.115 | 51.0% |
| MolPrompt (Scaffold) | Held-out shared/scaffold split from saved NPZ (N=435) | 0.129 | 0.071 | 0.057 | 44.6% |
| KV-PLM (Scaffold) | Held-out aligned scaffold test (N=435) | 0.055 | 0.034 | 0.021 | 37.5% |

## Secondary T-Choose-One Reference

These are useful as a secondary diagnostic, but should not replace
full-gallery retrieval metrics as the primary ranking criterion.

| Model | Eval Set | T=4 S->T | T=4 T->S | T=10 S->T | T=10 T->S | T=20 S->T | T=20 T->S |
|---|---|---:|---:|---:|---:|---:|---:|
| MoleculeSTM (zero-shot) | Full gallery from earlier head-to-head run (N=2699) | 0.926 | 0.933 | 0.830 | 0.862 | 0.739 | 0.763 |
| Thin Bridges (Global) | Full gallery from earlier head-to-head run (N=2699; train=test) | 1.000 | 0.999 | 1.000 | 0.997 | 1.000 | 0.997 |
| MoleculeLens (ours) | Held-out scaffold test from saved outputs (N=435) | 0.738 | 0.720 | 0.586 | 0.598 | 0.510 | 0.503 |
| MolPrompt (Scaffold) | Held-out shared/scaffold split from saved NPZ (N=435) | 0.834 | 0.834 | 0.715 | 0.701 | 0.577 | 0.554 |
| KV-PLM (Scaffold) | Held-out aligned scaffold test (N=435) | 0.639 | 0.657 | 0.430 | 0.432 | 0.310 | 0.301 |

## Notes

- `Thin Bridges (Global)` is included only as a memorization reference because it uses train=test.
- `MolPrompt (Global)` uses the same full-gallery train=val=test protocol to provide a like-for-like memorization-heavy reference.
- `KV-PLM (Global)` and `Graphormer (zero-shot, Global)` are full-gallery zero-shot evaluations over the same 2,699-pair gallery.
- `MoleculeLens (ours)` is the scientifically valid held-out result from the saved scaffold outputs.
- `MoleculeSTM (zero-shot)` and `Thin Bridges (Global)` come from the earlier head-to-head CSVs.
- Random Recall@1 differs by gallery size: about `1/2699 = 0.0004` for the full-gallery runs and `1/435 = 0.0023` for the scaffold test set.

## Sources

| Model | With Drug Source | No Drug Source |
|---|---|---|
| MoleculeSTM (zero-shot) | `/home/cheriearjun/MoleculeLens/outputs/comparison_withdrug/comparison_results.csv` | `/home/cheriearjun/MoleculeLens/outputs/comparison_nodrug/comparison_results.csv` |
| Thin Bridges (Global) | `/home/cheriearjun/MoleculeLens/outputs/comparison_withdrug/comparison_results.csv` | `/home/cheriearjun/MoleculeLens/outputs/comparison_nodrug/comparison_results.csv` |
| MoleculeLens (ours) | `/home/cheriearjun/MoleculeLens/outputs/Bt_test.pt` | `/home/cheriearjun/MoleculeLens/outputs/Bt_test_nodrug.pt` |
| MolPrompt (Scaffold) | `/home/cheriearjun/MolPrompt/save_model/retrieval_chembl_sharedsplit_bs6/chembl_retrieval_best_metrics.npz` | `/home/cheriearjun/MolPrompt/save_model/retrieval_chembl_sharedsplit_bs6_nodrug/chembl_retrieval_best_metrics.npz` |
| KV-PLM (Scaffold) | `/home/cheriearjun/MoleculeLens/results/kv_plm_aligned.json` | `/home/cheriearjun/MoleculeLens/results/kv_plm_aligned_nodrug.json` |
