# MoleculeLens

Canonical paper-artifact refresh from saved checkpoints and cached outputs:

```bash
cd /home/cheriearjun/MoleculeLens
bash scripts/run_paper_artifact_refresh.sh
```

This is the single entrypoint for regenerating the camera-ready tables and
figures. It refreshes:

- `results/model_comparison_table.md`
- `results/model_comparison_manifest.json`
- `results/attribution_validation.json`
- `results/incorrect_attribution_summary.json`
- `results/full_loss_multiseed_summary.json`
- `results/mechanism_only_ablation.json`
- `outputs/track1/leakage_per_pair.csv`
- `outputs/track1/attribution_validation_per_pair.csv`
- `outputs/track1/incorrect_attribution_summary.csv`
- `outputs/track1/wrong_close_analysis.csv`
- `outputs/robustness/bootstrap_ci.csv`
- `outputs/robustness/full_loss_multiseed_results.csv`
- `outputs/mechanism_only/`
- `outputs/track3/recall_by_layer.csv`
- `outputs/track3/recall_by_layer_family.csv`
- `outputs/track3/run_metadata.json`
- `results/paper_artifact_manifest.json`
- `MoleculeLens-paper/figures/*.pdf`
- `MoleculeLens-paper/neurips_2026.pdf`

The script runs inside the `MoleculeSTM` conda environment and assumes the
saved projection weights and cached outputs already present in this repo are the
intended paper sources of truth.

Writer responsibilities are intentionally split so each paper-facing artifact
has one canonical producer:

- `08_ecfp4_attribution.py`: raw Track 1 analysis only.
- `09b_diagonal_artifact_sync.py`: canonical Track 1 CSVs and bootstrap CI.
- `09c_attribution_validation.py`: validates Equation 2 against exact gradients and bit-drop scores.
- `09d_incorrect_attribution_diagnostic.py`: stratifies attribution fidelity by correct vs incorrect retrieval outcome.
- `08c_paper_figures.py`: Section 7 publication figures.
- `10_contrastive_logit_lens.py`: raw Track 3 exact-replay outputs only.
- `10b_logit_lens_figures.py`: Track 3 publication figures.
- `12_robustness.py`: raw robustness outputs only.
- `12c_full_loss_multiseed.py`: five-seed full-loss robustness check.
- `13_mechanism_only_summary.py`: mechanism-only ablation summary.
- `12b_robustness_figures.py`: robustness publication figure.
- `scripts/build_paper_comparison_table.py`: camera-ready comparison table and manifest.

`scripts/run_aligned_comparison_refresh.sh` is now baseline-only and does not
rewrite the camera-ready paper manifest.
