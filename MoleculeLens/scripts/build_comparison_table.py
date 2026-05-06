#!/usr/bin/env python3
"""Build a Markdown comparison table for retrieval models.

Usage:
    python scripts/build_comparison_table.py \
        --input "MoleculeLens=/abs/path/to/moleculelens.json" \
        --input "MoleculeSTM=/abs/path/to/readme_experiments.md" \
        --input "MolPrompt=/abs/path/to/molprompt.json" \
        --leakage-input "MoleculeLens=/abs/path/with.json|/abs/path/without.json" \
        --out /abs/path/to/model_comparison_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METRIC_KEYS = (
    "s2t_t4",
    "s2t_t10",
    "s2t_t20",
    "t2s_t4",
    "t2s_t10",
    "t2s_t20",
)

DISPLAY_NAMES = {
    "s2t_t4": "S→T T=4",
    "s2t_t10": "S→T T=10",
    "s2t_t20": "S→T T=20",
    "t2s_t4": "T→S T=4",
    "t2s_t10": "T→S T=10",
    "t2s_t20": "T→S T=20",
}

JSON_ALIASES = {
    "s2t_t4": ("s2t_t4", "T4_S2T", "t4_s2t"),
    "s2t_t10": ("s2t_t10", "T10_S2T", "t10_s2t"),
    "s2t_t20": ("s2t_t20", "T20_S2T", "t20_s2t"),
    "t2s_t4": ("t2s_t4", "T4_T2S", "t4_t2s"),
    "t2s_t10": ("t2s_t10", "T10_T2S", "t10_t2s"),
    "t2s_t20": ("t2s_t20", "T20_T2S", "t20_t2s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help='Repeated model spec in the form "ModelName=/abs/path/to/file"',
    )
    parser.add_argument(
        "--leakage-input",
        action="append",
        default=[],
        help='Optional paired spec in the form "ModelName=/abs/path/with|/abs/path/without"',
    )
    parser.add_argument(
        "--pending-leakage",
        action="append",
        default=[],
        help='Optional pending leakage note in the form "ModelName=status text"',
    )
    parser.add_argument(
        "--leakage-metric",
        default="t2s_t4",
        choices=METRIC_KEYS,
        help="Metric key to use for the leakage-ablation table",
    )
    parser.add_argument("--out", required=True, help="Output Markdown file path")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional JSON manifest path with metrics and exact source paths",
    )
    return parser.parse_args()


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --input spec: {spec!r}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip()).expanduser()
    if not name:
        raise ValueError(f"Missing model name in --input spec: {spec!r}")
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return name, path


def parse_leakage_spec(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --leakage-input spec: {spec!r}")
    name, raw_paths = spec.split("=", 1)
    name = name.strip()
    if "|" not in raw_paths:
        raise ValueError(f"Invalid --leakage-input spec: {spec!r}")
    with_raw, without_raw = raw_paths.split("|", 1)
    with_path = Path(with_raw.strip()).expanduser()
    without_path = Path(without_raw.strip()).expanduser()
    if not name:
        raise ValueError(f"Missing model name in --leakage-input spec: {spec!r}")
    if not with_path.exists():
        raise FileNotFoundError(f"Leakage input path does not exist: {with_path}")
    if not without_path.exists():
        raise FileNotFoundError(f"Leakage input path does not exist: {without_path}")
    return name, with_path, without_path


def parse_note_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid note spec: {spec!r}")
    name, note = spec.split("=", 1)
    name = name.strip()
    note = note.strip()
    if not name:
        raise ValueError(f"Missing model name in note spec: {spec!r}")
    if not note:
        raise ValueError(f"Missing note text in note spec: {spec!r}")
    return name, note


def maybe_scale_percent(metrics: dict[str, float]) -> dict[str, float]:
    values = list(metrics.values())
    if values and max(values) <= 1.5:
        return {k: v * 100.0 for k, v in metrics.items()}
    return metrics


def extract_json_metrics(data: dict) -> dict[str, float]:
    candidate = data.get("metrics", data)
    metrics: dict[str, float] = {}
    for target_key, aliases in JSON_ALIASES.items():
        for alias in aliases:
            if alias in candidate:
                metrics[target_key] = float(candidate[alias])
                break
        else:
            raise KeyError(f"Missing metric {target_key!r} in JSON input")
    return maybe_scale_percent(metrics)


def extract_csv_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    wanted = {
        "T=4 S->T": "s2t_t4",
        "T=10 S->T": "s2t_t10",
        "T=20 S->T": "s2t_t20",
        "T=4 T->S": "t2s_t4",
        "T=10 T->S": "t2s_t10",
        "T=20 T->S": "t2s_t20",
    }
    metrics: dict[str, float] = {}
    for row in rows:
        metric_name = row.get("Metric", "").strip()
        if metric_name not in wanted:
            continue
        value = row.get("ThinBridges")
        if value in (None, ""):
            raise KeyError(f"Missing ThinBridges value for {metric_name!r} in {path}")
        metrics[wanted[metric_name]] = float(value)
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise KeyError(f"Missing metrics in CSV input {path}: {missing}")
    return maybe_scale_percent(metrics)


def extract_moleculestm_readme_metrics(text: str) -> dict[str, float]:
    pattern = re.compile(
        r"\|\s*Given Structure → Retrieve Text\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*\n"
        r"\|\s*Given Text → Retrieve Structure\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    )
    match = pattern.search(text)
    if not match:
        raise ValueError("Could not find ChEMBL retrieval table in README input")
    s2t_t4, s2t_t10, s2t_t20, t2s_t4, t2s_t10, t2s_t20 = map(float, match.groups())
    return {
        "s2t_t4": s2t_t4,
        "s2t_t10": s2t_t10,
        "s2t_t20": s2t_t20,
        "t2s_t4": t2s_t4,
        "t2s_t10": t2s_t10,
        "t2s_t20": t2s_t20,
    }


def load_metrics(path: Path) -> dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return extract_json_metrics(json.load(handle))
    if suffix == ".md":
        return extract_moleculestm_readme_metrics(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return extract_csv_metrics(path)
    raise ValueError(f"Unsupported input format for {path}")


def format_value(value: float) -> str:
    return f"{value:.2f}"


def render_table(models: list[tuple[str, dict[str, float]]]) -> str:
    header = [
        "| Model | S→T T=4 | S→T T=10 | S→T T=20 | T→S T=4 | T→S T=10 | T→S T=20 | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for name, metrics in models:
        avg = sum(metrics[key] for key in METRIC_KEYS) / len(METRIC_KEYS)
        row = [name] + [format_value(metrics[key]) for key in METRIC_KEYS] + [format_value(avg)]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(header + rows)


def render_leakage_table(
    all_model_names: list[str],
    leakage_rows: dict[str, tuple[float, float]],
    pending_rows: dict[str, str],
    metric_key: str,
) -> str:
    metric_label = DISPLAY_NAMES[metric_key]
    header = [
        f"| Model | {metric_label} with names | {metric_label} without names | Abs Drop | % Drop | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rows = []
    for name in all_model_names:
        if name not in leakage_rows:
            status = pending_rows.get(name, "Not provided")
            rows.append(f"| {name} | N/A | N/A | N/A | N/A | {status} |")
            continue
        with_value, without_value = leakage_rows[name]
        abs_drop = with_value - without_value
        pct_drop = (abs_drop / with_value * 100.0) if with_value else 0.0
        rows.append(
            "| " + " | ".join(
                [
                    name,
                    format_value(with_value),
                    format_value(without_value),
                    format_value(abs_drop),
                    format_value(pct_drop),
                    "Complete",
                ]
            ) + " |"
        )
    return "\n".join(header + rows)


def render_main_sources_table(model_sources: list[tuple[str, Path]]) -> str:
    header = [
        "| Model | Source |",
        "|---|---|",
    ]
    rows = [f"| {name} | `{path}` |" for name, path in model_sources]
    return "\n".join(header + rows)


def render_leakage_sources_table(
    all_model_names: list[str],
    leakage_sources: dict[str, tuple[Path, Path]],
    pending_rows: dict[str, str],
) -> str:
    header = [
        "| Model | With Names Source | Without Names Source | Status |",
        "|---|---|---|---|",
    ]
    rows = []
    for name in all_model_names:
        if name in leakage_sources:
            with_path, without_path = leakage_sources[name]
            rows.append(f"| {name} | `{with_path}` | `{without_path}` | Complete |")
        else:
            status = pending_rows.get(name, "Not provided")
            rows.append(f"| {name} |  |  | {status} |")
    return "\n".join(header + rows)


def build_manifest(
    models: list[tuple[str, dict[str, float]]],
    model_sources: list[tuple[str, Path]],
    leakage_rows: dict[str, tuple[float, float]],
    leakage_sources: dict[str, tuple[Path, Path]],
    pending_rows: dict[str, str],
    leakage_metric: str,
) -> dict:
    source_map = {name: path for name, path in model_sources}
    model_entries = []
    for name, metrics in models:
        avg = sum(metrics[key] for key in METRIC_KEYS) / len(METRIC_KEYS)
        model_entries.append(
            {
                "model": name,
                "metrics": metrics,
                "avg": avg,
                "source": str(source_map[name]),
            }
        )

    leakage_entries = []
    for name, _metrics in models:
        if name in leakage_rows:
            with_value, without_value = leakage_rows[name]
            abs_drop = with_value - without_value
            pct_drop = (abs_drop / with_value * 100.0) if with_value else 0.0
            with_path, without_path = leakage_sources[name]
            leakage_entries.append(
                {
                    "model": name,
                    "metric": leakage_metric,
                    "with_names": with_value,
                    "without_names": without_value,
                    "abs_drop": abs_drop,
                    "pct_drop": pct_drop,
                    "status": "Complete",
                    "with_source": str(with_path),
                    "without_source": str(without_path),
                }
            )
        else:
            leakage_entries.append(
                {
                    "model": name,
                    "metric": leakage_metric,
                    "with_names": None,
                    "without_names": None,
                    "abs_drop": None,
                    "pct_drop": None,
                    "status": pending_rows.get(name, "Not provided"),
                    "with_source": None,
                    "without_source": None,
                }
            )

    return {
        "metrics_unit": "percent",
        "main_results": model_entries,
        "leakage_metric": leakage_metric,
        "leakage_results": leakage_entries,
    }


def main() -> None:
    args = parse_args()
    models: list[tuple[str, dict[str, float]]] = []
    model_sources: list[tuple[str, Path]] = []
    for spec in args.input:
        name, path = parse_input_spec(spec)
        model_sources.append((name, path))
        models.append((name, load_metrics(path)))

    leakage_rows: dict[str, tuple[float, float]] = {}
    leakage_sources: dict[str, tuple[Path, Path]] = {}
    for spec in args.leakage_input:
        name, with_path, without_path = parse_leakage_spec(spec)
        with_metrics = load_metrics(with_path)
        without_metrics = load_metrics(without_path)
        leakage_rows[name] = (
            with_metrics[args.leakage_metric],
            without_metrics[args.leakage_metric],
        )
        leakage_sources[name] = (with_path, without_path)

    pending_rows = dict(parse_note_spec(spec) for spec in args.pending_leakage)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Comparison Table",
        "",
        "Metrics are reported as percentages.",
        "",
        render_table(models),
        "",
        "## Main Result Sources",
        "",
        render_main_sources_table(model_sources),
        "",
    ]
    if args.leakage_input or args.pending_leakage:
        lines.extend(
            [
                "## Drug-Name Leakage Ablation",
                "",
                render_leakage_table(
                    [name for name, _ in models],
                    leakage_rows,
                    pending_rows,
                    args.leakage_metric,
                ),
                "",
                "## Leakage Result Sources",
                "",
                render_leakage_sources_table(
                    [name for name, _ in models],
                    leakage_sources,
                    pending_rows,
                ),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.manifest_out:
        manifest_path = Path(args.manifest_out).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            models=models,
            model_sources=model_sources,
            leakage_rows=leakage_rows,
            leakage_sources=leakage_sources,
            pending_rows=pending_rows,
            leakage_metric=args.leakage_metric,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
