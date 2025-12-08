#!/usr/bin/env python3
"""
Phase transition visualization for retrieval imagination runs.

This script scans retrieval_imagination result directories, reads the stored
metrics.json files, and plots precision@k as a function of corruption severity
for the fragment (identity) baseline, reconstructed embeddings (MLP or RAE),
and clean references. It also reports the maximum corruption severity at which
each variant stays above a configurable precision threshold to highlight how
restoration extends the “recoverable” regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot phase transitions for retrieval imagination runs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory that contains retrieval_imagination_* folders.",
    )
    parser.add_argument(
        "--glob",
        default="retrieval_imagination_*",
        help="Glob pattern (relative to results-root) for selecting runs.",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Optional list of corruption scheme names to include (default: all).",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=0.9,
        help="Precision@k threshold used to define the recoverable regime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "phase_transition.png",
        help="Where to write the resulting plot.",
    )
    return parser.parse_args()


def load_runs(root: Path, pattern: str) -> List[Dict]:
    runs: List[Dict] = []
    for directory in sorted(root.glob(pattern)):
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        runs.append({"dir": directory, "metrics": metrics})
    return runs


def extract_curve_data(runs: List[Dict], schemes: List[str] | None) -> Dict[str, List[Tuple[float, Dict]]]:
    grouped: Dict[str, List[Tuple[float, Dict]]] = {}
    for run in runs:
        metrics = run["metrics"]
        scheme = metrics.get("scheme", "unknown")
        if schemes and scheme not in schemes:
            continue
        corrupt_range = metrics.get("corrupt_range") or [None, None]
        severity = corrupt_range[0] if corrupt_range[0] is not None else 0.0
        grouped.setdefault(scheme, []).append((severity, metrics.get("precision", {})))
    # sort by severity for consistent lines
    for curve_list in grouped.values():
        curve_list.sort(key=lambda x: x[0])
    return grouped


def recoverable_threshold(severities: List[float], values: List[float], threshold: float) -> float:
    """
    Returns the maximum severity where the precision stays above the threshold.
    If precision never exceeds the threshold, returns 0.
    """
    max_severity = 0.0
    for sev, val in zip(severities, values):
        if val is None:
            continue
        if val >= threshold:
            max_severity = sev
    return max_severity


def plot_phase_transitions(
    grouped_data: Dict[str, List[Tuple[float, Dict]]],
    threshold: float,
    output_path: Path,
) -> None:
    if not grouped_data:
        raise ValueError("No runs found for the given filters; cannot plot.")

    num_schemes = len(grouped_data)
    fig, axes = plt.subplots(num_schemes, 1, figsize=(7, 4 * num_schemes), sharex=False)
    if num_schemes == 1:
        axes = [axes]

    summary_lines = []

    for ax, (scheme, entries) in zip(axes, grouped_data.items()):
        severities = [sev for sev, _ in entries]
        frag = [precisions.get("fragment") for _, precisions in entries]
        recon = [precisions.get("reconstruction") for _, precisions in entries]
        clean = [precisions.get("clean") for _, precisions in entries]

        ax.plot(severities, frag, marker="o", label="fragment (identity)")
        ax.plot(severities, recon, marker="o", label="reconstruction (MLP/memory)")
        ax.plot(severities, clean, marker="o", label="clean reference")
        ax.axhline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"threshold={threshold}")
        ax.set_title(f"{scheme} corruption")
        ax.set_xlabel("corruption severity (min area fraction)")
        ax.set_ylabel("precision@k")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        frag_regime = recoverable_threshold(severities, frag, threshold)
        recon_regime = recoverable_threshold(severities, recon, threshold)
        summary_lines.append(
            f"{scheme}: fragment >= {threshold:.2f} up to {frag_regime:.2f}, "
            f"reconstruction up to {recon_regime:.2f}"
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved phase transition plot to", output_path)
    print("\nRecoverable regime summary (severity where precision drops below threshold):")
    for line in summary_lines:
        print(" -", line)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.results_root, args.glob)
    if not runs:
        raise FileNotFoundError(f"No runs found under {args.results_root} matching {args.glob}")
    grouped = extract_curve_data(runs, args.schemes)
    plot_phase_transitions(grouped, args.precision_threshold, args.output)


if __name__ == "__main__":
    main()
