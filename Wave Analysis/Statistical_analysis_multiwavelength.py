# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:25:05 2026

@modified: sgao
"""
"""Compact replicate-level statistical analysis for the optical-flow results."""

from __future__ import annotations
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
COLORS = {"UV": "#7b2cbf", "BLUE": "#2878c8", "GREEN": "#2f9e44"}
ORDER = ["UV", "BLUE", "GREEN"]

ROOT = Path(r"C:/Users/sgao/OneDrive - ICIQ/Documents/SG/New data Ag_Fe2O3/Big circle")
OUTPUT = ROOT / "outputs" / "replicate_analysis_corrected_roi_first_5s"
OUTPUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260831)
BOOTSTRAP_ITERATIONS = 50_000

METRICS = {
    "radial_coherent": (
        "radial_coherent_response_um_s",
        "Radial coherent response",
    ),
    "total_coherent": (
        "total_coherent_response_um_s",
        "Total coherent response",
    ),
    "rms_radial": ("rms_radial_velocity_um_s", "RMS radial velocity"),
    "rms_total": ("rms_total_velocity_um_s", "RMS total velocity"),
    "Cr": ("Cr", "Radial motion fraction"),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def group_name(video: str) -> str:
    return Path(video).stem.split("_")[0].upper()


def finite_values(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def bootstrap_mean(values, rng=RNG, iterations=BOOTSTRAP_ITERATIONS) -> np.ndarray:
    values = finite_values(values)
    if len(values) == 0:
        return np.array([])
    return rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)


def bootstrap_ci(values) -> tuple[float, float]:
    return (
        tuple(np.percentile(values, [2.5, 97.5]))
        if len(values)
        else (np.nan, np.nan)
    )


def exact_permutation_p(a, b) -> float:
    a, b = finite_values(a), finite_values(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    combined = np.concatenate([a, b])
    observed = abs(np.mean(a) - np.mean(b))
    exceed = total = 0

    for selected in itertools.combinations(range(len(combined)), len(a)):
        mask = np.zeros(len(combined), dtype=bool)
        mask[list(selected)] = True
        diff = abs(np.mean(combined[mask]) - np.mean(combined[~mask]))
        exceed += diff >= observed - 1e-14
        total += 1

    return exceed / total


def cliffs_delta(a, b) -> float:
    a, b = finite_values(a), finite_values(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    d = np.subtract.outer(a, b)
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def holm_adjust(pvalues) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return adjusted

    indices = np.where(finite)[0]
    values = pvalues[finite]
    order = np.argsort(values)
    sorted_values = values[order]
    m = len(sorted_values)

    running = 0.0
    corrected = np.empty(m)
    for rank, value in enumerate(sorted_values):
        running = max(running, (m - rank) * value)
        corrected[rank] = min(running, 1.0)

    restored = np.empty(m)
    restored[order] = corrected
    adjusted[indices] = restored
    return adjusted


def read_summary() -> list[dict]:
    summary_file = OUTPUT / "summary.json"
    if not OUTPUT.exists():
        raise FileNotFoundError(
            f"\nThe analysis directory does not exist:\n\n{OUTPUT}\n\n"
            "Check the root directory and make sure the optical-flow analysis has already been run."
        )
    if not summary_file.exists():
        raise FileNotFoundError(
            f"\nCould not find summary.json:\n\n{summary_file}\n\n"
            "Run the optical-flow analysis first."
        )

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not summary:
        raise RuntimeError("summary.json contains no successfully analyzed videos.")
    return summary


def analysis_window(summary) -> float | None:
    durations = set()
    for item in summary:
        try:
            if "duration_s" in item:
                durations.add(round(float(item["duration_s"]), 6))
        except (TypeError, ValueError):
            pass
    return durations.pop() if len(durations) == 1 else None


# ---------------------------------------------------------------------
# Extract one independent row per video
# ---------------------------------------------------------------------
def extract_metrics(summary) -> pd.DataFrame:
    rows = []

    for item in summary:
        video = item["video"]
        condition = group_name(video)
        csv_file = OUTPUT / Path(video).stem / "collective_motion.csv"

        if not csv_file.exists():
            print(f"WARNING: missing collective_motion.csv for {video}")
            print(f"         Expected:\n         {csv_file}")
            continue

        print(f"Reading: {video}")
        data = pd.read_csv(csv_file)
        if data.empty:
            print("  WARNING: CSV is empty.")
            continue

        radial_coherent = float(item.get("coherent_radial_response_rms", np.nan))
        total_coherent = float(item.get("total_velocity_variation_rms", np.nan))
        rms_radial = float(item.get("overall_rms_radial_velocity", np.nan))
        rms_total = float(item.get("overall_rms_total_velocity", np.nan))
        Cr = rms_radial / rms_total if np.isfinite(rms_total) and rms_total > 0 else np.nan

        rows.append({
            "video": video,
            "condition": condition,
            "radial_coherent_response_um_s": radial_coherent,
            "rms_radial_velocity_um_s": rms_radial,
            "total_coherent_response_um_s": total_coherent,
            "rms_total_velocity_um_s": rms_total,
            "Cr": Cr,
            "frequency_detected": bool(item.get("radial_flow_frequency_detected", False)),
            "frequency_hz": item.get("radial_flow_frequency_hz", np.nan),
        })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(
            "\nNo videos could be extracted from the collective_motion.csv files.\n\n"
            "Check that the optical-flow analysis produced:\n"
            "  output/<video_name>/collective_motion.csv"
        )

    required = [
        "radial_coherent_response_um_s",
        "total_coherent_response_um_s",
        "rms_radial_velocity_um_s",
        "rms_total_velocity_um_s",
        "Cr",
    ]
    missing = [c for c in required if c not in metrics.columns]
    if missing:
        raise RuntimeError(
            "\nMissing required quantities:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + "\n\nCheck summary.json produced by Code 1."
        )

    return metrics


# ---------------------------------------------------------------------
# Descriptive statistics and bootstrap distributions
# ---------------------------------------------------------------------
def descriptive_statistics(metrics):
    boots = {key: {} for key in METRICS}
    rows = []

    for condition in ORDER:
        group = metrics[metrics.condition == condition]
        values = {
            key: finite_values(group[column])
            for key, (column, _) in METRICS.items()
        }

        for key, vals in values.items():
            boots[key][condition] = bootstrap_mean(vals)

        ci = {key: bootstrap_ci(boots[key][condition]) for key in METRICS}
        row = {"condition": condition, "n_videos": len(group)}

        for key, vals in values.items():
            column, _ = METRICS[key]
            label = key if key == "Cr" else key
            prefix = {
                "radial_coherent": "radial_coherent_response",
                "total_coherent": "total_coherent_response",
                "rms_radial": "rms_radial_velocity",
                "rms_total": "rms_total_velocity",
                "Cr": "Cr",
            }[label]

            row[f"mean_{prefix}_um_s" if key != "Cr" else "mean_Cr"] = (
                np.mean(vals) if len(vals) else np.nan
            )
            row[f"sd_{prefix}_um_s" if key != "Cr" else "sd_Cr"] = (
                np.std(vals, ddof=1) if len(vals) > 1 else np.nan
            )
            row[f"bootstrap_{prefix}_ci95_low" if key != "Cr" else "bootstrap_Cr_ci95_low"] = ci[key][0]
            row[f"bootstrap_{prefix}_ci95_high" if key != "Cr" else "bootstrap_Cr_ci95_high"] = ci[key][1]

        row["videos_with_detected_frequency"] = int(group["frequency_detected"].sum())
        rows.append(row)

    return pd.DataFrame(rows), boots


# ---------------------------------------------------------------------
# Pairwise statistics
# ---------------------------------------------------------------------
def calculate_pairwise_statistics(metrics, boots, metric_key):
    metric_column, metric_label = METRICS[metric_key]
    rows = []

    for a, b in [("UV", "BLUE"), ("UV", "GREEN"), ("BLUE", "GREEN")]:
        va = finite_values(metrics.loc[metrics.condition == a, metric_column])
        vb = finite_values(metrics.loc[metrics.condition == b, metric_column])

        boot_a, boot_b = boots[metric_key][a], boots[metric_key][b]
        difference_boot = boot_a - boot_b
        ratio_boot = boot_a / np.maximum(boot_b, 1e-12)

        mean_difference = (
            np.mean(va) - np.mean(vb) if len(va) and len(vb) else np.nan
        )
        mean_ratio = (
            np.mean(va) / np.mean(vb)
            if len(va) and len(vb) and np.mean(vb) != 0
            else np.nan
        )
        difference_lo, difference_hi = bootstrap_ci(difference_boot)
        ratio_lo, ratio_hi = bootstrap_ci(ratio_boot)

        rows.append({
            "metric": metric_label,
            "comparison": f"{a} - {b}",
            "mean_difference_um_s": mean_difference,
            "difference_ci95_low": difference_lo,
            "difference_ci95_high": difference_hi,
            "mean_ratio": mean_ratio,
            "ratio_ci95_low": ratio_lo,
            "ratio_ci95_high": ratio_hi,
            "exact_permutation_p": exact_permutation_p(va, vb),
            "cliffs_delta": cliffs_delta(va, vb),
        })

    return pd.DataFrame(rows)


def calculate_kruskal_wallis(metrics, metric_key):
    metric_column, metric_label = METRICS[metric_key]
    groups = [
        finite_values(metrics.loc[metrics.condition == c, metric_column])
        for c in ORDER
    ]

    if any(len(g) == 0 for g in groups):
        raise RuntimeError(
            f"At least one wavelength group has no valid replicates for {metric_label}."
        )

    h, p = kruskal(*groups)
    return {
        "metric": metric_label,
        "test": "Kruskal-Wallis",
        "statistic": float(h),
        "p_value": float(p),
        "groups": ORDER,
        "total_videos": len(metrics),
    }


# ---------------------------------------------------------------------
# Frequency interpretation
# ---------------------------------------------------------------------
def frequency_interpretation(metrics) -> str:
    detected = metrics[metrics.frequency_detected]
    if len(detected) == 0:
        return (
            "No recording passes the within-video periodicity screen, so no oscillation "
            "frequency is reported."
        )
    if len(detected) == 1:
        row = detected.iloc[0]
        return (
            f"Only {Path(row.video).stem} passes the within-video periodicity screen "
            f"at {float(row.frequency_hz):.3f} Hz. Because this is not reproduced in "
            "the other recordings of its condition, it is not interpreted as a "
            "condition-level frequency."
        )
    return (
        f"{len(detected)} recordings pass the within-video periodicity screen. "
        "Condition-level frequency inference requires agreement among independent replicates."
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def create_metric_plot(metrics, metric_key, title, ylabel, filename, boots):
    metric_column, _ = METRICS[metric_key]
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, condition in enumerate(ORDER):
        values = finite_values(metrics.loc[metrics.condition == condition, metric_column])
        x = RNG.normal(i, 0.035, len(values))

        ax.scatter(
            x, values, s=80, color=COLORS[condition], edgecolor="white",
            linewidth=0.7, zorder=3, alpha=0.7,
        )

        boot = boots[metric_key][condition]
        if len(boot):
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mean = np.mean(values)
            ax.errorbar(
                i, mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                color="black", capsize=10, zorder=4, markersize=14,
            )

    ax.set(
        xlabel="Condition",
        ylabel=ylabel,
        title=title,
        xticks=range(3),
        xticklabels=ORDER,
    )
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.title.set_size(18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
def build_report(desc, pairs, omnibus, frequency_statement, window):
    window_label = f"first {window:g} seconds" if window is not None else "configured analysis window"
    lines = [
        f"# Replicate-level statistical analysis — {window_label}",
        "",
        "Each video is treated as one independent experimental replicate. Frames and pixels "
        "are not counted as independent samples.",
        "",
        "## Definitions",
        "",
        "### Radial coherent response", "", "SD[Gσ(mean(vr(t)))]", "",
        "This is the coherent radial response imported directly from summary.json produced by "
        "the optical-flow analysis.",
        "",
        "### Total coherent response", "", "SD[Gσ(RMS(vt(t)))]", "",
        "This is the temporal variation of the frame-level RMS total velocity, imported directly "
        "from summary.json.",
        "",
        "### RMS radial velocity", "", "RMS(vr)", "",
        "### RMS total velocity", "", "RMS(vt)", "",
        "### Radial motion fraction", "", "Cr = RMS(vr) / RMS(vt)", "",
        "Cr is interpreted as the fraction of total RMS motion represented by radial motion, "
        "rather than as a formal statistical coherence coefficient.",
        "",
        "## Descriptive statistics", "",
        "| Condition | Videos | Radial coherent response (mean ± SD) | "
        "Total coherent response (mean ± SD) | RMS radial velocity (mean ± SD) | "
        "RMS total velocity (mean ± SD) | Cr (mean ± SD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in desc.itertuples():
        lines.append(
            f"| {r.condition} | {r.n_videos} | "
            f"{r.mean_radial_coherent_response_um_s:.4f} ± {r.sd_radial_coherent_response_um_s:.4f} | "
            f"{r.mean_total_coherent_response_um_s:.4f} ± {r.sd_total_coherent_response_um_s:.4f} | "
            f"{r.mean_rms_radial_velocity_um_s:.4f} ± {r.sd_rms_radial_velocity_um_s:.4f} | "
            f"{r.mean_rms_total_velocity_um_s:.4f} ± {r.sd_rms_total_velocity_um_s:.4f} | "
            f"{r.mean_Cr:.3f} ± {r.sd_Cr:.3f} |"
        )

    lines += [
        "",
        "95% bootstrap confidence intervals are reported in replicate_descriptive_statistics.csv.",
        "",
        "## Overall statistical tests", "",
        f"Radial coherent response — Kruskal-Wallis: "
        f"H = {omnibus['radial_coherent_response']['statistic']:.3f}, "
        f"p = {omnibus['radial_coherent_response']['p_value']:.4f}.",
        "",
        f"Total coherent response — Kruskal-Wallis: "
        f"H = {omnibus['total_coherent_response']['statistic']:.3f}, "
        f"p = {omnibus['total_coherent_response']['p_value']:.4f}.",
        "",
        "## Pairwise comparisons", "",
        "Pairwise statistics are calculated separately and in parallel for radial coherent "
        "response and total coherent response.",
        "",
        "| Metric | Comparison | Mean difference (µm/s) | 95% CI | Mean ratio | "
        "95% CI | Permutation p | Holm-adjusted p | Cliff's delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in pairs.itertuples():
        lines.append(
            f"| {r.metric} | {r.comparison} | {r.mean_difference_um_s:.4f} | "
            f"{r.difference_ci95_low:.4f}–{r.difference_ci95_high:.4f} | "
            f"{r.mean_ratio:.2f} | {r.ratio_ci95_low:.2f}–{r.ratio_ci95_high:.2f} | "
            f"{r.exact_permutation_p:.4f} | {r.holm_adjusted_p:.4f} | {r.cliffs_delta:.2f} |"
        )

    lines += [
        "",
        "## Frequency interpretation", "",
        frequency_statement,
        "",
        "A tallest spectral bin alone is not treated as evidence for a condition-level "
        "oscillation frequency.",
        "",
        "## Interpretation limits", "",
        "The statistical unit is the independent video. The number of frames or pixels does "
        "not increase the number of independent biological or experimental replicates.",
        "",
        "The radial coherent response is SD[Gσ(mean(vr(t)))].",
        "",
        "The total coherent response is SD[Gσ(RMS(vt(t)))].",
        "",
        "RMS radial velocity and RMS total velocity quantify the magnitude of radial and total "
        "particle motion, respectively.",
        "",
        "Cr is defined as RMS radial velocity divided by RMS total velocity. It is interpreted "
        "as the fraction of total motion represented by radial motion, rather than as a formal "
        "statistical coherence coefficient.",
        "",
        "The total-speed quantity is non-negative. Its temporal spectrum can therefore contain "
        "harmonics of the underlying signed radial motion.",
        "",
        "Absolute velocity accuracy still requires validation against manual particle tracks or "
        "known synthetic displacements, a no-light control, and uncertainty in the calibration.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("INDEPENDENT-VIDEO STATISTICAL ANALYSIS")
    print("=" * 70 + "\n")

    summary = read_summary()
    window = analysis_window(summary)
    metrics = extract_metrics(summary)

    for condition in ORDER:
        n = int((metrics.condition == condition).sum())
        print(f"{condition}: {n} independent videos")

    metrics.to_csv(OUTPUT / "replicate_level_metrics.csv", index=False)

    desc, boots = descriptive_statistics(metrics)
    desc.to_csv(OUTPUT / "replicate_descriptive_statistics.csv", index=False)

    pairs = pd.concat(
        [
            calculate_pairwise_statistics(metrics, boots, "radial_coherent"),
            calculate_pairwise_statistics(metrics, boots, "total_coherent"),
        ],
        ignore_index=True,
    )

    for metric in ("radial_coherent", "total_coherent"):
        mask = pairs.metric == METRICS[metric][1]
        pairs.loc[mask, "holm_adjusted_p"] = holm_adjust(pairs.loc[mask, "exact_permutation_p"])

    pairs.to_csv(OUTPUT / "replicate_pairwise_statistics.csv", index=False)

    omnibus = {
        "radial_coherent_response": calculate_kruskal_wallis(metrics, "radial_coherent"),
        "total_coherent_response": calculate_kruskal_wallis(metrics, "total_coherent"),
    }
    (OUTPUT / "replicate_omnibus_tests.json").write_text(
        json.dumps(omnibus, indent=2), encoding="utf-8"
    )

    plots = [
        ("radial_coherent", "Radial coherent response",
         r"SD[Gσ(mean(vr(t)))] (µm/s)", "replicate_statistics_radial_coherent_response.png"),
        ("total_coherent", "Total coherent response",
         r"SD[Gσ(RMS(vt(t)))] (µm/s)", "replicate_statistics_total_coherent_response.png"),
        ("rms_radial", "RMS radial velocity",
         "RMS radial velocity (µm/s)", "replicate_statistics_rms_radial_velocity.png"),
        ("rms_total", "RMS total velocity",
         "RMS total velocity (µm/s)", "replicate_statistics_rms_total_velocity.png"),
        ("Cr", "Radial motion fraction",
         "Cr = RMS radial / RMS total", "replicate_statistics_Cr.png"),
    ]
    for key, title, ylabel, filename in plots:
        create_metric_plot(metrics, key, title, ylabel, filename, boots)

    frequency_statement = frequency_interpretation(metrics)
    report = build_report(desc, pairs, omnibus, frequency_statement, window)
    (OUTPUT / "replicate_statistics_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70 + "\n")
    print(desc.to_string(index=False))

    print("\n" + "=" * 70)
    print("PAIRWISE STATISTICS")
    print("=" * 70 + "\n")
    print(pairs.to_string(index=False))

    print("\n" + "=" * 70)
    print("KRUSKAL-WALLIS TESTS")
    print("=" * 70 + "\n")
    print(json.dumps(omnibus, indent=2))

    print("\n" + "=" * 70)
    print("OUTPUT DIRECTORY")
    print("=" * 70 + "\n")
    print(OUTPUT)
    print("\nStatistical analysis completed successfully.")


if __name__ == "__main__":
    main()
