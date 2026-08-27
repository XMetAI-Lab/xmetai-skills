"""Metric curve and summary chart rendering."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .common import lead_label


def plot_tcc_curves(report: dict, output_dir: Path, freq: int = 24) -> None:
    """Plot TCC per-lead curve and weekly average bar chart."""
    if "tcc_per_level" not in report:
        return

    levels = report["levels"]
    n_leads = report["n_leads"]
    leads = list(range(1, n_leads + 1))
    lead_labels = [lead_label(l - 1, freq) for l in leads]
    tcc_data = report["tcc_per_level"]

    # --- TCC per-lead curves ---
    cols = 3
    nrows = int(np.ceil(len(levels) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 3 * nrows), squeeze=False)
    for idx, level in enumerate(levels):
        ax = axes[idx // cols][idx % cols]
        ax.plot(leads, tcc_data[level], marker="o", ms=3, color="tab:blue")
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="skill threshold (0.5)")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.set_title(level, fontsize=10)
        ax.set_xlabel("lead time")
        ax.set_ylabel("TCC")
        ax.set_ylim(-0.2, 1.0)
        ax.set_xticks(leads)
        ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(levels), nrows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("TCC per channel (across init dates)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "tcc_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out}")

    # --- TCC weekly average bar chart ---
    if "tcc_weekly" not in report:
        return

    tcc_weekly = report["tcc_weekly"]
    week_labels = report["week_labels"]
    n_weeks = len(week_labels)
    x = np.arange(n_weeks)
    width = 0.8 / max(len(levels), 1)

    fig, ax = plt.subplots(figsize=(max(6, n_weeks * 1.5), 5))
    for idx, level in enumerate(levels):
        vals = tcc_weekly[level]
        offset = (idx - len(levels) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=level, alpha=0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="skill threshold (0.5)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Forecast week")
    ax.set_ylabel("TCC")
    ax.set_ylim(-0.2, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(week_labels)
    ax.set_title("TCC Weekly Average (across init dates)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=min(4, len(levels)))
    fig.tight_layout()
    out = output_dir / "tcc_weekly.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out}")


def plot_ps_curves(report: dict, output_dir: Path, freq: int = 24) -> None:
    """Plot PS per-lead curves and overall bar chart."""
    if "ps_per_level" not in report:
        return

    levels = report["levels"]
    n_leads = report["n_leads"]
    leads = list(range(1, n_leads + 1))
    lead_labels = [lead_label(l - 1, freq) for l in leads]
    ps_data = report["ps_per_level"]

    # PS per-lead curves
    cols = 3
    nrows = int(np.ceil(len(levels) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 3 * nrows), squeeze=False)
    for idx, level in enumerate(levels):
        ax = axes[idx // cols][idx % cols]
        ax.plot(leads, ps_data[level], marker="o", ms=3, color="tab:green")
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="perfect (100)")
        ax.axhline(y=60, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="reference (60)")
        ax.set_title(level, fontsize=10)
        ax.set_xlabel("lead time")
        ax.set_ylabel("PS")
        ax.set_ylim(0, 105)
        ax.set_xticks(leads)
        ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(levels), nrows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("PS per channel (Climate Business Score)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "ps_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out}")

    # PS overall bar chart
    if "ps_overall" in report:
        ps_overall = report["ps_overall"]
        fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.2), 5))
        x = np.arange(len(levels))
        vals = [ps_overall[lev] for lev in levels]
        bars = ax.bar(x, vals, color="tab:green", alpha=0.85)
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="perfect (100)")
        ax.axhline(y=60, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="reference (60)")
        ax.set_xlabel("Channel")
        ax.set_ylabel("PS")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_title("PS Overall (averaged over all leads)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out = output_dir / "ps_overall.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")


def plot_ips_curves(report: dict, output_dir: Path, freq: int = 24) -> None:
    """Plot IPS per-pentad curves and component breakdown."""
    if "ips_per_level" not in report:
        return

    levels = report["levels"]
    ips_data = report["ips_per_level"]
    pentad_labels = report.get("pentad_labels", [])
    if not pentad_labels:
        return

    n_pentads = len(pentad_labels)
    x = np.arange(n_pentads)

    # IPS per-pentad curves
    cols = 3
    nrows = int(np.ceil(len(levels) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 3 * nrows), squeeze=False)
    for idx, level in enumerate(levels):
        ax = axes[idx // cols][idx % cols]
        ax.plot(x, ips_data[level]["ips"], marker="o", ms=3, color="tab:purple", label="IPS")
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="no skill (50)")
        ax.axhline(y=100, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label="perfect (100)")
        ax.set_title(level, fontsize=10)
        ax.set_xlabel("Pentad")
        ax.set_ylabel("IPS")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(pentad_labels, fontsize=8, rotation=45)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(levels), nrows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("IPS per channel (Integrated Pattern Score)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "ips_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out}")

    # IPS component breakdown (PCC and AS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (metric, color, title) in enumerate([
        ("pcc", "tab:blue", "PCC (Pearson Correlation)"),
        ("as", "tab:orange", "AS (Anomaly Sign Agreement)")
    ]):
        ax = axes[idx]
        for level in levels:
            ax.plot(x, ips_data[level][metric], marker="o", ms=3, label=level)
        ax.axhline(y=0 if metric == "pcc" else 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Pentad")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(pentad_labels, fontsize=8, rotation=45)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=min(4, len(levels)))
    fig.suptitle("IPS Components", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "ips_components.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out}")

    # IPS overall bar chart
    if "ips_overall" in report:
        ips_overall = report["ips_overall"]
        fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.2), 5))
        vals = [ips_overall[lev] for lev in levels]
        bars = ax.bar(np.arange(len(levels)), vals, color="tab:purple", alpha=0.85)
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="no skill (50)")
        ax.axhline(y=100, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label="perfect (100)")
        ax.set_xlabel("Channel")
        ax.set_ylabel("IPS")
        ax.set_ylim(0, 105)
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_title("IPS Overall (weighted by pentad)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out = output_dir / "ips_overall.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")


def plot_metric_curves(report: dict, metrics: list[str], thresholds: list[float], output_dir: Path, freq: int = 24) -> None:
    leads = list(range(1, report["n_leads"] + 1))
    lead_labels = [lead_label(l - 1, freq) for l in leads]

    if "rmse" in metrics:
        levels = report["levels"]
        cols = 3
        rows = int(np.ceil(len(levels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
        for idx, level in enumerate(levels):
            ax = axes[idx // cols][idx % cols]
            ax.plot(leads, report["rmse_per_level"][level], marker="o", ms=3)
            ax.set_title(level, fontsize=10)
            ax.set_xlabel("lead time")
            ax.set_xticks(leads)
            ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
            ax.grid(alpha=0.3)
        for idx in range(len(levels), rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.suptitle("RMSE per channel", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = output_dir / "rmse_curves.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")

    if "ts" in metrics and report.get("threshold_metrics"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t in thresholds:
            key = str(t)
            ax.plot(leads, report["threshold_metrics"][key]["ts"], marker="o", ms=3, label=f">= {t}")
        ax.set_xlabel("lead time")
        ax.set_ylabel("TS")
        ax.set_title(f"Threat Score ({report['channel']})")
        ax.set_xticks(leads)
        ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / "ts_curves.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        tm = report["threshold_metrics"]
        for metric, style in (("ts", "-o"), ("pod", "-s"), ("far", "-^")):
            for lead in range(report["n_leads"]):
                values = [tm[str(t)][metric][lead] for t in thresholds]
                label = f"{metric} lead {lead + 1}" if lead == 0 else None
                ax.plot(thresholds, values, style, ms=3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("threshold (physical unit)")
        ax.set_ylabel("score")
        ax.set_title(f"Threshold metrics ({report['channel']})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = output_dir / "threshold_metrics.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")



    if "tcc" in metrics:
        plot_tcc_curves(report, output_dir, freq)

    if "ps" in metrics:
        plot_ps_curves(report, output_dir, freq)

    if "ips" in metrics:
        plot_ips_curves(report, output_dir, freq)
