import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


BASE = Path(r"G:\529\result")
OUT = BASE / "uncertainty_figures"

SOURCES = {
    ("MIL", "origin"): BASE / "MIL_test_analysis" / "origin_test.csv",
    ("MIL", "EDL"): BASE / "MIL_test_analysis" / "EDL_test.csv",
    ("MIL", "EDL-prototype"): BASE / "MIL_test_analysis" / "EDL-prototype_test.csv",
    ("GLAM", "origin"): BASE / "GLAM_test_analysis" / "origin_test.csv",
    ("GLAM", "EDL"): BASE / "GLAM_test_analysis" / "EDL_test.csv",
    ("GLAM", "EDL-prototype"): BASE / "GLAM_test_analysis" / "EDL-prototype_test.csv",
}

MODEL_ORDER = ["origin", "EDL", "EDL-prototype"]
FAMILY_ORDER = ["MIL", "GLAM"]


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)


def quantile_bins(values, bins=5):
    ranks = pd.Series(values).rank(method="first")
    q = min(bins, int(pd.Series(values).nunique()), len(values))
    if q < 2:
        return np.ones(len(values), dtype=int)
    return pd.qcut(ranks, q=q, labels=False) + 1


def safe_metric(y_true, scores, metric):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return math.nan
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, scores))
    if metric == "auprc":
        return float(average_precision_score(y_true, scores))
    raise ValueError(metric)


def load_frames():
    frames = []
    for (family, model), path in SOURCES.items():
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == "test"].copy()
        label_col = "label" if "label" in df.columns else "cancer"
        if model == "origin":
            score_col = "pred_score"
            uncertainty = binary_entropy(df[score_col].to_numpy(dtype=float))
            measure_name = "entropy(pred_score)"
        else:
            score_col = "prob_1"
            uncertainty = df["uncertainty"].to_numpy(dtype=float)
            measure_name = "Dirichlet uncertainty"
        df["family"] = family
        df["model"] = model
        df["measure_name"] = measure_name
        df["y_true"] = df[label_col].astype(int)
        df["score"] = df[score_col].astype(float)
        df["uncertainty_or_proxy"] = uncertainty
        df["pred"] = (df["score"] >= 0.5).astype(int)
        df["wrong"] = (df["pred"] != df["y_true"]).astype(int)
        df["label_name"] = np.where(df["y_true"] == 1, "positive", "negative")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_label(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        wrong = g["wrong"].to_numpy(dtype=int)
        unc = g["uncertainty_or_proxy"].to_numpy(dtype=float)
        rows.append(
            {
                "family": family,
                "model": model,
                "measure_name": g["measure_name"].iloc[0],
                "label_name": label_name,
                "n": len(g),
                "wrong_n": int(wrong.sum()),
                "error_type": "FN_rate" if label_name == "positive" else "FP_rate",
                "error_rate": float(wrong.mean()),
                "unc_or_proxy_mean": float(g["uncertainty_or_proxy"].mean()),
                "unc_or_proxy_median": float(g["uncertainty_or_proxy"].median()),
                "score_mean": float(g["score"].mean()),
                "score_uncertainty_corr": float(g["score"].corr(g["uncertainty_or_proxy"])),
                "failure_auroc_within_label": safe_metric(wrong, unc, "roc_auc"),
                "failure_auprc_within_label": safe_metric(wrong, unc, "auprc"),
            }
        )
    return pd.DataFrame(rows)


def summarize_high_low(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        g = g.sort_values("uncertainty_or_proxy").reset_index(drop=True)
        k = max(1, int(round(len(g) * 0.2)))
        low = g.head(k)
        high = g.tail(k)
        low_err = float(low["wrong"].mean())
        high_err = float(high["wrong"].mean())
        rows.append(
            {
                "family": family,
                "model": model,
                "measure_name": g["measure_name"].iloc[0],
                "label_name": label_name,
                "n": len(g),
                "low20_n": len(low),
                "high20_n": len(high),
                "low20_error_rate": low_err,
                "high20_error_rate": high_err,
                "high_minus_low_error_rate": high_err - low_err,
                "high_over_low_error_rate": high_err / low_err if low_err > 0 else math.inf,
            }
        )
    return pd.DataFrame(rows)


def summarize_quintile(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        g = g.copy()
        g["unc_bin"] = quantile_bins(g["uncertainty_or_proxy"], bins=5)
        for bin_id, b in g.groupby("unc_bin", sort=True):
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "measure_name": g["measure_name"].iloc[0],
                    "label_name": label_name,
                    "uncertainty_quintile": int(bin_id),
                    "n": len(b),
                    "wrong_n": int(b["wrong"].sum()),
                    "error_rate": float(b["wrong"].mean()),
                    "unc_or_proxy_mean": float(b["uncertainty_or_proxy"].mean()),
                    "score_mean": float(b["score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_quintiles(quint, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True, sharey=True)
    x_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    for row, family in enumerate(FAMILY_ORDER):
        for col, model in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            g = quint[(quint["family"] == family) & (quint["model"] == model)]
            for label_name, color, marker, display in [
                ("negative", "#4C78A8", "o", "Specificity among negatives"),
                ("positive", "#F58518", "s", "Sensitivity among positives"),
            ]:
                sub = g[g["label_name"] == label_name].sort_values("uncertainty_quintile")
                performance = 1.0 - sub["error_rate"]
                ax.plot(
                    sub["uncertainty_quintile"],
                    performance,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=5,
                    label=display,
                )
            title = f"{family} {model}"
            subtitle = "origin entropy" if model == "origin" else "Dirichlet uncertainty"
            ax.set_title(f"{title}\n({subtitle})", fontsize=11, weight="bold")
            ax.set_xticks([1, 2, 3, 4, 5], x_labels, rotation=20)
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("Within-label uncertainty/proxy percentile")
            if col == 0:
                ax.set_ylabel("Performance")
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Origin vs EDL: Sensitivity/Specificity by Uncertainty Percentile", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_high_low(high_low, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    colors = {"negative": "#4C78A8", "positive": "#F58518"}
    width = 0.36
    x = np.arange(len(MODEL_ORDER))
    for ax, family in zip(axes, FAMILY_ORDER):
        g = high_low[high_low["family"] == family]
        for offset, label_name in [(-width / 2, "negative"), (width / 2, "positive")]:
            sub = g[g["label_name"] == label_name].set_index("model").reindex(MODEL_ORDER)
            label = "negative: high20-low20 FP" if label_name == "negative" else "positive: high20-low20 FN"
            vals = sub["high_minus_low_error_rate"].to_numpy(dtype=float)
            bars = ax.bar(x + offset, vals, width=width, color=colors[label_name], label=label)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.018 if val >= 0 else -0.045),
                    f"{val:+.3f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                )
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(f"{family}", fontsize=12, weight="bold")
        ax.set_xticks(x, MODEL_ORDER)
        ax.set_ylabel("High 20% minus low 20% error rate")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Which Label Is More Affected by High Uncertainty?", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_separate(quint, label_name, metric_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True, sharey=True)
    x_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    colors = {
        "origin": "#4C78A8",
        "EDL": "#F58518",
        "EDL-prototype": "#54A24B",
    }
    markers = {
        "origin": "o",
        "EDL": "s",
        "EDL-prototype": "^",
    }
    for ax, family in zip(axes, FAMILY_ORDER):
        for model in MODEL_ORDER:
            sub = quint[
                (quint["family"] == family)
                & (quint["model"] == model)
                & (quint["label_name"] == label_name)
            ].sort_values("uncertainty_quintile")
            performance = 1.0 - sub["error_rate"]
            ax.plot(
                sub["uncertainty_quintile"],
                performance,
                color=colors[model],
                marker=markers[model],
                linewidth=2.2,
                markersize=5.5,
                label=model,
            )
        ax.set_title(f"{family}", fontsize=13, weight="bold")
        ax.set_xticks([1, 2, 3, 4, 5], x_labels, rotation=20)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.set_xlabel("Within-label uncertainty/proxy percentile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel(metric_name)
    fig.suptitle(f"{metric_name} by Uncertainty Percentile", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary(label_summary, high_low, out_path):
    lines = [
        "Origin vs EDL label-specific uncertainty comparison",
        "===================================================",
        "",
        "Origin has no Dirichlet uncertainty, so entropy(pred_score) is used as a proxy.",
        "EDL and EDL-prototype use their exported Dirichlet uncertainty.",
        "All comparisons are based on within-label uncertainty/proxy ranking.",
        "",
    ]
    for family in FAMILY_ORDER:
        lines.append(f"{family}:")
        for model in MODEL_ORDER:
            sub = label_summary[(label_summary["family"] == family) & (label_summary["model"] == model)]
            hl = high_low[(high_low["family"] == family) & (high_low["model"] == model)]
            neg = sub[sub["label_name"] == "negative"].iloc[0]
            pos = sub[sub["label_name"] == "positive"].iloc[0]
            neg_hl = hl[hl["label_name"] == "negative"].iloc[0]
            pos_hl = hl[hl["label_name"] == "positive"].iloc[0]
            lines.extend(
                [
                    f"- {model} ({neg.measure_name}):",
                    f"  negative FP rate={neg.error_rate:.4f}, AUROC={neg.failure_auroc_within_label:.4f}, high20-low20 FP={neg_hl.high_minus_low_error_rate:.4f}.",
                    f"  positive FN rate={pos.error_rate:.4f}, AUROC={pos.failure_auroc_within_label:.4f}, high20-low20 FN={pos_hl.high_minus_low_error_rate:.4f}.",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "Main read:",
            "- MIL origin entropy mainly ranks positive false negatives: high-entropy positives have much higher FN risk.",
            "- MIL EDL shifts the useful uncertainty signal toward negative false positives: high-uncertainty negatives have much higher FP risk.",
            "- MIL EDL-prototype has the best classification performance, but its Dirichlet uncertainty barely ranks errors within labels.",
            "- For positive samples, none of the uncertainty/proxy measures reliably identify false negatives; n=66 makes this especially noisy.",
            "- The exception is MIL origin entropy, which ranks positive false negatives well, but this is a score-entropy proxy rather than EDL uncertainty.",
            "- GLAM EDL-prototype is degenerate at threshold 0.5: all negatives are TN and all positives are FN, so uncertainty cannot separate outcomes.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frames()
    label_summary = summarize_label(df)
    high_low = summarize_high_low(df)
    quint = summarize_quintile(df)

    label_summary.to_csv(OUT / "origin_vs_edl_label_specific_summary.csv", index=False)
    high_low.to_csv(OUT / "origin_vs_edl_label_specific_high_low.csv", index=False)
    quint.to_csv(OUT / "origin_vs_edl_label_specific_quintiles.csv", index=False)

    plot_quintiles(quint, OUT / "11_origin_vs_edl_label_specific_error_by_uncertainty_quintile.png")
    plot_high_low(high_low, OUT / "12_origin_vs_edl_high_low_gap_by_label.png")
    plot_metric_separate(
        quint,
        "negative",
        "Specificity",
        OUT / "13_specificity_by_uncertainty_percentile.png",
    )
    plot_metric_separate(
        quint,
        "positive",
        "Sensitivity",
        OUT / "14_sensitivity_by_uncertainty_percentile.png",
    )
    write_summary(label_summary, high_low, OUT / "origin_vs_edl_label_specific_summary.txt")
    print("Wrote origin-vs-EDL label-specific comparison to", OUT)


if __name__ == "__main__":
    main()
