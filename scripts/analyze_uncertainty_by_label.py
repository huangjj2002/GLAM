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
    ("MIL", "EDL"): BASE / "MIL_test_analysis" / "EDL_test.csv",
    ("MIL", "EDL-prototype"): BASE / "MIL_test_analysis" / "EDL-prototype_test.csv",
    ("GLAM", "EDL"): BASE / "GLAM_test_analysis" / "EDL_test.csv",
    ("GLAM", "EDL-prototype"): BASE / "GLAM_test_analysis" / "EDL-prototype_test.csv",
}

COLORS = {
    "negative": "#4C78A8",
    "positive": "#F58518",
}


def quantile_bins(values, bins=5):
    ranks = pd.Series(values).rank(method="first")
    n_unique = int(pd.Series(values).nunique())
    q = min(bins, n_unique, len(values))
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
        df["family"] = family
        df["model"] = model
        df["y_true"] = df[label_col].astype(int)
        df["score"] = df["prob_1"].astype(float)
        df["pred"] = (df["score"] >= 0.5).astype(int)
        df["wrong"] = (df["pred"] != df["y_true"]).astype(int)
        df["label_name"] = np.where(df["y_true"] == 1, "positive", "negative")
        df["outcome"] = np.select(
            [
                (df["y_true"] == 0) & (df["pred"] == 0),
                (df["y_true"] == 0) & (df["pred"] == 1),
                (df["y_true"] == 1) & (df["pred"] == 0),
                (df["y_true"] == 1) & (df["pred"] == 1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="NA",
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_by_label(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        wrong = g["wrong"].to_numpy()
        unc = g["uncertainty"].to_numpy(dtype=float)
        label = int(g["y_true"].iloc[0])
        rows.append(
            {
                "family": family,
                "model": model,
                "label": label,
                "label_name": label_name,
                "n": len(g),
                "wrong_n": int(wrong.sum()),
                "error_type": "FN_rate" if label == 1 else "FP_rate",
                "error_rate": float(wrong.mean()),
                "unc_mean": float(np.mean(unc)),
                "unc_std": float(np.std(unc, ddof=1)) if len(unc) > 1 else math.nan,
                "unc_median": float(np.median(unc)),
                "unc_q10": float(np.quantile(unc, 0.10)),
                "unc_q25": float(np.quantile(unc, 0.25)),
                "unc_q75": float(np.quantile(unc, 0.75)),
                "unc_q90": float(np.quantile(unc, 0.90)),
                "score_mean": float(g["score"].mean()),
                "score_median": float(g["score"].median()),
                "score_uncertainty_corr": float(g["score"].corr(g["uncertainty"])),
                "failure_auroc_within_label": safe_metric(wrong, unc, "roc_auc"),
                "failure_auprc_within_label": safe_metric(wrong, unc, "auprc"),
            }
        )
    return pd.DataFrame(rows)


def summarize_high_low(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        g = g.sort_values("uncertainty").reset_index(drop=True)
        k = max(1, int(round(len(g) * 0.2)))
        low = g.head(k)
        high = g.tail(k)
        low_err = float(low["wrong"].mean())
        high_err = float(high["wrong"].mean())
        rows.append(
            {
                "family": family,
                "model": model,
                "label_name": label_name,
                "n": len(g),
                "low20_n": len(low),
                "high20_n": len(high),
                "low20_error_rate": low_err,
                "high20_error_rate": high_err,
                "high_minus_low_error_rate": high_err - low_err,
                "high_over_low_error_rate": high_err / low_err if low_err > 0 else math.inf,
                "low20_unc_mean": float(low["uncertainty"].mean()),
                "high20_unc_mean": float(high["uncertainty"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_quintiles(df):
    rows = []
    for (family, model, label_name), g in df.groupby(["family", "model", "label_name"], sort=False):
        g = g.copy()
        g["unc_bin"] = quantile_bins(g["uncertainty"], bins=5)
        for bin_id, b in g.groupby("unc_bin", sort=True):
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "label_name": label_name,
                    "uncertainty_quintile": int(bin_id),
                    "n": len(b),
                    "wrong_n": int(b["wrong"].sum()),
                    "error_rate": float(b["wrong"].mean()),
                    "unc_mean": float(b["uncertainty"].mean()),
                    "score_mean": float(b["score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_outcomes(df):
    rows = []
    for (family, model, outcome), g in df.groupby(["family", "model", "outcome"], sort=False):
        rows.append(
            {
                "family": family,
                "model": model,
                "outcome": outcome,
                "n": len(g),
                "unc_mean": float(g["uncertainty"].mean()),
                "unc_median": float(g["uncertainty"].median()),
                "score_mean": float(g["score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_uncertainty_by_label(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for ax, ((family, model), g) in zip(axes.flat, df.groupby(["family", "model"], sort=False)):
        neg = g.loc[g["y_true"] == 0, "uncertainty"].to_numpy()
        pos = g.loc[g["y_true"] == 1, "uncertainty"].to_numpy()
        parts = ax.violinplot([neg, pos], positions=[0, 1], showmeans=False, showmedians=True)
        for body, color in zip(parts["bodies"], [COLORS["negative"], COLORS["positive"]]):
            body.set_facecolor(color)
            body.set_edgecolor("#222222")
            body.set_alpha(0.35)
        for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.1)
        ax.boxplot(
            [neg, pos],
            positions=[0, 1],
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.0},
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
        )
        rng = np.random.default_rng(20260531)
        pos_sample = pos if len(pos) <= 120 else rng.choice(pos, 120, replace=False)
        ax.scatter(
            np.full(len(pos_sample), 1.0) + rng.normal(0, 0.035, len(pos_sample)),
            pos_sample,
            s=12,
            color=COLORS["positive"],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
            zorder=4,
        )
        ax.set_title(f"{family} {model}", fontsize=12, weight="bold")
        ax.set_xticks([0, 1], ["Negative (n=%d)" % len(neg), "Positive (n=%d)" % len(pos)])
        ax.set_ylabel("Uncertainty")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Uncertainty Distribution by True Label", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_label_specific_error(quint, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True)
    for ax, ((family, model), g) in zip(axes.flat, quint.groupby(["family", "model"], sort=False)):
        for label_name, color, marker in [
            ("negative", COLORS["negative"], "o"),
            ("positive", COLORS["positive"], "s"),
        ]:
            sub = g[g["label_name"] == label_name].sort_values("uncertainty_quintile")
            label = "FP rate among negatives" if label_name == "negative" else "FN rate among positives"
            ax.plot(
                sub["uncertainty_quintile"],
                sub["error_rate"],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=5,
                label=label,
            )
            for x, y, n in zip(sub["uncertainty_quintile"], sub["error_rate"], sub["n"]):
                ax.text(x, y + 0.025, f"n={n}", color=color, fontsize=7, ha="center")
        ax.set_title(f"{family} {model}", fontsize=12, weight="bold")
        ax.set_xlabel("Within-label uncertainty quintile")
        ax.set_ylabel("Error rate")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Does Higher Uncertainty Affect Negatives or Positives?", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_outcome_uncertainty(outcome, out_path):
    order = ["TN", "FP", "FN", "TP"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=False)
    for ax, ((family, model), g) in zip(axes.flat, outcome.groupby(["family", "model"], sort=False)):
        g = g.set_index("outcome").reindex(order).reset_index()
        colors = ["#72B7B2", "#E45756", "#E45756", "#72B7B2"]
        ax.bar(g["outcome"], g["unc_mean"], color=colors, edgecolor="#333333", linewidth=0.7)
        for x, row in enumerate(g.itertuples()):
            if pd.notna(row.n):
                ax.text(x, row.unc_mean, f"n={int(row.n)}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{family} {model}", fontsize=12, weight="bold")
        ax.set_ylabel("Mean uncertainty")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Mean Uncertainty by Confusion-Matrix Outcome", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_text(label_summary, high_low, out_path):
    lines = [
        "Label-specific uncertainty summary",
        "===================================",
        "",
        "Interpretation:",
        "- Negative label: uncertainty effect is measured by false-positive rate (FP rate).",
        "- Positive label: uncertainty effect is measured by false-negative rate (FN rate).",
        "- Positive samples are only n=66 per model, so positive-label trends are useful but noisy.",
        "",
        "Main takeaways:",
    ]

    for family, model in [("MIL", "EDL"), ("MIL", "EDL-prototype"), ("GLAM", "EDL"), ("GLAM", "EDL-prototype")]:
        sub = label_summary[(label_summary["family"] == family) & (label_summary["model"] == model)]
        hl = high_low[(high_low["family"] == family) & (high_low["model"] == model)]
        neg = sub[sub["label_name"] == "negative"].iloc[0]
        pos = sub[sub["label_name"] == "positive"].iloc[0]
        neg_hl = hl[hl["label_name"] == "negative"].iloc[0]
        pos_hl = hl[hl["label_name"] == "positive"].iloc[0]
        lines.extend(
            [
                f"- {family} {model}:",
                f"  negative FP rate={neg.error_rate:.4f}, within-label AUROC={neg.failure_auroc_within_label:.4f}, high20-low20 FP={neg_hl.high_minus_low_error_rate:.4f}.",
                f"  positive FN rate={pos.error_rate:.4f}, within-label AUROC={pos.failure_auroc_within_label:.4f}, high20-low20 FN={pos_hl.high_minus_low_error_rate:.4f}.",
                f"  mean uncertainty negative={neg.unc_mean:.6f}, positive={pos.unc_mean:.6f}.",
                f"  corr(prob_1, uncertainty): negative={neg.score_uncertainty_corr:.4f}, positive={pos.score_uncertainty_corr:.4f}.",
            ]
        )

    lines.extend(
        [
            "",
            "Practical read:",
            "- If the negative high20-low20 FP gap is larger, uncertainty mainly flags benign cases likely to become false positives.",
            "- If the positive high20-low20 FN gap is larger, uncertainty mainly flags cancer cases likely to be missed.",
            "- For MIL EDL, the larger negative-label gap suggests uncertainty is mostly acting on false-positive benign cases, not missed positive cases.",
            "- In this dataset, inspect MIL EDL first: it is the only model where uncertainty has a clear error-ranking signal overall.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frames()

    label_summary = summarize_by_label(df)
    high_low = summarize_high_low(df)
    quint = summarize_quintiles(df)
    outcome = summarize_outcomes(df)

    label_summary.to_csv(OUT / "label_specific_uncertainty_summary.csv", index=False)
    high_low.to_csv(OUT / "label_specific_high_low_uncertainty.csv", index=False)
    quint.to_csv(OUT / "label_specific_uncertainty_quintiles.csv", index=False)
    outcome.to_csv(OUT / "label_specific_outcome_uncertainty.csv", index=False)

    plot_uncertainty_by_label(df, OUT / "08_uncertainty_by_true_label.png")
    plot_label_specific_error(quint, OUT / "09_label_specific_error_by_uncertainty_quintile.png")
    plot_outcome_uncertainty(outcome, OUT / "10_uncertainty_by_confusion_outcome.png")
    write_text(label_summary, high_low, OUT / "label_specific_uncertainty_summary.txt")

    print("Wrote label-specific uncertainty analysis to", OUT)


if __name__ == "__main__":
    main()
