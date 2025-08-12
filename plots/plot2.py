#!/usr/bin/env python3
"""
Visualizing self-sycophancy experiment results.

Usage:
  python plot_self_sycophancy_results.py --csv results.csv --out plots/ --n 100

What it makes:
  - Calibrated effect by model (per dataset), for harmfulness and correctness
  - Aggregated calibrated effect by model family (per dataset)
  - Scatter: calibrated effect vs position bias (anchoring)
  - Scatter: calibrated effect vs prefill amplification
  - Scatter: calibrated effect vs own-choice minus random label
  - Detection rate vs calibrated effect
  - Prints summary stats per domain/dataset/model family

Notes:
  - We approximate z-scores for calibrated effects using std of raw SSI and N (items/run).
    This is only an approximation; use with caution.
"""

import argparse
import os
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define consistent color schemes
MODEL_FAMILY_COLORS = {
    "Anthropic": "#E74C3C",  # Red
    "OpenAI": "#3498DB",  # Blue
    "Google": "#2ECC71",  # Green
    "OSS": "#F39C12",  # Orange
    "Other": "#9B59B6"  # Purple
}

DOMAIN_COLORS = {
    "harmfulness": "#1ABC9C",  # Teal
    "correctness": "#E67E22",  # Dark Orange
    "unknown": "#95A5A6"  # Gray
}

# Marker styles for domains
DOMAIN_MARKERS = {
    "harmfulness": "o",
    "correctness": "s",
    "unknown": "^"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="self-sycophancy-1-dedup.csv", help="Path to CSV with results")
    p.add_argument("--out", default="plots", help="Output directory for charts")
    p.add_argument("--n", type=int, default=100, help="Items per run (default 100)")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    p.add_argument("--style", default="whitegrid", help="Seaborn style")
    return p.parse_args()


def ensure_dir(pth):
    os.makedirs(pth, exist_ok=True)
    return pth


def parse_domain_dataset(task_name: str):
    """
    Expected task_name format:
      self_sycophancy_<domain>_<dataset>_comprehensive
    e.g., self_sycophancy_harmfulness_commonsense_comprehensive
          self_sycophancy_correctness_arc_challenge_comprehensive
    """
    if not isinstance(task_name, str):
        return "unknown", "unknown"
    parts = task_name.split("_")
    if len(parts) < 4:
        return "unknown", "unknown"
    domain = parts[2]
    # dataset can be multi-token (e.g., arc_challenge); strip trailing 'comprehensive' if present
    ds_tokens = parts[3:]
    if ds_tokens and ds_tokens[-1].lower() == "comprehensive":
        ds_tokens = ds_tokens[:-1]
    dataset = "_".join(ds_tokens) if ds_tokens else "unknown"
    return domain, dataset


def parse_model_family(model: str):
    if not isinstance(model, str):
        return "Other"
    m = model.lower()
    if "/openai/gpt-oss" in m:
        return "OSS"
    if "/openai/" in m:
        return "OpenAI"
    if "/anthropic/" in m:
        return "Anthropic"
    if "/google/" in m:
        return "Google"
    return "Other"


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_derived(df, n_items=100):
    # domain/dataset
    parsed = df["task_name"].apply(parse_domain_dataset)
    df["domain"] = [d for d, _ in parsed]
    df["dataset"] = [s for _, s in parsed]
    # family
    df["model_family"] = df["model"].apply(parse_model_family)

    # coerce numerics
    num_cols = [
        "score_calibrated_effect_scorer_mean",
        "score_comprehensive_ssi_scorer_mean",
        "score_comprehensive_ssi_scorer_std",
        "score_choice_vs_prefill_scorer_mean",
        "score_actual_vs_random_scorer_mean",
        "score_comprehensive_detection_scorer_mean",
        "score_consistency_scorer_mean",
        "score_position_bias_scorer_mean",
    ]
    df = coerce_numeric(df, num_cols)

    # position bias SD (sqrt of variance)
    df["position_bias_sd"] = np.sqrt(df["score_position_bias_scorer_mean"].clip(lower=0))

    # approximate SE for calibrated effect using raw SSI std and N (approx only)
    # Beware: std here is of raw SSI; using for calibrated SE is an approximation.
    df["ssi_se_approx"] = df["score_comprehensive_ssi_scorer_std"] / np.sqrt(max(n_items, 1))
    # z approx and significance flag
    df["z_cal_approx"] = df["score_calibrated_effect_scorer_mean"] / df["ssi_se_approx"].replace(0, np.nan)
    df["sig_approx"] = df["z_cal_approx"].abs() > 1.96

    # nicer labels
    df["model_label"] = df["model_display_name"].fillna(df["model"])
    df["ds_label"] = df["dataset"].str.replace("_", " ").str.title()
    df["domain_label"] = df["domain"].str.title()

    return df


def bar_calibrated_by_model(df, outdir, dpi):
    # one chart per domain/dataset: horizontal bars sorted by calibrated effect
    cols = ["domain", "dataset", "model_label", "model_family", "score_calibrated_effect_scorer_mean", "ssi_se_approx"]
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean"])[cols].copy()
    if d.empty:
        return

    for (domain, dataset), g in sorted(d.groupby(["domain", "dataset"])):
        if g.empty:
            continue
        g = g.sort_values("score_calibrated_effect_scorer_mean", ascending=True)

        # Map colors
        colors = [MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in g["model_family"]]

        plt.figure(figsize=(10, max(4, 0.35 * len(g))))
        ax = plt.gca()

        # Create horizontal bar chart with custom colors
        y_pos = np.arange(len(g))
        bars = ax.barh(y_pos, g["score_calibrated_effect_scorer_mean"], color=colors)

        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(g["model_label"])
        ax.set_title(f"Calibrated Effect by Model\n{domain.title()} • {dataset.replace('_', ' ').title()}")
        ax.set_xlabel("Calibrated effect (SSI_actual − mean(SSI_forced))")

        # annotate values
        for i, (idx, row) in enumerate(g.iterrows()):
            ax.text(
                row["score_calibrated_effect_scorer_mean"] + (
                    0.02 if row["score_calibrated_effect_scorer_mean"] >= 0 else -0.02),
                i,
                f"{row['score_calibrated_effect_scorer_mean']:.2f}",
                va="center",
                ha="left" if row["score_calibrated_effect_scorer_mean"] >= 0 else "right",
                fontsize=8,
                color="black"
            )

        # Add legend for model families
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=MODEL_FAMILY_COLORS[fam], label=fam)
                           for fam in sorted(g["model_family"].unique())]
        ax.legend(handles=legend_elements, title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        fname = os.path.join(outdir, f"calibrated_by_model_{domain}_{dataset}.png")
        plt.savefig(fname, dpi=dpi)
        plt.close()


def grouped_bar_family(df, outdir, dpi):
    # aggregate by domain/dataset/model_family
    cols = ["domain", "dataset", "model_family", "score_calibrated_effect_scorer_mean"]
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean"])[cols].copy()
    if d.empty:
        return
    agg = d.groupby(["domain", "dataset", "model_family"], as_index=False)["score_calibrated_effect_scorer_mean"].mean()

    for domain, g in sorted(agg.groupby("domain")):
        plt.figure(figsize=(10, 5))

        # Get sorted list of model families for consistent ordering
        families = sorted(g["model_family"].unique())
        palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}

        ax = sns.barplot(
            data=g,
            x="dataset",
            y="score_calibrated_effect_scorer_mean",
            hue="model_family",
            palette=palette,
            hue_order=families
        )
        ax.axhline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(f"Calibrated Effect by Model Family • {domain.title()}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean calibrated effect")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fname = os.path.join(outdir, f"calibrated_by_family_{domain}.png")
        plt.savefig(fname, dpi=dpi)
        plt.close()


def scatter_effect_vs_anchor(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "position_bias_sd"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(8, 6))

    # Get sorted lists for consistent ordering
    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    ax = sns.scatterplot(
        data=d,
        x="position_bias_sd",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=70
    )
    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Calibrated Effect vs Position Bias (SD across forced labels)")
    ax.set_xlabel("Position bias (SD)")
    ax.set_ylabel("Calibrated effect")

    # annotate top |effect| points
    d["abs_effect"] = d["score_calibrated_effect_scorer_mean"].abs()
    for _, row in d.nlargest(5, "abs_effect").iterrows():
        ax.annotate(
            f"{row['model_family']} | {row['dataset']}",
            (row["position_bias_sd"], row["score_calibrated_effect_scorer_mean"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8
        )
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_position_bias.png"), dpi=dpi)
    plt.close()


def scatter_effect_vs_prefill(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_choice_vs_prefill_scorer_mean"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(8, 6))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    ax = sns.scatterplot(
        data=d,
        x="score_choice_vs_prefill_scorer_mean",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=70
    )
    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Calibrated Effect vs Prefill Amplification")
    ax.set_xlabel("Prefill amplification (post_prefilled − post_choice)")
    ax.set_ylabel("Calibrated effect")

    # trend line
    sns.regplot(
        data=d, x="score_choice_vs_prefill_scorer_mean", y="score_calibrated_effect_scorer_mean",
        scatter=False, color="gray", line_kws={"alpha": 0.6}
    )
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_prefill.png"), dpi=dpi)
    plt.close()


def scatter_effect_vs_random(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_actual_vs_random_scorer_mean"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(8, 6))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    ax = sns.scatterplot(
        data=d,
        x="score_actual_vs_random_scorer_mean",
        y="score_calibrated_effect_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=70
    )
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.axhline(0, color="k", lw=1, alpha=0.6)
    ax.set_title("Own-Choice vs Random-Label Bias")
    ax.set_xlabel("SSI(actual) − SSI(random prefill)")
    ax.set_ylabel("Calibrated effect")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_random.png"), dpi=dpi)
    plt.close()


def scatter_detection_vs_effect(df, outdir, dpi):
    d = df.dropna(subset=["score_calibrated_effect_scorer_mean", "score_comprehensive_detection_scorer_mean"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(8, 6))

    families = sorted(d["model_family"].unique())
    domains = sorted(d["domain"].unique())

    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}
    markers = {dom: DOMAIN_MARKERS.get(dom, "^") for dom in domains}

    ax = sns.scatterplot(
        data=d,
        x="score_calibrated_effect_scorer_mean",
        y="score_comprehensive_detection_scorer_mean",
        hue="model_family",
        style="domain",
        palette=palette,
        hue_order=families,
        style_order=domains,
        markers=markers,
        s=70
    )
    ax.axvline(0.5, color="r", ls="--", lw=1, alpha=0.5, label="Threshold=0.5 (effect)")
    ax.set_title("Detection Rate vs Calibrated Effect")
    ax.set_xlabel("Calibrated effect")
    ax.set_ylabel("Detection rate (fraction > 0.5)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "detection_vs_effect.png"), dpi=dpi)
    plt.close()


def bar_consistency(df, outdir, dpi):
    d = df.dropna(subset=["score_consistency_scorer_mean"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(10, 5))

    families = sorted(d["model_family"].unique())
    palette = {fam: MODEL_FAMILY_COLORS.get(fam, "#95A5A6") for fam in families}

    ax = sns.boxplot(
        data=d,
        x="domain",
        y="score_consistency_scorer_mean",
        hue="model_family",
        palette=palette,
        hue_order=families
    )
    ax.set_title("Consistency (higher is better) by Domain and Model Family")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Consistency (≈ 1/(1 + drift))")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consistency_by_domain_family.png"), dpi=dpi)
    plt.close()


def print_summary(df, n_items):
    def fmt(x):
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    print("\n=== Summary (aggregated across rows) ===")
    # By domain / dataset
    gcols = ["domain", "dataset"]
    agg = df.groupby(gcols).agg(
        mean_calibrated=("score_calibrated_effect_scorer_mean", "mean"),
        sd_calibrated=("score_calibrated_effect_scorer_mean", "std"),
        mean_position_var=("score_position_bias_scorer_mean", "mean"),
        mean_position_sd=("position_bias_sd", "mean"),
        mean_prefill=("score_choice_vs_prefill_scorer_mean", "mean"),
        mean_detection=("score_comprehensive_detection_scorer_mean", "mean"),
        mean_consistency=("score_consistency_scorer_mean", "mean"),
        n=("score_calibrated_effect_scorer_mean", "count")
    ).reset_index()
    for _, row in agg.iterrows():
        print(f"- {row['domain']}/{row['dataset']}: "
              f"cal_mean={fmt(row['mean_calibrated'])}, "
              f"pos_sd={fmt(row['mean_position_sd'])}, "
              f"prefill={fmt(row['mean_prefill'])}, "
              f"detect={fmt(row['mean_detection'])}, "
              f"consistency={fmt(row['mean_consistency'])}, "
              f"runs={int(row['n'])}")

    # By model family
    print("\n=== Calibrated effect by model family (overall) ===")
    fam = df.groupby("model_family")["score_calibrated_effect_scorer_mean"].mean().sort_values(ascending=False)
    for k, v in fam.items():
        print(f"- {k}: {fmt(v)}")

    # Count of approximate significant positives/negatives
    sig = df.dropna(subset=["z_cal_approx"])
    if not sig.empty:
        pos = ((sig["z_cal_approx"] > 1.96) & (sig["score_calibrated_effect_scorer_mean"] > 0)).sum()
        neg = ((sig["z_cal_approx"] < -1.96) & (sig["score_calibrated_effect_scorer_mean"] < 0)).sum()
        tot = len(sig)
        print(f"\nApprox significant (z>1.96 using raw SSI std, N={n_items}): "
              f"pos={pos}, neg={neg}, total={tot}")
    else:
        print("\nApprox significance not computed (insufficient columns).")


def main():
    args = parse_args()
    sns.set_style(args.style)
    ensure_dir(args.out)

    # Load
    df = pd.read_csv(args.csv)

    # Prepare
    df = add_derived(df, n_items=args.n)

    # Separate for clarity if needed
    # df_harm = df[df["domain"] == "harmfulness"].copy()
    # df_corr = df[df["domain"] == "correctness"].copy()

    # Charts
    bar_calibrated_by_model(df, args.out, args.dpi)
    grouped_bar_family(df, args.out, args.dpi)
    scatter_effect_vs_anchor(df, args.out, args.dpi)
    scatter_effect_vs_prefill(df, args.out, args.dpi)
    scatter_effect_vs_random(df, args.out, args.dpi)
    scatter_detection_vs_effect(df, args.out, args.dpi)
    bar_consistency(df, args.out, args.dpi)

    # Summary text
    print_summary(df, args.n)

    print(f"\nCharts saved to: {os.path.abspath(args.out)}")

    # Print color legend for reference
    print("\n=== Color Scheme Reference ===")
    print("Model Families:")
    for family, color in sorted(MODEL_FAMILY_COLORS.items()):
        print(f"  {family}: {color}")
    print("\nDomains:")
    for domain, color in sorted(DOMAIN_COLORS.items()):
        print(f"  {domain}: {color} (marker: {DOMAIN_MARKERS.get(domain, '^')})")


if __name__ == "__main__":
    main()