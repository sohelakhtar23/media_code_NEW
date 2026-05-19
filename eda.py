"""
EDA for VIDEM — Commercial Video Memorability Prediction
MediaEval 2026, Subtask 2

Run from the project root:
    python eda.py --csv path/to/devset_videolist_GT.csv --stt path/to/devset-stt
"""

import argparse
import os
from unittest import result
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#888",
    "ytick.color":      "#888",
    "text.color":       "#eee",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "font.family":      "monospace",
    "figure.titlesize": 14,
})
ACCENT1 = "#00d4ff"   # cyan  — video memorability
ACCENT2 = "#ff6b6b"   # red   — brand memorability
ACCENT3 = "#ffd166"   # gold  — joint / neutral

# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True, help="Path to devset_videolist_GT.csv")
    p.add_argument("--stt",  default=None,  help="Path to devset-stt/ directory")
    p.add_argument("--out",  default="eda_output", help="Output directory for figures")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & BASIC SANITY
# ═════════════════════════════════════════════════════════════════════════════
def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print("\n" + "═"*60)
    print("  DATASET OVERVIEW")
    print("═"*60)
    print(f"  Rows            : {len(df)}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"\n  Nulls per column:")
    print(df.isnull().sum().to_string(header=False))
    print(f"\n  dtypes:")
    print(df.dtypes.to_string(header=False))

    # Derived features we'll reuse throughout
    df["log_views"]      = np.log1p(df["viewsCount"])
    df["log_likes"]      = np.log1p(df["likesCount"])
    df["log_comments"]   = np.log1p(df["commentsCount"])
    df["log_duration"]   = np.log1p(df["durationSeconds"])
    df["is_long"]        = (df["durationSeconds"] > 60).astype(int)
    df["first60_frac"]   = np.minimum(df["durationSeconds"], 60) / df["durationSeconds"]

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. TARGET DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════
def plot_targets(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Target Score Distributions", fontsize=15, color="#eee", y=1.02)

    # --- memorability_score histogram
    ax = axes[0]
    ax.hist(df["memorability_score"], bins=30, color=ACCENT1, alpha=0.85, edgecolor="#0f0f0f")
    ax.axvline(df["memorability_score"].mean(), color="white", lw=1.5, linestyle="--", label=f"mean={df['memorability_score'].mean():.3f}")
    ax.axvline(df["memorability_score"].median(), color=ACCENT3, lw=1.5, linestyle=":", label=f"median={df['memorability_score'].median():.3f}")
    ax.set_title("Video Memorability", color=ACCENT1)
    ax.set_xlabel("Score"); ax.legend(fontsize=8)

    # --- brand_memorability histogram
    ax = axes[1]
    ax.hist(df["brand_memorability"], bins=30, color=ACCENT2, alpha=0.85, edgecolor="#0f0f0f")
    ax.axvline(df["brand_memorability"].mean(), color="white", lw=1.5, linestyle="--", label=f"mean={df['brand_memorability'].mean():.3f}")
    ax.axvline(df["brand_memorability"].median(), color=ACCENT3, lw=1.5, linestyle=":", label=f"median={df['brand_memorability'].median():.3f}")
    ax.set_title("Brand Memorability", color=ACCENT2)
    ax.set_xlabel("Score"); ax.legend(fontsize=8)

    # --- joint scatter
    ax = axes[2]
    sc = ax.scatter(df["memorability_score"], df["brand_memorability"],
                    alpha=0.5, s=25, c=df["log_duration"], cmap="plasma")
    r, p = stats.spearmanr(df["memorability_score"], df["brand_memorability"])
    ax.set_title(f"Joint (Spearman ρ={r:.3f}, p={p:.2e})", color=ACCENT3)
    ax.set_xlabel("Video Memorability", color=ACCENT1)
    ax.set_ylabel("Brand Memorability", color=ACCENT2)
    plt.colorbar(sc, ax=ax, label="log(duration)")

    plt.tight_layout()
    fig.savefig(out / "01_target_distributions.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # --- Print summary stats
    print("\n" + "═"*60)
    print("  TARGET STATISTICS")
    print("═"*60)
    for col in ["memorability_score", "brand_memorability"]:
        s = df[col]
        skew  = stats.skew(s.dropna())
        kurt  = stats.kurtosis(s.dropna())
        print(f"\n  {col}")
        print(f"    mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
        print(f"    skewness={skew:.3f}  kurtosis={kurt:.3f}")
        print(f"    25th pct={s.quantile(.25):.4f}  75th pct={s.quantile(.75):.4f}")

    r, p = stats.spearmanr(df["memorability_score"], df["brand_memorability"])
    print(f"\n  Spearman ρ(video, brand) = {r:.4f}  p = {p:.2e}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. CHANNEL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def plot_channels(df: pd.DataFrame, out: Path):
    ch_stats = (
        df.groupby("channelName")[["memorability_score", "brand_memorability"]]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    ch_stats.columns = ["mem_mean", "mem_std", "mem_n",
                         "brand_mean", "brand_std", "brand_n"]
    ch_stats = ch_stats.sort_values("mem_mean", ascending=False)

    print("\n" + "═"*60)
    print("  CHANNEL STATISTICS (sorted by video memorability)")
    print("═"*60)
    print(ch_stats.to_string())

    top_n = min(20, len(ch_stats))
    ch_top = ch_stats.head(top_n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Per-Channel Memorability (top 20 by video mem)", fontsize=13, color="#eee")

    y = np.arange(top_n)
    for ax, col, err_col, color, label in [
        (axes[0], "mem_mean",   "mem_std",   ACCENT1, "Video Memorability"),
        (axes[1], "brand_mean", "brand_std", ACCENT2, "Brand Memorability"),
    ]:
        vals  = ch_top[col].values
        errs  = ch_top[err_col].fillna(0).values
        bars  = ax.barh(y, vals, xerr=errs, color=color, alpha=0.8,
                        error_kw=dict(ecolor="#666", capsize=3), edgecolor="#0f0f0f")
        ax.set_yticks(y)
        ax.set_yticklabels(ch_top.index, fontsize=8)
        ax.set_title(label, color=color)
        ax.invert_yaxis()
        # annotate count
        for i, (v, n) in enumerate(zip(vals, ch_top["mem_n"])):
            ax.text(v + 0.002, i, f"n={int(n)}", va="center", fontsize=7, color="#888")

    plt.tight_layout()
    fig.savefig(out / "02_channel_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 4. METADATA FEATURES vs TARGETS
# ═════════════════════════════════════════════════════════════════════════════
def plot_metadata_correlations(df: pd.DataFrame, out: Path):
    feat_cols = [
        "log_views", "log_likes", "log_comments", "engagementRate",
        "log_duration", "nb_annotations", "durationSeconds",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    print("\n" + "═"*60)
    print("  SPEARMAN CORRELATIONS — METADATA vs TARGETS")
    print("═"*60)

    rows = []
    for fc in feat_cols:
        valid = df[[fc, "memorability_score", "brand_memorability"]].dropna()
        r1, p1 = stats.spearmanr(valid[fc], valid["memorability_score"])
        r2, p2 = stats.spearmanr(valid[fc], valid["brand_memorability"])
        rows.append({"feature": fc, "ρ_video": r1, "p_video": p1,
                                     "ρ_brand": r2, "p_brand": p2})
        print(f"  {fc:25s}  ρ_video={r1:+.3f} (p={p1:.2e})   ρ_brand={r2:+.3f} (p={p2:.2e})")

    corr_df = pd.DataFrame(rows).set_index("feature")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(corr_df))
    w = 0.35
    ax.bar(x - w/2, corr_df["ρ_video"], width=w, label="Video Memorability",
           color=ACCENT1, alpha=0.85, edgecolor="#0f0f0f")
    ax.bar(x + w/2, corr_df["ρ_brand"], width=w, label="Brand Memorability",
           color=ACCENT2, alpha=0.85, edgecolor="#0f0f0f")
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Metadata Feature Correlations with Targets")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "03_metadata_correlations.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 5. DURATION ANALYSIS (critical because of 60s annotation cap)
# ═════════════════════════════════════════════════════════════════════════════
def plot_duration(df: pd.DataFrame, out: Path):
    print("\n" + "═"*60)
    print("  DURATION ANALYSIS")
    print("═"*60)
    pct_long = (df["durationSeconds"] > 60).mean() * 100
    print(f"  Videos > 60s  : {pct_long:.1f}%")
    print(f"  Median dur    : {df['durationSeconds'].median():.0f}s  ({df['durationSeconds'].median()/60:.1f} min)")
    print(f"  Max dur       : {df['durationSeconds'].max():.0f}s  ({df['durationSeconds'].max()/60:.1f} min)")
    print(f"  Min dur       : {df['durationSeconds'].min():.0f}s")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Duration Analysis (annotators watched ≤60s)", fontsize=13, color="#eee")

    # histogram of durations
    ax = axes[0]
    ax.hist(df["durationSeconds"].clip(upper=600), bins=40,
            color=ACCENT3, alpha=0.8, edgecolor="#0f0f0f")
    ax.axvline(60,  color=ACCENT2, lw=2, linestyle="--", label="60s cap")
    ax.axvline(df["durationSeconds"].median(), color="white", lw=1.5,
               linestyle=":", label=f"median={df['durationSeconds'].median():.0f}s")
    ax.set_xlabel("Duration (s, clipped at 600)")
    ax.set_title("Duration Distribution")
    ax.legend(fontsize=8)

    # memorability vs is_long
    ax = axes[1]
    for i, (label, color) in enumerate(
        [("≤60s (n=%d)" % (df["is_long"]==0).sum(), ACCENT1),
         (">60s (n=%d)" % (df["is_long"]==1).sum(), ACCENT2)]
    ):
        subset = df[df["is_long"] == i]["memorability_score"].dropna()
        ax.hist(subset, bins=20, alpha=0.65, label=label, color=color, edgecolor="#0f0f0f")
    r, p = stats.mannwhitneyu(
        df[df["is_long"]==0]["memorability_score"].dropna(),
        df[df["is_long"]==1]["memorability_score"].dropna(),
    )
    ax.set_title(f"Video Mem: short vs long\nMann-Whitney p={p:.3f}")
    ax.set_xlabel("memorability_score")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "04_duration_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 6. ANNOTATION COUNT BIAS
# ═════════════════════════════════════════════════════════════════════════════
def plot_annotation_bias(df: pd.DataFrame, out: Path):
    print("\n" + "═"*60)
    print("  ANNOTATION COUNT ANALYSIS")
    print("═"*60)
    print(f"  nb_annotations — min={df['nb_annotations'].min()}  max={df['nb_annotations'].max()}  "
          f"mean={df['nb_annotations'].mean():.1f}  std={df['nb_annotations'].std():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Annotation Count vs Target Reliability", fontsize=13, color="#eee")

    for ax, col, color in [
        (axes[0], "memorability_score", ACCENT1),
        (axes[1], "brand_memorability",  ACCENT2),
    ]:
        ax.scatter(df["nb_annotations"], df[col], alpha=0.4, s=20, color=color)
        r, p = stats.spearmanr(df["nb_annotations"].dropna(), df[col].dropna())
        ax.set_xlabel("nb_annotations")
        ax.set_ylabel(col)
        ax.set_title(f"ρ={r:.3f}, p={p:.3f}")

    plt.tight_layout()
    fig.savefig(out / "05_annotation_bias.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 7. CATEGORY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def plot_categories(df: pd.DataFrame, out: Path):
    if "categoryName" not in df.columns:
        return

    cat_stats = (
        df.groupby("categoryName")[["memorability_score", "brand_memorability"]]
        .agg(["mean", "count"])
    )
    print("\n" + "═"*60)
    print("  CATEGORY BREAKDOWN")
    print("═"*60)
    print(cat_stats.to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 8. STT TRANSCRIPT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def analyze_stt(df: pd.DataFrame, stt_dir: str, out: Path):
    if stt_dir is None or not os.path.isdir(stt_dir):
        print("\n  [STT] No STT directory provided / found — skipping.")
        return

    stt_path = Path(stt_dir)
    lengths, empty_ids = [], []

    for _, row in df.iterrows():
        fpath = stt_path / f"{row['id']}.txt"
        if fpath.exists():
            txt = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            lengths.append(len(txt.split()))
            if len(txt) == 0:
                empty_ids.append(row["id"])
        else:
            lengths.append(np.nan)

    df["stt_word_count"] = lengths

    print("\n" + "═"*60)
    print("  STT TRANSCRIPT ANALYSIS")
    print("═"*60)
    print(f"  Missing transcript files : {df['stt_word_count'].isna().sum()}")
    print(f"  Empty transcripts        : {len(empty_ids)}  — ids: {empty_ids[:5]}")
    wc = df["stt_word_count"].dropna()
    print(f"  Word count — min={wc.min():.0f}  max={wc.max():.0f}  "
          f"mean={wc.mean():.0f}  median={wc.median():.0f}")

    r1, p1 = stats.spearmanr(wc, df.loc[wc.index, "memorability_score"])
    r2, p2 = stats.spearmanr(wc, df.loc[wc.index, "brand_memorability"])
    print(f"\n  ρ(word_count, video_mem) = {r1:.3f}  p={p1:.3f}")
    print(f"  ρ(word_count, brand_mem) = {r2:.3f}  p={p2:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("STT Word Count vs Targets", fontsize=13, color="#eee")
    for ax, col, color, r, p in [
        (axes[0], "memorability_score", ACCENT1, r1, p1),
        (axes[1], "brand_memorability",  ACCENT2, r2, p2),
    ]:
        ax.scatter(df["stt_word_count"], df[col], alpha=0.4, s=20, color=color)
        ax.set_xlabel("STT word count")
        ax.set_ylabel(col)
        ax.set_title(f"ρ={r:.3f}, p={p:.3f}")

    plt.tight_layout()
    fig.savefig(out / "06_stt_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 9. FEATURE CORRELATION HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame, out: Path):
    num_cols = [c for c in [
        "memorability_score", "brand_memorability",
        "log_views", "log_likes", "log_comments",
        "engagementRate", "log_duration", "nb_annotations",
        "stt_word_count",
    ] if c in df.columns]

    corr = df[num_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#0f0f0f",
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlation Heatmap (all numeric features)", color="#eee", pad=12)
    plt.tight_layout()
    fig.savefig(out / "07_correlation_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(args.csv)
    plot_targets(df, out)
    plot_channels(df, out)
    plot_metadata_correlations(df, out)
    plot_duration(df, out)
    plot_annotation_bias(df, out)
    plot_categories(df, out)

    # df = analyze_stt(df, args.stt, out) or df
    # plot_correlation_heatmap(df, out)
    result = analyze_stt(df, args.stt, out)
    if result is not None:
        df = result
    # df = analyze_stt(df, args.stt, out) or df
        plot_correlation_heatmap(df, out)

    print("\n" + "═"*60)
    print(f"  EDA complete. Figures saved to: {out.resolve()}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
