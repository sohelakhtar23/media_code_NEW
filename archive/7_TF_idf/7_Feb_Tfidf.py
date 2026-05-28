import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Load dataset

df = pd.read_csv("devset_videolist_GT.csv")
y_mem = df["memorability_score"].values
y_brand = df["brand_memorability"].values


# Create brand-isolated folds

def build_folds(df, n_folds=5):
    channel_counts = df["channelName"].value_counts()

    fold_map = {"Goldman Sachs": 0}
    fold_sizes = {i: 0 for i in range(n_folds)}
    fold_sizes[0] = channel_counts.get("Goldman Sachs", 0)

    for channel, count in channel_counts.drop("Goldman Sachs", errors="ignore").items():
        target_fold = min(range(1, n_folds), key=lambda f: fold_sizes[f])
        fold_map[channel] = target_fold
        fold_sizes[target_fold] += count

    return df["channelName"].map(fold_map)


df["fold"] = build_folds(df)


# Load transcripts

def load_transcripts(video_ids, stt_dir="devset-stt"):
    transcripts = []
    stt_dir = Path(stt_dir)

    for vid in video_ids:
        text = ""

        for name in [vid, f"_{vid}"]:
            file_path = stt_dir / f"{name}.txt"

            if file_path.exists():
                text = file_path.read_text(encoding="utf-8").strip()
                break

        transcripts.append(text)

    return transcripts


corpus = load_transcripts(df["id"])


# TF-IDF features

tfidf = TfidfVectorizer(
    max_features=3000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 3),
    stop_words="english",
    sublinear_tf=True,
)

X_tfidf = tfidf.fit_transform(corpus)


# Rank aggregation evaluation

def rank_aggregate_cv(features, y, folds, threshold=0.03):
    predictions = np.zeros(len(y))

    for fold_id in range(5):
        val_mask = folds == fold_id
        train_mask = ~val_mask

        X_train = features[train_mask]
        X_val = features[val_mask]
        y_train = y[train_mask]

        scores = np.zeros(val_mask.sum())

        for col in range(X_train.shape[1]):
            r, _ = spearmanr(X_train[:, col], y_train)

            if np.isnan(r) or abs(r) < threshold:
                continue

            ranked = rankdata(X_val[:, col])
            scores += np.sign(r) * ranked

        predictions[val_mask] = scores

    return spearmanr(y, predictions).correlation, predictions


# Search best SVD dimensions

best_mem_score, best_mem_comp = -1, 50
best_brand_score, best_brand_comp = -1, 50

print(f"{'n_comp':<10} {'Mem r':>8} {'Brand r':>8}")
print("-" * 30)

for n_comp in [50, 80, 100, 120, 150, 200]:
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)

    mem_score, _ = rank_aggregate_cv(X_lsa, y_mem, df["fold"].values)
    brand_score, _ = rank_aggregate_cv(X_lsa, y_brand, df["fold"].values)

    print(f"{n_comp:<10} {mem_score:>8.4f} {brand_score:>8.4f}")

    if mem_score > best_mem_score:
        best_mem_score = mem_score
        best_mem_comp = n_comp

    if brand_score > best_brand_score:
        best_brand_score = brand_score
        best_brand_comp = n_comp


print(f"\nBest memorability n_comp: {best_mem_comp}")
print(f"Best brand n_comp: {best_brand_comp}")


# Final evaluation

def evaluate_target(n_comp, y):
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)

    score, predictions = rank_aggregate_cv(X_lsa, y, df["fold"].values)
    return score, predictions


mem_score, mem_preds = evaluate_target(best_mem_comp, y_mem)
brand_score, brand_preds = evaluate_target(best_brand_comp, y_brand)

print(f"\nMemorability Spearman r: {mem_score:.4f}")
print(f"Brand Memorability Spearman r: {brand_score:.4f}")
