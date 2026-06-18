from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import ensure_nltk_resources
from src.visualization import PALETTE, setup_style


PROCESSED_PATH = ROOT / "data" / "processed" / "cleaned_full.csv"
TABLES_DIR = ROOT / "results" / "tables"
SCREENSHOTS_DIR = ROOT / "results" / "screenshots"

DOMAIN_ASPECTS = {
    "sleep": ["sleep", "insomnia", "nightmare", "tired"],
    "energy": ["energy", "exhausted", "fatigue", "drained"],
    "motivation": ["motivation", "motivated", "drive", "productive"],
    "appetite": ["appetite", "eat", "eating", "food", "weight"],
    "concentration": ["focus", "concentrate", "attention", "memory"],
    "social_support": ["friend", "family", "alone", "lonely", "relationship"],
    "medication": ["medication", "medicine", "meds", "pill", "drug"],
    "therapy": ["therapy", "therapist", "counseling", "doctor"],
    "self_harm": ["suicide", "suicidal", "selfharm", "harm", "kill"],
    "hopelessness": ["hopeless", "worthless", "empty", "pain", "life"],
}


def save_vader_distribution(df: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df, x="status", y="vader_compound", ax=ax, palette=PALETTE, cut=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("VADER Compound Sentiment Distribution by Mental Health Class")
    ax.set_xlabel("Mental health status")
    ax.set_ylabel("VADER compound score")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "vader_sentiment_distribution.png", dpi=300)
    plt.close(fig)


def aspect_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status, group in df.groupby("status"):
        class_text = " ".join(group["clean_text"].astype(str).tolist()).split()
        counts = Counter(class_text)
        total_tokens = max(1, sum(counts.values()))
        for aspect, keywords in DOMAIN_ASPECTS.items():
            hit_count = sum(counts[keyword] for keyword in keywords)
            rows.append({"status": status, "aspect": aspect, "mentions_per_10k_tokens": hit_count / total_tokens * 10000})
    return pd.DataFrame(rows)


def save_aspect_heatmap(heatmap_df: pd.DataFrame) -> None:
    setup_style()
    pivot = heatmap_df.pivot(index="aspect", columns="status", values="mentions_per_10k_tokens")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Mental Health Aspect Mentions per 10,000 Clean Tokens")
    ax.set_xlabel("Mental health status")
    ax.set_ylabel("Aspect")
    plt.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "absa_heatmap.png", dpi=300)
    plt.close(fig)


def windowed_aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for _, row in df.iterrows():
        tokens = str(row["clean_text"]).split()
        raw_text = str(row["statement"])
        for aspect, keywords in DOMAIN_ASPECTS.items():
            if any(keyword in tokens for keyword in keywords):
                score = analyzer.polarity_scores(raw_text)["compound"]
                if score >= 0.05:
                    sentiment = "positive"
                elif score <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                rows.append({"status": row["status"], "aspect": aspect, "sentiment": sentiment, "compound": score})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_nltk_resources()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Run scripts/run_eda_preprocessing.py first.")

    df = pd.read_csv(PROCESSED_PATH)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = df["statement"].astype(str).map(analyzer.polarity_scores)
    vader_df = pd.DataFrame(vader_scores.tolist()).add_prefix("vader_")
    scored_df = pd.concat([df.reset_index(drop=True), vader_df], axis=1)
    scored_df.to_csv(TABLES_DIR / "vader_scores_full.csv", index=False)

    vader_summary = scored_df.groupby("status")["vader_compound"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    vader_summary.to_csv(TABLES_DIR / "vader_sentiment_summary_by_class.csv", index=False)
    save_vader_distribution(scored_df)

    heatmap_df = aspect_heatmap(scored_df)
    heatmap_df.to_csv(TABLES_DIR / "aspect_mentions_by_class.csv", index=False)
    save_aspect_heatmap(heatmap_df)

    aspect_sentiment = windowed_aspect_sentiment(scored_df)
    aspect_sentiment.to_csv(TABLES_DIR / "aspect_sentiment_instances.csv", index=False)
    if not aspect_sentiment.empty:
        sentiment_summary = (
            aspect_sentiment.groupby(["status", "aspect", "sentiment"])
            .size()
            .reset_index(name="count")
            .sort_values(["status", "aspect", "count"], ascending=[True, True, False])
        )
        sentiment_summary.to_csv(TABLES_DIR / "aspect_sentiment_summary.csv", index=False)

    print(vader_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
