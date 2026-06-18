from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.preprocessing import LABEL_COLUMN, join_texts


PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#3B8EA5", "#6A994E", "#BC4749", "#7057A3"]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"


def save_class_distribution(distribution: pd.DataFrame, output_path: str | Path) -> None:
    setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = distribution.sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=ordered, x="count", y="status", hue="status", ax=ax, palette=PALETTE, legend=False)
    for patch, (_, row) in zip(ax.patches, ordered.iterrows()):
        ax.text(patch.get_width() + 150, patch.get_y() + patch.get_height() / 2, f"{row['count']:,} ({row['percentage']:.2f}%)", va="center")
    ax.set_title("Class Distribution After Final Cleaning")
    ax.set_xlabel("Number of posts")
    ax.set_ylabel("Mental health status")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_text_length_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(data=df, x="text_length_chars", hue=LABEL_COLUMN, bins=60, log_scale=(False, True), element="step", ax=ax)
    ax.set_title("Raw Text Length Distribution by Class")
    ax.set_xlabel("Characters per post")
    ax.set_ylabel("Number of posts (log scale)")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_wordcloud_grid(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(df[LABEL_COLUMN].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    for ax, label in zip(axes, labels):
        text = join_texts(df.loc[df[LABEL_COLUMN] == label, "clean_text"])
        cloud = WordCloud(width=900, height=500, background_color="white", colormap="viridis", max_words=120).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(label)
        ax.axis("off")
    for ax in axes[len(labels):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metric_bar(results: pd.DataFrame, metric: str, output_path: str | Path, title: str) -> None:
    setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = PALETTE[: max(1, len(ordered))]
    sns.barplot(data=ordered, x=metric, y="model", hue="model", ax=ax, palette=palette, legend=False)
    ax.set_title(title)
    ax.set_xlim(0, max(1.0, float(ordered[metric].max()) + 0.05))
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Model")
    for patch, value in zip(ax.patches, ordered[metric]):
        ax.text(patch.get_width() + 0.01, patch.get_y() + patch.get_height() / 2, f"{value:.3f}", va="center")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def top_terms_by_class(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby(LABEL_COLUMN):
        counter = Counter()
        for text in group["clean_text"].dropna():
            counter.update(str(text).split())
        for term, count in counter.most_common(top_n):
            rows.append({"status": label, "term": term, "count": count})
    return pd.DataFrame(rows)


def save_top_terms_chart(top_terms: pd.DataFrame, output_path: str | Path) -> None:
    setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(top_terms["status"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.ravel()
    for ax, label in zip(axes, labels):
        subset = top_terms[top_terms["status"] == label].sort_values("count", ascending=True)
        sns.barplot(data=subset, x="count", y="term", ax=ax, color="#2E86AB")
        ax.set_title(label)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("")
    for ax in axes[len(labels):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
