from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


RAW_TEXT_COLUMN = "statement"
LABEL_COLUMN = "status"
RANDOM_STATE = 42


NEGATION_WORDS = {"no", "nor", "not", "never", "none", "nobody", "nothing", "nowhere", "neither"}


def ensure_nltk_resources() -> None:
    """Download small NLTK resources required by the preprocessing notebooks."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the original CSV and keep only the modelling columns."""
    df = pd.read_csv(path)
    missing = {RAW_TEXT_COLUMN, LABEL_COLUMN}.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df[[RAW_TEXT_COLUMN, LABEL_COLUMN]].copy()


def _basic_normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str, keep_negations: bool = True) -> str:
    """Normalize, tokenize, remove stopwords, and lemmatize text."""
    ensure_nltk_resources()
    normalized = _basic_normalize(text)
    try:
        tokens = word_tokenize(normalized)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(normalized)

    stop_words = set(stopwords.words("english"))
    if keep_negations:
        stop_words = stop_words.difference(NEGATION_WORDS)

    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 1
    ]
    return " ".join(cleaned_tokens)


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the planned preprocessing pipeline and return final rows plus audit metrics."""
    raw_rows = len(df)
    null_statements = int(df[RAW_TEXT_COLUMN].isna().sum())
    non_null = df.dropna(subset=[RAW_TEXT_COLUMN]).copy()
    non_null[RAW_TEXT_COLUMN] = non_null[RAW_TEXT_COLUMN].astype(str)
    non_null["text_length_chars"] = non_null[RAW_TEXT_COLUMN].str.len()
    non_null["word_count_raw"] = non_null[RAW_TEXT_COLUMN].str.split().str.len()
    non_null["clean_text"] = non_null[RAW_TEXT_COLUMN].map(clean_text)
    empty_after_cleaning = int((non_null["clean_text"].str.len() == 0).sum())
    final_df = non_null[non_null["clean_text"].str.len() > 0].copy()
    final_df["clean_word_count"] = final_df["clean_text"].str.split().str.len()

    audit = {
        "raw_rows": raw_rows,
        "null_statements": null_statements,
        "rows_after_null_removal": len(non_null),
        "empty_after_cleaning": empty_after_cleaning,
        "final_modelling_rows": len(final_df),
        "avg_text_length_chars_after_null": round(float(non_null["text_length_chars"].mean()), 2),
        "median_text_length_chars_after_null": int(non_null["text_length_chars"].median()),
        "max_text_length_chars_after_null": int(non_null["text_length_chars"].max()),
        "avg_word_count_after_null": round(float(non_null["word_count_raw"].mean()), 2),
        "median_word_count_after_null": int(non_null["word_count_raw"].median()),
        "max_word_count_after_null": int(non_null["word_count_raw"].max()),
    }
    return final_df, audit


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[LABEL_COLUMN].value_counts().rename_axis("status").reset_index(name="count")
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = df.copy()
    encoded["label"] = encoder.fit_transform(encoded[LABEL_COLUMN])
    return encoded, encoder


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def join_texts(texts: Iterable[str]) -> str:
    return " ".join(str(text) for text in texts if isinstance(text, str) and text.strip())
