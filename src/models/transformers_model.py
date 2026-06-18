from __future__ import annotations

MODEL_CONFIGS = {
    "BERT": "bert-base-uncased",
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa": "roberta-base",
}


def require_transformers():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError("PyTorch and transformers are required for transformer fine-tuning.") from exc
    return torch, transformers
