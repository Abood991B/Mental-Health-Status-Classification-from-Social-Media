# Mental Health Status Classification from Social Media Text

Multi-class classification and opinion mining pipeline for identifying mental health conditions from social media posts.

## Overview

This project builds a text classification pipeline that categorises social media posts into seven mental health statuses: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, and Personality disorder. It combines traditional machine learning, deep learning, and transformer-based approaches with opinion mining and aspect-based analysis.

The full workflow is contained in a single Jupyter notebook, with reusable modules in `src/` and standalone scripts in `scripts/`.

## Dataset

- **Source**: Kaggle - "Sentiment Analysis for Mental Health" by Suchintika Sarkar
- **File**: `data/Combined_Data.csv`
- **Records**: 53,043 raw; 52,543 after preprocessing
- **Classes**: 7 mental health status labels

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The notebook includes a deep learning section (TensorFlow) and a transformer section (PyTorch). The final run used TensorFlow CPU on Windows and PyTorch CUDA on an NVIDIA GTX 1660 Ti (6 GB VRAM). BiLSTM, BERT, and RoBERTa are included as optional extensions but were not executed due to hardware constraints.

## How to Run

Open and run the notebook:

```
notebooks/CDS6344_Full_Analysis_Notebook.ipynb
```

Individual pipeline stages can also be run via scripts:

```bash
python scripts/run_eda_preprocessing.py
python scripts/train_traditional_ml.py
python scripts/run_opinion_mining.py
python scripts/build_model_comparison_table.py
python scripts/verify_outputs.py
```

## Models

**Traditional ML** (TF-IDF features):
- Naive Bayes
- Logistic Regression (balanced class weights)
- Linear SVM (balanced class weights)

**Deep Learning** (Keras embeddings):
- CNN
- GRU

**Transformer** (PyTorch):
- DistilBERT fine-tuning

## Results

Tables and figures are saved under `results/tables/` and `results/screenshots/`. The table below summarises which models were executed:

| Model | Family | Status |
|---|---|---|
| Naive Bayes | Traditional ML | Executed |
| Logistic Regression | Traditional ML | Executed |
| Linear SVM | Traditional ML | Executed |
| CNN | Deep Learning | Executed |
| GRU | Deep Learning | Executed |
| DistilBERT | Transformer | Executed |
| BiLSTM | Deep Learning | Pending |
| BERT | Transformer | Pending |
| RoBERTa | Transformer | Pending |

## Generated Visualisations

- Class distribution and text length distribution
- Word clouds per class
- Confusion matrices for each traditional ML model
- Macro F1 comparison bar chart
- VADER sentiment distribution
- Top opinion targets per class
- Aspect mention heatmap

## Group Members

- Abdulrahman Baidaq - 241UC240L7 - TT2L

## Instructor

- Dr. Mohammad Shadab Khan (SHADAB.KHAN@MMU.EDU.MY)
