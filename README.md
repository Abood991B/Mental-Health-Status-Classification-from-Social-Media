# Mental Health Status Classification from Social Media Text
## CDS6344 Social Media Computing - Individual Assignment Submission

### Project Overview

This repository contains the implementation for the CDS6344 Social Media Computing assignment titled "Mental Health Status Classification from Social Media Text: A Multi-Class Sentiment Analysis and Opinion Mining Pipeline."

The project builds a reproducible NLP pipeline for classifying social-media/user-generated text into seven mental-health-related status classes: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, and Personality disorder. It includes verified dataset preprocessing, exploratory analysis, traditional machine-learning baselines, opinion mining, ABSA-style aspect analysis, and GPU-ready deep-learning and transformer experiment sections in one combined notebook.

### Problem Statement

Millions of individuals express mental health struggles, including depression, anxiety, and suicidal ideation, on platforms like Twitter and Reddit. Traditional clinical systems are too slow and costly for early detection at scale. This project builds a multi-class sentiment analysis and opinion mining pipeline to classify social media posts by mental health status, enabling early detection of at-risk individuals and community mental health trend monitoring.

### Dataset

- Source: Kaggle - "Sentiment Analysis for Mental Health" by Suchintika Sarkar
- URL: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- Local file: `data/Combined_Data.csv`
- Size: 53,043 raw records; 52,681 rows after null removal; 52,543 final modelling rows
- Classes: 7 mental-health status labels

### Setup and Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

TensorFlow is required for the deep-learning section. On native Windows with Python 3.11, TensorFlow 2.16.2 is CPU-only; the final laptop run executes CNN and GRU, while BiLSTM is retained as a slower optional extension.

PyTorch CUDA is used for the transformer section. The local laptop environment has been verified with `torch==2.5.1+cu121` on an NVIDIA GeForce GTX 1660 Ti. Because the GPU has 6 GB VRAM, the final laptop run executes DistilBERT with batch size 4 and gradient accumulation. BERT and RoBERTa remain optional extension runs if time and GPU memory allow.

### How to Run

Run the single combined notebook:

1. `notebooks/CDS6344_Full_Analysis_Notebook.ipynb`

The notebook contains all sections in order: setup, preprocessing, EDA, traditional ML, selectable deep learning, selectable transformers, opinion mining, ABSA-style aspect analysis, consolidated results, and verification.

Final compute-aware execution path:

1. Run the full notebook with deep learning and transformers disabled to verify preprocessing, EDA, traditional ML, opinion mining, ABSA-style outputs, and final checks.
2. Enable and run `CNN` and `GRU` in the deep-learning section.
3. Enable and run `distilbert-base-uncased` in the transformer section.
4. Treat BiLSTM, BERT, and RoBERTa as optional extension experiments because they require additional runtime and GPU memory on the laptop environment.

Equivalent section scripts are available:

```bash
python scripts/run_eda_preprocessing.py
python scripts/train_traditional_ml.py
python scripts/run_opinion_mining.py
python scripts/build_model_comparison_table.py
python scripts/verify_outputs.py
```

### Models Implemented

- Naive Bayes with TF-IDF
- Logistic Regression with TF-IDF and balanced class weights
- Linear SVM with TF-IDF and balanced class weights
- CNN with Keras embeddings - executed deep-learning run
- GRU with Keras embeddings - executed optional deep-learning comparison
- DistilBERT fine-tuning - executed transformer run
- BiLSTM with Keras embeddings - optional extension not executed in the final laptop run
- BERT fine-tuning - optional extension not executed in the final laptop run
- RoBERTa fine-tuning - optional extension not executed in the final laptop run

### Results

Generated tables are stored in `results/tables/`. Generated figures are stored in `results/screenshots/`.

Do not copy model performance numbers into the report until the corresponding notebook/script has been executed successfully.

Current local execution status:

- Executed: Naive Bayes, Logistic Regression, Linear SVM, CNN, GRU, and DistilBERT
- Optional compute-constrained extensions not executed in the final laptop run: BiLSTM, BERT, and RoBERTa
- Reason: the assignment names these model families, but the practical submission path demonstrates traditional ML, deep learning, and transformer modelling without launching every expensive extension model on a 6 GB laptop GPU/native Windows CPU runtime.

### Data Visualizations

The repository generates:

- Class distribution chart
- Word clouds by class
- Text length distribution
- Traditional ML confusion matrices
- Traditional ML macro-F1 comparison
- VADER sentiment distribution
- Top opinion targets per class
- Aspect mention heatmap

### Future Work

Future work should extend the project with larger and more diverse social-media datasets from platforms such as Instagram, TikTok comments, and YouTube comments; longitudinal tracking of users over time to detect deteriorating mental-health trends; explainability methods such as SHAP or LIME for clinical trust; a real-time Streamlit or REST API deployment; multilingual modelling for Bahasa Malaysia and Arabic posts using XLM-RoBERTa; clinical validation with mental-health professionals; and fine-grained ABSA using span-level models such as BART or T5.

### Group Members

- Abdulrahman Baidaq - 241UC240L7 - TT1L - Individual submission

### Instructor and Tutor

- Shadab Khan (shadab.khan@mmu.edu.my)
