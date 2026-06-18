# Mental Health Status Classification from Social Media Text
## CDS6344 Social Media Computing - Individual Assignment Submission

### Project Overview

This repository contains the implementation for the CDS6344 Social Media Computing assignment titled "Mental Health Status Classification from Social Media Text: A Multi-Class Sentiment Analysis and Opinion Mining Pipeline."

The project builds a reproducible NLP pipeline for classifying social-media/user-generated text into seven mental-health-related status classes: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, and Personality disorder. It includes verified dataset preprocessing, exploratory analysis, traditional machine-learning baselines, opinion mining, ABSA-style aspect analysis, and notebooks for deep-learning and transformer experiments.

### Problem Statement

Millions of individuals express mental health struggles, including depression, anxiety, and suicidal ideation, on platforms like Twitter and Reddit. Traditional clinical systems are too slow and costly for early detection at scale. This project builds a multi-class sentiment analysis and opinion mining pipeline to classify social media posts by mental health status, enabling early detection of at-risk individuals and community mental health trend monitoring.

### Dataset

- Source: Kaggle - "Sentiment Analysis for Mental Health" by Suchintika Sarkar
- URL: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- Local file: `data/Combined_Data.csv`
- Size: 53,043 raw records; 52,681 rows after null removal; 52,542 final modelling rows
- Classes: 7 mental-health status labels

### Setup and Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

TensorFlow is required for the BiLSTM, CNN, and GRU notebook. If TensorFlow is not available locally, run that notebook in Google Colab.

### How to Run

Run the notebooks in order:

1. `notebooks/01_EDA_and_Preprocessing.ipynb`
2. `notebooks/02_Traditional_ML_Models.ipynb`
3. `notebooks/03_Deep_Learning_Models.ipynb`
4. `notebooks/04_Transformer_Models.ipynb`
5. `notebooks/05_Opinion_Mining_ABSA.ipynb`
6. `notebooks/06_Visualization_and_Results.ipynb`

Equivalent section scripts are available:

```bash
python scripts/run_eda_preprocessing.py
python scripts/train_traditional_ml.py
python scripts/run_opinion_mining.py
```

### Models Implemented

- Naive Bayes with TF-IDF
- Logistic Regression with TF-IDF and balanced class weights
- Linear SVM with TF-IDF and balanced class weights
- BiLSTM with GloVe embeddings
- CNN with GloVe embeddings
- GRU with GloVe embeddings
- BERT fine-tuning
- DistilBERT fine-tuning
- RoBERTa fine-tuning

### Results

Generated tables are stored in `results/tables/`. Generated figures are stored in `results/screenshots/`.

Do not copy model performance numbers into the report until the corresponding notebook/script has been executed successfully.

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
