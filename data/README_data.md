# Dataset

Dataset: Kaggle, "Sentiment Analysis for Mental Health" by Suchintika Sarkar.

Local file: `data/Combined_Data.csv`.

The raw CSV contains 53,043 rows and three columns:

- `Unnamed: 0`: source index column, ignored during modelling
- `statement`: social-media/user-generated text
- `status`: mental-health status label

Verified preprocessing counts:

- Raw rows: 53,043
- Null `statement` rows removed: 362
- Rows after null removal: 52,681
- Rows that become empty after planned cleaning: 138
- Final modelling rows: 52,543

The final task is seven-class text classification:

- Normal
- Depression
- Suicidal
- Anxiety
- Bipolar
- Stress
- Personality disorder
