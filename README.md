# Social Media Text Analytics: Cleaning, Trend Detection & Prediction on Raw Tweets

An end-to-end NLP pipeline on raw, unstructured Twitter data — from text cleaning through
descriptive analytics to predictive modeling. This is the core workflow behind social
listening, brand/sentiment tracking, and marketing analytics: turning messy public text into
structured, decision-useful signal.

## What this shows
- **Text preprocessing:** built a custom tokenization/POS-tagging/lemmatization pipeline
  (`lemmatizer.py`) to clean raw tweet text for downstream analysis.
- **Descriptive analytics:** computed distributional statistics (min/mean/median/max) for
  tweet length and hashtag usage, and identified the ten most common hashtags with their
  usage counts — surfacing what topics were actually driving conversation.
- **Modeling:** used the cleaned text features to train a predictive model, with results
  visualized and interpreted in the accompanying report.

## Repo structure
```
lemmatizer.py         # custom text cleaning / lemmatization pipeline
notebook.ipynb         # descriptive analysis, hashtag trends, modeling
lab_report.pdf          # summary report
report.pdf               # full write-up with methodology and findings
images/                   # key charts (hashtag frequency, distribution plots)
predictions.csv            # model output
```

## Tech stack
Python, NLTK, Pandas, Matplotlib

## Running it
```bash
pip install nltk pandas matplotlib jupyter
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
jupyter notebook notebook.ipynb
```

---
*Originally developed for a graduate data science course (University of Rochester,
DSCC 383W).*
