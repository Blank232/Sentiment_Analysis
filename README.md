# Twitter Sentiment Analysis

This repository contains a Python script that performs sentiment analysis on tweets using three different models: TextBlob, VADER, and RoBERTa. The script fetches recent tweets based on a given query and evaluates their sentiment using these models, comparing their performance with accuracy, precision, recall, and F1-score metrics.

## Features
- Fetches recent tweets using Twitter API v2.
- Performs sentiment analysis using:
  - **TextBlob** (Lexicon-based approach)
  - **VADER** (Rule-based approach for social media text)
  - **RoBERTa** (Pre-trained transformer model for sentiment analysis)
- Compares the models using standard evaluation metrics.

## Requirements
Before running the script, ensure you have the following dependencies installed:

```bash
pip install requests textblob vaderSentiment transformers torch scikit-learn
```

## Usage

1. **Get Twitter API Credentials:**
   - Sign up for a Twitter Developer account.
   - Generate a Bearer Token from the Twitter Developer Portal.

2. **Set Up the Script:**
   - Replace `BEARER_TOKEN` in the script with your actual Twitter Bearer Token.

3. **Run the Script:**
   ```bash
   python sentiment_analysis.py
   ```

4. **Output:**
   - Displays tweets fetched from Twitter.
   - Provides sentiment analysis results from all three models.
   - Computes accuracy, precision, recall, and F1-score for model comparison.

## Example Output
```
Tweet 1: "The political scenario in Maharashtra is unpredictable."
  TextBlob Sentiment: Neutral
  VADER Sentiment: Negative
  RoBERTa Sentiment: Negative
--------------------------------------------------

TextBlob Metrics:
  Accuracy: 0.70
  Precision: 0.72
  Recall: 0.70
  F1 Score: 0.71
--------------------------------------------------
```

## Performance Evaluation
The script uses a set of predefined ground truth labels for comparison. The models' predictions are evaluated based on standard classification metrics to determine the most effective approach for sentiment analysis.

## Notes
- **Rate Limits:** Twitter API has rate limits; if exceeded, the script waits before retrying.
- **Predefined Ground Truth Labels:** Used for evaluation purposes, replace with actual labels if available.
- **RoBERTa Model:** Requires `transformers` and `torch` libraries.

## License
This project is licensed under the MIT License.

## Author
Ishir Srivats

