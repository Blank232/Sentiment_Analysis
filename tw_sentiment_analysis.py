import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time


class TwitterClient:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"

        # Initialize RoBERTa model for sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.labels = ["negative", "neutral", "positive"]

    def create_headers(self):
        """Create headers for the API request using the Bearer Token."""
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def search_recent_tweets(self, query, max_results=10):
        
        search_url = f"{self.base_url}/tweets/search/recent"
        headers = self.create_headers()
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "text",
        }

        retry_attempts = 3  # Number of retries for rate limiting
        for attempt in range(retry_attempts):
            response = requests.get(search_url, headers=headers, params=params)

            if response.status_code == 200:
                tweets = response.json().get("data", [])
                return [tweet["text"] for tweet in tweets]
            elif response.status_code == 429:
                # Rate limit reached; wait and retry
                reset_time = response.headers.get("x-rate-limit-reset")
                if reset_time:
                    wait_time = max(0, int(reset_time) - int(time.time()))
                else:
                    wait_time = 15  # Default wait time if header not available
                print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retry_attempts})")
                time.sleep(wait_time)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break

        return []

    def textblob_sentiment(self, tweet):
        """Perform sentiment analysis using TextBlob."""
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity == 0:
            return "neutral"
        else:
            return "negative"

    def vader_sentiment(self, tweet):
        """Perform sentiment analysis using VADER."""
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(tweet)
        if scores['compound'] >= 0.05:
            return "positive"
        elif scores['compound'] <= -0.05:
            return "negative"
        else:
            return "neutral"

    def roberta_sentiment(self, tweet):
        """Perform sentiment analysis using RoBERTa."""
        tokens = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**tokens)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = self.labels[torch.argmax(probabilities).item()]
        return sentiment


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate and display metrics for a model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print("-" * 50)
    return accuracy, precision, recall, f1


def main():
    # Replace 'YOUR_BEARER_TOKEN' with your actual Bearer Token
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAItHxQEAAAAASO6XVfFvzDGYT6lddUXswedqSCo%3DsNS9nnfWUnGVx6YquMPVA7eg3wcYzfUWUOjgaSYPcFy7yh0rRM"

    api = TwitterClient(bearer_token=BEARER_TOKEN)
    query = "What is the current political scenario in maharashtra"  # Example query
    tweets = api.search_recent_tweets(query=query, max_results=10)

    if not tweets:
        print("No tweets found or an error occurred.")
        return

    # Simulated ground truth labels (for demonstration)
    # Replace these with actual labels if available
    ground_truth = ["positive", "negative", "neutral", "neutral", "positive",
                    "negative", "positive", "neutral", "neutral", "negative"]

    # Encode ground truth for scoring
    label_encoder = LabelEncoder()
    ground_truth_encoded = label_encoder.fit_transform(ground_truth)

    # Analyze tweets with TextBlob, VADER, and RoBERTa
    textblob_predictions = []
    vader_predictions = []
    roberta_predictions = []

    print("\nTweets and Sentiment Analysis Results:")
    for i, tweet in enumerate(tweets, 1):
        print(f"Tweet {i}: {tweet}")

        # TextBlob Analysis
        textblob_sentiment = api.textblob_sentiment(tweet)
        textblob_predictions.append(textblob_sentiment)
        print(f"  TextBlob Sentiment: {textblob_sentiment}")

        # VADER Analysis
        vader_sentiment = api.vader_sentiment(tweet)
        vader_predictions.append(vader_sentiment)
        print(f"  VADER Sentiment: {vader_sentiment}")

        # RoBERTa Analysis
        roberta_sentiment = api.roberta_sentiment(tweet)
        roberta_predictions.append(roberta_sentiment)
        print(f"  RoBERTa Sentiment: {roberta_sentiment}")
        print("-" * 80)

    # Encode predictions for scoring
    textblob_encoded = label_encoder.transform(textblob_predictions)
    vader_encoded = label_encoder.transform(vader_predictions)
    roberta_encoded = label_encoder.transform(roberta_predictions)

    # Calculate and display metrics for each model
    calculate_metrics(ground_truth_encoded, textblob_encoded, "TextBlob")
    calculate_metrics(ground_truth_encoded, vader_encoded, "VADER")
    calculate_metrics(ground_truth_encoded, roberta_encoded, "RoBERTa")


if __name__ == "__main__":
    main()
