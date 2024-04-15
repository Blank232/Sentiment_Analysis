import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob


class TwitterClient(object):

    def __init__(self):

        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_token_secret = ''

        try:

            self.auth = OAuthHandler(consumer_key, consumer_secret)

            self.auth.set_access_token(access_token, access_token_secret)

            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):

        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q=query, count=count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)

                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def textblob_sentiment(self, tweet):

        analysis = TextBlob(self.clean_tweet(tweet))
        print("Polarity of the tweet: ", analysis.sentiment.polarity)
        print("subjectivity of the tweet: ", analysis.sentiment.subjectivity)
        print(" Tweet rated by Textblob as:", end=" ")
        if analysis.sentiment.polarity > 0:
            print('positive')
        elif analysis.sentiment.polarity == 0:
            print('neutral')
        else:
            print('negative')


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    que = "Farmers law"  # the query to be searched in twitter.
    tweets = api.get_tweets(query=que, count=10)
    print("----------------------------------------------------------------------------------------------------------")
    print("                                    TextBlob sentiment analysis                                           ")
    print("----------------------------------------------------------------------------------------------------------")

    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    # percentage of neutral tweets
    print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)))

    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:2]:
        print(tweet['text'])

    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:2]:
        print(tweet['text'])
    print("----------------------------------------------------------------------------------------------------------")
    print("                                Textblob sentiment analysis showcase                                      ")
    print("----------------------------------------------------------------------------------------------------------")

    try:
        print(str(ptweets[0]))
        print("\n")
        api.textblob_sentiment(str(ptweets[0]))
        print("\n")
    except:
        print("no positive tweets according to Textblob \n")

    try:
        print(str(ntweets[0]))
        print("\n")
        api.textblob_sentiment(str(ntweets[0]))
        print("\n")
    except:
        print("no negative tweets according to Textblob \n")


if __name__ == "__main__":
    # calling main function
    main()