# Default python packages
import json
import os
import re

# Pip installed python packages
from flask import Flask, request, jsonify
import numpy as np
import requests
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

if torch.cuda.is_available():
    device = 0
else:
    device = 1

batch_size = 5
search_url = 'https://api.twitter.com/2/tweets/search/recent'

summarizer_path = './summarizer_model'
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_path, padding=True, truncation=True)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path)
summarizer = pipeline('summarization', model=summarizer_model, tokenizer=summarizer_tokenizer, device=device)

classifier_path = './sentiment_model'
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path, padding=True, truncation=True)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
classifier = pipeline('sentiment-analysis', model=classifier_model, tokenizer=classifier_tokenizer, return_all_scores=True, device=device)

bearer_token = os.environ['bearer_token']

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers['Authorization'] = f"Bearer {bearer_token}"
    r.headers['User-Agent'] = 'v2RecentSearchPython'
    return r


def connect_to_endpoint(query):
    try:
        print('Pulling Tweets...')
        url = 'https://api.twitter.com/2/tweets/search/recent'
        query = f"-is:retweet lang:en -has:links -has:media {query}"
        query_params = {'query': query, 'max_results': 100}
        response = requests.get(url, auth=bearer_oauth, params=query_params)
        data = []
        data.append([tweet['text'] for tweet in response.json()['data']])
        return data
    except Exception as e:
        return jsonify({"issue pulling from twitter": e}), 500


def apply_regexes(tweet):
    """Adding this to further clean and not lose data"""
    text_cleaning_re = '(?:amp;|@|http|^rt|#)([^ ]+|\$)'
    tweet = re.sub(text_cleaning_re, ' ', tweet)
    tweet = re.sub(' +', ' ', tweet)
    return tweet.strip()


def clean_tweets(data):
    """Cleans pulled tweets"""
    tweets = list(dict.fromkeys(sum(data, [])))
    tweets = [apply_regexes(tweet) for tweet in tweets]
    return list(filter(None, tweets))


def chunk(tweets):
    """Simpler/Faster chunking method that allows you to set the num chunks equal to batch size"""
    return [' '.join(items) for items in np.array_split(tweets, batch_size)]


def summarize(chunks):
    """Summarizes the chunks one by one, joins the summaries and summarizes it again."""
    print('Summarizing Tweets...')
    try:
        chunked_summaries = summarizer(chunks, batch_size=batch_size, truncation=True, do_sample=True, top_k=0, typical_p=0.7, early_stopping=True)
        combined_summaries = ' '.join([item['summary_text'] for item in chunked_summaries])
        return summarizer(combined_summaries, max_length=200, do_sample=True, typical_p=0.7, top_k=0, early_stopping=True)[0]['summary_text']
    except Exception as e:
        return jsonify({"summarizer error": e}), 500


def sentiment(chunks):
    """Plots sentiment scores."""
    print('Calculating Sentiment...')
    try:
        sentiments = classifier(' '.join(chunks), truncation=True, padding=True)[0]
        sentiment_scores = dict(sorted({dictionary['label']: dictionary['score'] for dictionary in sentiments}.items(), key=lambda x: x[1], reverse=True))
        return sentiment_scores
    except Exception as e:
        return jsonify({"classifier error": e}), 500


def make_story(base_text):
    result = dict()
    try:
        json_response = connect_to_endpoint(base_text)
        tweets = clean_tweets(json_response)
        chunks = chunk(tweets)
    except Exception as e:
        return jsonify({'twitter pull or data processing error': e}), 500
    try:
        result['prediction'] = summarize(chunks)
        result['sentiment'] = sentiment(chunks)
        return jsonify(result)
    except Exception as e:
        return jsonify({'summarizer or classification error': e}), 500


@app.route("/summarize", methods=["POST"])
def main():
    try:
        base_text = request.form.get('base_text')
    except Exception as e:
        return jsonify({'did not pull from backend':e}), 500
    summary_and_sentiment = make_story(base_text)
    return summary_and_sentiment


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)