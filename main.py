# Default python packages
import base64
import re
# import json
# import os

# Pip installed python packages

from flask import Flask, request, jsonify
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import requests

#from dotenv import load_dotenv

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
summarizer = pipeline('summarization', model=summarizer_model,tokenizer=summarizer_tokenizer, device=device)

classifier_path = './sentiment_model'
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path, padding=True, truncation=True)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
classifier = pipeline('sentiment-analysis', model=classifier_model,tokenizer=classifier_tokenizer, return_all_scores=True, device=device)

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers['Authorization'] = f"Bearer AAAAAAAAAAAAAAAAAAAAAPMFYwEAAAAAq2FFaFpxS5QsGI%2F01SZf%2BgMjrBU%3DGP2vV7JDr5jrlWLcVcbD6Syb68yl1IwHovIH61t4d629ZjIqHj"
    r.headers['User-Agent'] = 'v2RecentSearchPython'
    return r

def connect_to_endpoint(query):
    try:
        print('Pulling Tweets...')
        url = 'https://api.twitter.com/2/tweets/search/recent'
        query = f"-is:retweet lang:en -has:links -has:media {query}"
        query_params = {'query': query, 'max_results': 100}#, 'tweet.fields': 'public_metrics'}
        response = requests.get(url, auth=bearer_oauth, params=query_params)
        data = []
        data.append([tweet['text'] for tweet in response.json()['data']])
        return data
    except:
        return "issue pulling from twitter"


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
        chunked_summaries = summarizer(chunks, batch_size=batch_size, truncation=True, do_sample=True, top_k=0, typical_p=0.7,early_stopping=True)
        combined_summaries = ' '.join([item['summary_text'] for item in chunked_summaries])
        result = dict()
        result['prediction'] = summarizer(combined_summaries, max_length=200, do_sample=True, typical_p=0.7, top_k=0,early_stopping=True)[0]['summary_text']
        return jsonify(result)
    except:
        return "pipeline didnt work"


def sentiment(chunks):
    """Plots sentiment scores."""
    print('Calculating Sentiment...')
    sentiments = classifier(' '.join(chunks), truncation=True, padding=True)[0]
    sentiment_scores = dict(
        sorted({dictionary['label']: dictionary['score'] for dictionary in sentiments}.items(), key=lambda x: x[1],
               reverse=True))
    fig = plt.figure()
    sns.barplot(x=list(sentiment_scores.keys()), y=list(sentiment_scores.values())).set(title='Sentiment Scores')
    chart = dict()
    chart['chart'] = base64.b64encode(fig)
    return jsonify(chart) #may not be able to jsonify this


def make_story(base_text):
    try:
        json_response = connect_to_endpoint(base_text)
        tweets = clean_tweets(json_response)
        chunks = chunk(tweets)
    except Exception as e:
        print('Error pulling tweets', e)
        return jsonify({'error': e}), 500

    try:
        summary, chart = summarize(chunks), sentiment(chunks)
        return summary
    except Exception as e:
        print('Summarizer failed', e)
        return jsonify({'error': e}), 500


@app.route("/summarize", methods=["GET"])
def main():
    try:
        base_text = request.form.get('base_text')
    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    summary = make_story(base_text)

    return summary

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)