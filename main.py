
__authors__ = ['Jonathan Lampkin', 'Jeff Yarbro']
__email__ = 'jmlampkin@gmail.com'

# Default python packages
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
#import streamlit as st

app = Flask(__name__)

if torch.cuda.is_available():
    device = 0
else:
    device = 1

batch_size = 5
search_url = 'https://api.twitter.com/2/tweets/search/recent'

summarizer_path = './summarizer_model'
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_path, padding=True, truncation=True)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path)#.to(device)
summarizer = pipeline('summarization', model=summarizer_model,tokenizer=summarizer_tokenizer, device=device)
#
# classifier_path = './sentiment_model'
# classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path, padding=True, truncation=True)
# classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
# classifier = pipeline('sentiment-analysis', model=classifier_model,tokenizer=classifier_tokenizer, return_all_scores=True)#device=0)

# try both
#bearer_token = './bearer_token_folder/bearer_token.txt' #try call on the file
#bearer_auth = './authorizer/authorizer.py'

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers['Authorization'] = f"Bearer AAAAAAAAAAAAAAAAAAAAAPMFYwEAAAAAq2FFaFpxS5QsGI%2F01SZf%2BgMjrBU%3DGP2vV7JDr5jrlWLcVcbD6Syb68yl1IwHovIH61t4d629ZjIqHj"
    r.headers['User-Agent'] = 'v2RecentSearchPython'
    return r

def connect_to_endpoint(query: str):
    try:
        print('Pulling Tweets...')
        url = 'https://api.twitter.com/2/tweets/search/recent'
        query = f"-is:retweet lang:en -has:links -has:media {query}"
        query_params = {'query': query, 'max_results': 100}#, 'tweet.fields': 'public_metrics'}
        response_query = requests.get(url, auth=bearer_oauth, params=query_params)
    except:
        return "issue pulling from twitter"
    try:
        data = []
        data.append([tweet['text'] for tweet in response_query.json()['data']])
    except Exception as e:
        return "issue joining tweets"
    return data

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
        result['prediction'] = summarizer(combined_summaries, max_length=200, do_sample=True, typical_p=0.7, top_k=0,early_stopping=True)
        return jsonify(result)
    except:
        return "pipeline didnt work"
    # try:
    #     input_ids = [summarizer_tokenizer.encode(chunk, return_tensors='pt') for chunk in chunks]
    #     #input_ids = summarizer_tokenizer.encode(chunks[0], return_tensors='pt')
    # except:
    #     return "tokenizer didnt encode"
    # try:
    #     outputs = [summarizer_model.generate(input_id) for input_id in input_ids]
    #     #outputs = summarizer_model.generate(input_ids)
    # except:
    #     return "summarizer didnt generate"
    # try:
    #     first_summary = [summarizer_tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs]
    #     #result['prediction'] = summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     #return jsonify(result)
    # except:
    #     "summarizer didnt decode"
    #     try:
    #         input_ids = [summarizer_tokenizer.encode(chunk, return_tensors='pt') for chunk in chunks]
    #         # input_ids = summarizer_tokenizer.encode(chunks[0], return_tensors='pt')
    #     except:
    #         return "tokenizer didnt encode"
    #     try:
    #         outputs = [summarizer_model.generate(input_id) for input_id in input_ids]
    #         # outputs = summarizer_model.generate(input_ids)
    #     except:
    #         return "summarizer didnt generate"
    #     try:
    #         result = dict()
    #         result["prediction"] = [summarizer_tokenizer.decode(output[0], skip_special_tokens=True) for output in
    #                                 outputs]

        #chunked_summaries = summarizer(chunks, batch_size=batch_size, truncation=True, do_sample=True, top_k=0, typical_p=0.7, early_stopping=True)
        #combined_summaries = ' '.join([item['summary_text'] for item in chunked_summaries])

    #return jsonify(summarizer(combined_summaries, max_length=200, do_sample=True, typical_p=0.7, top_k=0, early_stopping=True))

# def sentiment(chunks):
#     """Plots sentiment scores."""
#     print('Calculating Sentiment...')
#     sentiments = classifier(' '.join(chunks), truncation=True, padding=True)[0]
#     sentiment_scores = dict(
#         sorted({dictionary['label']: dictionary['score'] for dictionary in sentiments}.items(), key=lambda x: x[1],
#                reverse=True))
#     fig = plt.figure()
#     sns.barplot(x=list(sentiment_scores.keys()), y=list(sentiment_scores.values())).set(title='Sentiment Scores')
#     return fig


# @app.route("/summarize", methods=["GET"]) # try if get: if post:
# def main():
#     try:
#         json_response = connect_to_endpoint('will smith')
#     except:
#         return "twitter connection fail"
#     try:
#         tweets = clean_tweets(json_response)
#     except:
#         return "clean tweets failed"
#     try:
#         chunks = chunk(tweets)
#     except:
#         return "failed to chunk"
#     try:
#         result = summarize(chunks)
#         return result
#     except Exception as e:
#         return "failed to summarize tweets"
######################################################################################################


def summarize_and_classify(search_phrase):
    try:
        json_response = connect_to_endpoint(search_phrase)
        tweets = clean_tweets(json_response)
        chunks = chunk(tweets)
    except Exception as e:
        print('Error pulling tweets', e)
        return jsonify({'error': e}), 500

    try:
        result = summarize(chunks)
        return result
    except Exception as e:
        print('Summarizer failed', e)
        return jsonify({'error': e}), 500


@app.route("/summarize", methods=["POST"])
def main():
    try:
        search_phrase = request.form.get('search_phrase')

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    try:
        prediction = summarize_and_classify(search_phrase)
        return prediction
    except:
        return "summarize and classify failed"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)