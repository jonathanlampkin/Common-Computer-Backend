import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def send_request(text):
    api_url = 'https://main-common-computer-backend-jonathanlampkin.endpoint.ainize.ai/summarize'
    files = {'base_text': (None, text)}
    response = requests.post(api_url, files=files)
    status_code = response.status_code
    return status_code, response


st.title("People's Thoughts Demo")
st.header("Generate Twitter Summary and Sentiment")

base_story = st.text_input("Type Search Phrase", "\"Will Smith\"")
if st.button("Submit"):
    status_code, response = send_request(base_story)
    if status_code == 200:
        prediction = response.json()
        st.success(prediction["prediction"])
        fig, ax = plt.subplots()
        sns.barplot(ax=ax, x=list(sentiment_scores.keys()), y=list(sentiment_scores.values())).set(title='Sentiment Scores')
        st.pyplot(fig)
    else:
        st.error(str(status_code) + " Error")