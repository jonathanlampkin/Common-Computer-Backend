import streamlit as st
import requests
import pandas as pd

def send_request(text):
    api_url = 'https://main-common-computer-backend-jonathanlampkin.endpoint.ainize.ai/summarize'
    files = {'base_text': (None, text)}
    response = requests.post(api_url, files=files)
    status_code = response.status_code
    return status_code, response


st.title("People's Thoughts Demo")
st.header("Generate Twitter Summary and Sentiment")

base_story = st.text_input("Type Search Phrase", "\"Johnny Depp\"")
if st.button("Submit"):
    status_code, response = send_request(base_story)
    if status_code == 200:
        prediction = response.json()
        st.success(prediction["prediction"])
        st.bar_chart(pd.DataFrame.from_dict(prediction['sentiment']))
    else:
        st.error(str(status_code) + " Error")