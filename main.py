import streamlit as st
import requests
import base64
from PIL import Image

def send_request(text):
    api_url = 'https://main-common-computer-backend-jonathanlampkin.endpoint.ainize.ai/summarize'
    files = {'image': text}
    response = requests.post(api_url, files=files)
    status_code = response.status_code

    return status_code, response


st.title("People's Thoughts Demo")
st.header("Generate Twitter Summary and Sentiment")

base_story = st.text_input("Type Base Story", "\"I love him.\"")
if st.button("Submit"):
    status_code, response = send_request(base_story)
    if status_code == 200:
        #prediction = response.json()
        #st.success(prediction["prediction"])
        image = Image.open(response)
        st.image(image, caption='Sentiment Scores')
    else:
        st.error(str(status_code) + " Error")



