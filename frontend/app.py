import streamlit as st
import requests

def send_request(text):
    try:
        api_url = 'https://main-common-computer-jonathanlampkin.endpoint.ainize.ai/summarize'
        files = {'search_phrase': (None, text)}
        response_send_request = requests.post(api_url, files=files)
        status_code_send_request = response.status_code
        return status_code_send_request, response_send_request
    except:
        return "error send request"


st.title("People's Thoughts Demo")
st.header("Generate Twitter Summary and Sentiment")

#length_slider = st.sidebar.slider("Length", 0, 300)
try:
    search_phrase_input = st.text_input("Type Search Phrase", "\"Will Smith\"")
    if st.button("Submit"):
        status_code, response = send_request(search_phrase_input)
        if status_code == 200:
            prediction = response.json()
            st.success(prediction["prediction"])
        else:
            st.error(str(status_code) + "Error")
except:
    print("search_phrase_input section failed")

try:
    st.markdown('''
    <div style="display:flex">
            <a target="_blank" href="https://ainize.ai/scy6500/ainize-tutorial-front?branch=main">
                <img src="https://i.imgur.com/UnJzwth.png"/>
            </a>
            <a style="margin-left:10px" target="_blank" href="https://github.com/scy6500/ainize-tutorial-front">
                <img src="https://i.imgur.com/ASkTsnj.png"/>
            </a>
    <div>
        ''',
        unsafe_allow_html=True
    )
except Exception as e:
    print("markdown section failed")