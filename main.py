import streamlit as st
import requests

def send_request(text):
    api_url = 'https://main-common-computer-backend-jonathanlampkin.endpoint.ainize.ai/summarize'
    files = {
        'base_text': (None, text)
    }
    response = requests.post(api_url, files=files)
    status_code = response.status_code

    return status_code, response


st.title("People's Thoughts Demo")
st.header("Generate Twitter Summary and Sentiment")

base_story = st.text_input("Type Base Story", "\"I love him. He's not proud. I was wrong. I was entirely wrong about him.\"")
if st.button("Submit"):
    status_code, response = send_request(base_story)
    if status_code == 200:
        prediction = response.json()
        st.success(prediction["prediction"])
    else:
        st.error(str(status_code) + " Error")

# st.markdown('''
# <div style="display:flex">
#         <a target="_blank" href="https://ainize.ai/jonathanlampkin/Common-Computer?branch=main">
#             <img src="https://i.imgur.com/UnJzwth.png"/>
#         </a>
#         <a style="margin-left:10px" target="_blank" href="https://github.com/jonathanlampkin/Common-Computer">
#             <img src="https://i.imgur.com/ASkTsnj.png"/>
#         </a>
# <div>
#     ''',
#     unsafe_allow_html=True
# )