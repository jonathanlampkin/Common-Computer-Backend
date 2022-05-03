#from nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM python:3.9
#FROM jonathanlampkin/peoplesthoughts:10

# -----------------------------------------------------------------------------------------------------------------
WORKDIR /app

RUN pip install streamlit requests Pillow
# --------------------------------------------------------------------------------------------------------------------
#EXPOSE 8000

# ---------------------------------------------------------------------------------------------------------------------
# Pulling from Docker
COPY . .

# Pushing to Docker
# COPY ./sentiment_model /app/sentiment_model/
# COPY ./summarizer_model /app/summarizer_model/
# COPY ./bearer_token_folder /app/bearer_token_folder
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#RUN pip3 install matplotlib==3.5.1 numpy==1.21.6 python-dotenv==0.20.0 requests==2.27.1 Flask==2.1.1 seaborn==0.11.2 transformers==4.17.0 torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
#RUN pip3 install matplotlib numpy python-dotenv==0.20.0 requests Flask seaborn transformers==4.17.0 streamlit torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html


# -----------------------------------------------------------------------------------------------------------------------------------------------
CMD ["streamlit", "run", "main.py"]

#CMD ["python3", "./main.py"]
#CMD ["python3", "./main.py", "streamlit", "run", "./main.py"]