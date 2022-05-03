FROM jonathanlampkin / peoplesthoughts: 10

WORKDIR / app

RUN pip3 install numpy python-dotenv==0.20.0 requests flask transformers==4.17.0 torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

COPY ..

CMD["python3", "main.py"]