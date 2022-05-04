FROM python:3.9

WORKDIR /app

RUN pip3 install requests streamlit matplotlib seaborn

COPY . .

CMD ["streamlit", "run", "main.py"]