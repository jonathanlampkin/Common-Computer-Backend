FROM python:3.9

WORKDIR /app

RUN pip3 install pandas requests streamlit

COPY . .

CMD ["streamlit", "run", "main.py"]