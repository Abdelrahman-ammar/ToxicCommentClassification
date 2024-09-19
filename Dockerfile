FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 9000

CMD python app.py