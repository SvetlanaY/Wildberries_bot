FROM python:3.8
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download ru_core_news_lg
COPY . .
CMD python3 telegram.py
