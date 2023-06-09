import re
import json
import spacy
import datetime
import datefinder
import ir_datasets
import pandas as pd
import os
from autocorrect import Speller
from nltk.tokenize import word_tokenize

def get_datasets_and_convert_to_csv_files():
    # Read Quora Dataset
    dataset = ir_datasets.load("beir/quora/dev")

    # Load To DataFrame
    queries = pd.DataFrame(dataset.queries_iter())
    docs = pd.DataFrame(dataset.docs)
    qrels = pd.DataFrame(dataset.qrels_iter())

    # Convert TO CSV
    queries.to_csv(os.path.join(os.getcwd(), "resources", "original_queries.csv"), index=False)
    docs.to_csv(os.path.join(os.getcwd(), "resources", "original_docs.csv"), index=False)
    qrels.to_csv(os.path.join(os.getcwd(), "resources", "original_qrels.csv"), index=False)


# Read Files
with open(os.path.join(os.getcwd(), "resources", "abbreviations.json")) as json_file:
    data = json.load(json_file)

original_queries = pd.read_csv(os.path.join(os.getcwd(), "resources", "original_queries.csv"))
original_docs = pd.read_csv(os.path.join(os.getcwd(), "resources", "original_docs.csv"))

# Initialize Important Variables
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
spell_correction = Speller(lang='en')


# Data Processing Functions

def date_processing(text):
    text = text.replace('_', ' ')
    matches = datefinder.find_dates(text, source=True, strict=True)
    for match in matches:
        converted_token = match[0].strftime("%Y_%m_%d")
        text = text.replace(match[1], converted_token)
    return text


def lemmatize_text(text):
    result = []
    for word in word_tokenize(text):
        doc = nlp(word)
        lemmas = [token.lemma_ for token in doc if not token.is_stop]
        if len(lemmas) > 0:
            result.append(lemmas[0])
    return ' '.join(result)


def abbreviations_processing(text):
    result = []
    for token in text.split():
        if token in data.keys():
            result.append(data.get(token))
        else:
            result.append(token)
    return ' '.join(result)


def data_processing(data_frame_texts):
    docs = []
    for text in data_frame_texts:

        # If Text Contains Numbers Only
        text = str(text)

        # Date Processing
        text = date_processing(text)

        # Abbreviations Processing
        text = abbreviations_processing(text)

        # To lower case
        text = text.lower()

        # Remove Punctuation
        text = re.sub(r'[^_\w\s]', '', text)

        # Lemmatize text using spaCy
        if text != '':
            text = lemmatize_text(text)

        # Remove non-alphanumeric Characters
        text = re.compile('[^_a-zA-Z0-9\s]').sub('', str(text))

        # Spell Correction
        text = spell_correction(text)

        # Add updated text
        docs.append(text)
    return docs


def queries_processing():
    time_before_run = datetime.datetime.now()

    original_queries['text'] = data_processing(original_queries['text'])
    original_queries.to_csv(os.path.join(os.getcwd(), "resources", "clean_queries.csv"), index=True, index_label='id')

    time_after_run = datetime.datetime.now()
    print("End with: " + str(time_after_run - time_before_run))


def docs_processing():
    time_before_run = datetime.datetime.now()

    original_docs['text'] = data_processing(original_docs['text'])
    original_docs.to_csv(os.path.join(os.getcwd(), "resources", "clean_docs.csv"), index=True, index_label='id')

    time_after_run = datetime.datetime.now()
    print("End with: " + str(time_after_run - time_before_run))
