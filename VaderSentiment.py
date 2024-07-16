#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:13:51 2024

@author: rohan1809
"""
import spacy
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Define the path to the text files
base_path = '/Users/rohan1809/Desktop/EAAI25-main/Server'

# Load adjectives and nouns from text files in the base path
with open(os.path.join(base_path, 'adjectives.txt'), 'r') as file:
    adjectives = file.read().splitlines()

with open(os.path.join(base_path, 'nouns.txt'), 'r') as file:
    nouns = file.read().splitlines()

# Create a comprehensive sentiment dictionary using VADER
sentiment_dict = {}
for word in adjectives + nouns:
    ss = sid.polarity_scores(word)
    sentiment_dict[word] = ss['compound']

def tokenize_and_tag(text):
    doc = nlp(text)
    tokens = [(token.text, token.pos_) for token in doc]
    return tokens

def calculate_sentiment(text):
    # Use VADER to calculate sentiment
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Example usage
sentences = [
    "The brave wolf is handsome.",
    "The brave wolf is extremely handsome.",
    "The brave wolf is not handsome.",
    "The cake is very delicious.",
    "The weather is not good.",
    "She is quite amazing.",
    "He never goes there."
]

for sentence in sentences:
    tokens = tokenize_and_tag(sentence)
    sentiment_scores = calculate_sentiment(sentence)
    print(f"Tokens for '{sentence}':", tokens)
    print(f"Sentiment scores for '{sentence}':", sentiment_scores)
