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

def calculate_sentiment(tokens):
    sentiment_score = 0
    for word, pos in tokens:
        if word.lower() in sentiment_dict:
            sentiment_score += sentiment_dict[word.lower()]
    return sentiment_score

# Example usage
text = "The brave wolf is very beautiful."
tokens = tokenize_and_tag(text)
sentiment_score = calculate_sentiment(tokens)
print("Tokens:", tokens)
print("Sentiment Score:", sentiment_score)
