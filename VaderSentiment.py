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
text = "The brave wolf is ugly."
tokens = tokenize_and_tag(text)
sentiment_scores = calculate_sentiment(text)

print("Tokens:", tokens)
print("Sentiment Scores:", sentiment_scores)

# Additional sentences to analyze
sentence1 = "The brave wolf is handsome."
sentence2 = "The brave wolf is extremely handsome."
sentence3 = "The brave wolf is not handsome."

# Calculate sentiment scores for additional sentences
score1 = calculate_sentiment(sentence1)
score2 = calculate_sentiment(sentence2)
score3 = calculate_sentiment(sentence3)

print(f"Sentiment scores for '{sentence1}': {score1}")
print(f"Sentiment scores for '{sentence2}': {score2}")
print(f"Sentiment scores for '{sentence3}': {score3}")
