#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:13:51 2024

@author: rohan1809
"""
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Initialize SpaCy and VADER
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()

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

# Define lists for intensifiers and negations
intensifiers = {
    "very": 1.5,
    "extremely": 2.0,
    "absolutely": 2.0,
    "incredibly": 2.0,
    "quite": 1.2,
    "really": 1.5
}
negations = ["not", "never", "no", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely"]

def tokenize_and_tag(text):
    doc = nlp(text)
    tokens = [(token.text, token.pos_) for token in doc]
    return tokens

def calculate_sentiment(tokens):
    sentiment_score = 0
    prev_word = ""
    intensifier_multiplier = 1.0

    for word, pos in tokens:
        word_lower = word.lower()
        
        if word_lower in intensifiers:
            intensifier_multiplier = intensifiers[word_lower]
            continue
        if word_lower in negations:
            intensifier_multiplier = -1.0
            continue

        word_sentiment = sentiment_dict.get(word_lower, 0)
        sentiment_score += intensifier_multiplier * word_sentiment
        intensifier_multiplier = 1.0  # Reset the multiplier after applying it
        prev_word = word

    return sentiment_score

# Example usage
text1 = "The cool wolf is handsome."
text2 = "The cool wolf is extremely handsome."

tokens1 = tokenize_and_tag(text1)
tokens2 = tokenize_and_tag(text2)

sentiment_score1 = calculate_sentiment(tokens1)
sentiment_score2 = calculate_sentiment(tokens2)

print("Tokens 1:", tokens1)
print("Sentiment Score 1:", sentiment_score1)

print("Tokens 2:", tokens2)
print("Sentiment Score 2:", sentiment_score2)
