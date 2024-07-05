#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:03:43 2024

@author: rohan1809
"""
import spacy
import os

nlp = spacy.load("en_core_web_sm")

base_path = '/Users/rohan1809/Desktop/EAAI25-main/Server'

with open(os.path.join(base_path, 'adjectives.txt'), 'r') as file:
    adjectives = file.read().splitlines()

with open(os.path.join(base_path, 'nouns.txt'), 'r') as file:
    nouns = file.read().splitlines()

sentiment_dict = {
    # Manually assigned sentiment values for some adjectives
    "abandoned": -2, "able": 1, "absolute": 1, "adorable": 3, "adventurous": 2,
    "academic": 1, "acceptable": 1, "acclaimed": 2, "accomplished": 2, "accurate": 1,
    "aching": -1, "acidic": -1, "acrobatic": 1, "active": 1, "actual": 0,
    "adept": 2, "admirable": 2, "admired": 2, "adolescent": 0, "adorable": 3,
    "adored": 2, "advanced": 2, "afraid": -2, "affectionate": 2, "aged": 0,
    "aggravating": -2, "aggressive": -1, "agile": 1, "agitated": -1, "agonizing": -3,
    "agreeable": 2, "ajar": 0, "alarmed": -2, "alarming": -2, "alert": 1,
    "alienated": -2, "alive": 2, "all": 0, "altruistic": 2, "amazing": 3,
    "ambitious": 2, "ample": 1, "amused": 2, "amusing": 2, "anchored": 1,
    "ancient": 0, "angelic": 2, "angry": -3, "anguished": -3, "animated": 1,
    # Manually assigned sentiment values for some nouns
    "abandonment": -2, "ability": 1, "absence": -1, "abundance": 2, "academy": 1,
    "acceptance": 2, "access": 1, "accident": -2, "accomplishment": 2, "achievement": 2,
    "action": 1, "activity": 1, "actor": 1, "adventure": 2, "adversity": -2,
    "advice": 1, "affection": 2, "age": 0, "aggression": -2, "agony": -3,
    "alertness": 1, "alienation": -2, "amazement": 3, "ambition": 2, "amount": 0,
    "amusement": 2, "anchor": 1, "anger": -3, "anguish": -3, "animation": 1,
    # Add more words here...
}

# Adding remaining adjectives and nouns to the dictionary with neutral scores
for word in adjectives + nouns:
    if word not in sentiment_dict:
        sentiment_dict[word] = 0  # Neutral words

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

# Example 
text = "alive academy angelic"
tokens = tokenize_and_tag(text)
sentiment_score = calculate_sentiment(tokens)
print("Tokens:", tokens)
print("Sentiment Score:", sentiment_score)



