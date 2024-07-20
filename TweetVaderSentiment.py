#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:49:46 2024

@author: rohan1809
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import os

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Define the path to the CSV file on your desktop
csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/training3.csv')
data = pd.read_csv(csv_file_path)

def calculate_sentiment_from_parts_of_speech(nouns, adjectives):
    # Combine nouns and adjectives into a single string
    text = " ".join(nouns + adjectives)
    # Calculate sentiment
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Apply sentiment calculation for each row
data['sentiment'] = data.apply(lambda row: calculate_sentiment_from_parts_of_speech(eval(row['nouns']), eval(row['adjectives'])), axis=1)

# Display the updated DataFrame
print(data.head())

# Save the updated DataFrame to a new CSV file
output_csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/training3_with_sentiment.csv')
data.to_csv(output_csv_file_path, index=False)