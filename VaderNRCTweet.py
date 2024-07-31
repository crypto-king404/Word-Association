#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:09:39 2024

@author: rohan1809
"""
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Define the path to the CSV file on your desktop
csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/training3.csv')
data = pd.read_csv(csv_file_path)

# Display the head of the DataFrame
print(data.head())

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def vader_sentiment_analysis(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

def nrc_emotion_analysis(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

def calculate_sentiment_and_emotion(nouns, adjectives):
    # Combine nouns and adjectives into a single string
    text = " ".join(nouns + adjectives)
    
    # Calculate sentiment using VADER
    vader_score = vader_sentiment_analysis(text)
    
    # Calculate emotions using NRCLex
    emotions = nrc_emotion_analysis(text)
    
    return vader_score, emotions

# Apply sentiment and emotion calculation for each row
data[['vader_score', 'emotions']] = data.apply(lambda row: pd.Series(calculate_sentiment_and_emotion(eval(row['nouns']), eval(row['adjectives']))), axis=1)

# Determine sentiment from VADER score
data['vader_sentiment'] = data['vader_score'].apply(lambda x: 'positive' if x >= 0 else 'negative')

# Display the updated DataFrame
print(data.head())

# Print sentiment distribution
print("VADER Sentiment Distribution:")
print(data['vader_sentiment'].value_counts())

# Save the results to a new CSV file
output_csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/TRIAL1-training3_with_sentiment_and_emotions.csv')
data.to_csv(output_csv_file_path, index=False)

# If you had target labels, calculate accuracy and generate reports
if 'target' in data.columns:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Map 'target' to binary sentiments if needed
    # data['target'] = data['target'].map({0: 'negative', 4: 'positive'})

    # Calculate accuracy
    accuracy_vader = accuracy_score(data['target'], data['vader_sentiment'])
    print(f"VADER Accuracy: {accuracy_vader}")

    # Generate classification report
    report_vader = classification_report(data['target'], data['vader_sentiment'])
    print("VADER Classification Report:")
    print(report_vader)

    # Generate confusion matrix
    conf_matrix_vader = confusion_matrix(data['target'], data['vader_sentiment'])
    print("VADER Confusion Matrix:")
    print(conf_matrix_vader)
else:
    print("No target labels found. Displaying sentiment distribution.")




