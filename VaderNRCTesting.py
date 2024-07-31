#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:01:43 2024

@author: rohan1809
"""
import pandas as pd
import os

# Define the path to the CSV file on your desktop
csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/TRIAL1-training3_with_sentiment_and_emotions.csv')
data = pd.read_csv(csv_file_path)

# Display the head of the DataFrame
print("Initial DataFrame:")
print(data.head())

# Check if 'vader_sentiment' column exists
if 'vader_sentiment' in data.columns:
    print("VADER Sentiment Distribution:")
    print(data['vader_sentiment'].value_counts())
else:
    print("Error: 'vader_sentiment' column not found in the DataFrame.")

# Extract emotions and calculate their distribution
def extract_emotions(emotion_dict):
    if isinstance(emotion_dict, str):
        return eval(emotion_dict)
    return {}

if 'emotions' in data.columns:
    data['emotions_dict'] = data['emotions'].apply(extract_emotions)
    all_emotions = data['emotions_dict'].apply(pd.Series).fillna(0)
    emotion_distribution = all_emotions.sum()

    print("Emotion Distribution:")
    print(emotion_distribution)

    # Visualize the emotion distribution
    import matplotlib.pyplot as plt

    emotion_distribution.plot(kind='bar')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Error: 'emotions' column not found in the DataFrame.")
