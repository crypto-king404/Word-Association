#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:32:01 2024

@author: rohan1809
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os

# Load the DataFrame from the Excel file
file_path = '~/Desktop/EAAI25-main/UserTweetsData.xlsx'
tweets_df = pd.read_excel(os.path.expanduser(file_path))

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment with VADER
def analyze_vader_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

# Function to analyze emotion with Sentic API
def analyze_sentic_emotion(text):
    try:
        # Replace this with your actual Sentic API key and correct base URL
        api_key = "GqUQ3m0uJiWPD"
        base_url = "https://sentic.net/api/en/" + api_key + ".py"
        
        # Encode the text properly for the URL
        encoded_text = text.replace(" ", "+")
        url = f"{base_url}?text={encoded_text}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text.strip()
        else:
            print(f"Error: {response.status_code} - {response.reason}")
            return "No emotion data"
    except Exception as e:
        print(f"Request error: {str(e)}")
        return "No emotion data"

# Apply VADER sentiment analysis to each tweet
tweets_df['vader_sentiment'] = tweets_df['Tweet'].apply(analyze_vader_sentiment)

# Apply Sentic emotion analysis to each tweet
tweets_df['sentic_emotion'] = tweets_df['Tweet'].apply(analyze_sentic_emotion)

# Display the DataFrame with sentiment and emotion data
print(tweets_df.head())

# Calculate overall sentiment for each user using VADER
user_vader_sentiment = tweets_df.groupby('User')['vader_sentiment'].mean().reset_index()
user_vader_sentiment.columns = ['User', 'average_vader_sentiment']

# Calculate most frequent emotion for each user using Sentic
user_sentic_emotion = tweets_df.groupby('User')['sentic_emotion'].agg(lambda x: x.value_counts().index[0]).reset_index()
user_sentic_emotion.columns = ['User', 'dominant_sentic_emotion']

# Combine sentiment and emotion data
user_analysis = pd.merge(user_vader_sentiment, user_sentic_emotion, on='User')

# Display the overall sentiment and emotion for each user
print(user_analysis)

# Save the user analysis to a new Excel file in the EAAI25-main folder
output_file_path = os.path.expanduser('~/Desktop/EAAI25-main/UserSentimentEmotionData.xlsx')
user_analysis.to_excel(output_file_path, index=False)

print(f"User sentiment and emotion data saved to {output_file_path}")




