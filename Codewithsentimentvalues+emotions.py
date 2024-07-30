# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:10:41 2024

@author: smeno
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from nrclex import NRCLex

# Initialize VADER and BERT models
vader_analyzer = SentimentIntensityAnalyzer()
bert_emotion_model = pipeline('sentiment-analysis', model='bhadresh-savani/distilbert-base-uncased-emotion')

def analyze_sentiment_and_emotion(text):
    # Step 1: VADER for initial sentiment analysis
    vader_result = vader_analyzer.polarity_scores(text)
    sentiment = vader_result['compound']
    
    # Step 2: NRC Emotion Lexicon for basic emotions
    nrc = NRCLex(text)
    emotions = nrc.raw_emotion_scores
    
    # Step 3: BERT for contextual emotion analysis
    bert_emotion_analysis = bert_emotion_model(text)
    
    # Step 4: Combine results
    combined_result = {
        'vader_sentiment': vader_result,
        'nrc_emotions': emotions,
        'bert_emotions': bert_emotion_analysis
    }
    
    return combined_result

# Example usage
text = "I am happy but also a bit sad about the situation."
result = analyze_sentiment_and_emotion(text)
print(result)
