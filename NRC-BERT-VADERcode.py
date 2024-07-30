# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:16:20 2024

@author: smeno
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from nrclex import NRCLex

# Initialize VADER and BERT models
vader_analyzer = SentimentIntensityAnalyzer()
bert_emotion_model = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

def analyze_emotion(word):
    # Construct a synthetic sentence to provide context
    sentence = f"This card represents {word}"
    
    # Step 1: NRC Emotion Lexicon
    nrc = NRCLex(word)
    nrc_emotions = nrc.raw_emotion_scores
    
    # Step 2: BERT for contextual emotion analysis
    bert_emotion_analysis = bert_emotion_model(sentence)
    
    # Process BERT results to extract the primary emotion
    if bert_emotion_analysis:
        bert_emotion = bert_emotion_analysis[0]
        final_emotion = {
            'label': bert_emotion['label'],
            'score': bert_emotion['score']
        }
    else:
        final_emotion = {
            'label': 'unknown',
            'score': 0.0
        }
    
    # Step 3: VADER for sentiment analysis
    vader_result = vader_analyzer.polarity_scores(sentence)
    
    # Combine NRC, BERT, and VADER results
    combined_emotions = {
        'nrc_emotions': nrc_emotions,
        'bert_emotions': final_emotion,
        'vader_sentiment': vader_result
    }
    
    return combined_emotions

# Example usage
word = "Unfortunate"
result = analyze_emotion(word)
print(result)
