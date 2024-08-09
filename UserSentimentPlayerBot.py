#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:15:50 2024

@author: rohan1809
"""
import random
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import bigrams
from itertools import product

# Load the DataFrame from the Excel file
file_path = '~/Desktop/EAAI25-main/UserTweetsData.xlsx'
tweets_df = pd.read_excel(file_path)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

# Apply sentiment analysis to each tweet
tweets_df['sentiment'] = tweets_df['Tweet'].apply(analyze_sentiment)

# Calculate overall sentiment for each user
user_sentiment = tweets_df.groupby('User')['sentiment'].mean().reset_index()
user_sentiment.columns = ['User', 'average_sentiment']

# Generate bigrams for each tweet
tweets_df['bigrams'] = tweets_df['Tweet'].apply(lambda x: list(bigrams(x.split())))

class SentimentNGramPlayer:
    def __init__(self, user, user_sentiment, tweets_df):
        self.user = user
        self.sentiment = user_sentiment
        self.tweets_df = tweets_df[tweets_df['User'] == user]
        self.bigrams = self.generate_bigrams()

    def generate_bigrams(self):
        bigram_dict = {}
        for bigram_list in self.tweets_df['bigrams']:
            for bigram in bigram_list:
                word1, word2 = bigram
                if word1 not in bigram_dict:
                    bigram_dict[word1] = []
                bigram_dict[word1].append(word2)
        return bigram_dict

    def choose_card(self, target_adjective, hand):
        target_bigrams = self.bigrams.get(target_adjective, [])
        sentiment_scores = [sid.polarity_scores(card)['compound'] for card in hand]

        best_card = None
        best_score = -float('inf')

        for card in hand:
            score = 0
            if card in target_bigrams:
                score += 1  # Bigram match
            score += 1 - abs(self.sentiment - sid.polarity_scores(card)['compound'])  # Sentiment alignment

            if score > best_score:
                best_score = score
                best_card = card

        if best_card is None:
            best_card = random.choice(hand)

        return best_card

    def judge_card(self, target_adjective, player_cards):
        best_card = None
        best_score = -float('inf')

        for card in player_cards:
            score = 0
            sentiment_score = sid.polarity_scores(card)['compound']
            if card in self.bigrams.get(target_adjective, []):
                score += 1
            score += 1 - abs(self.sentiment - sentiment_score)

            if score > best_score:
                best_score = score
                best_card = card

        if best_card is None:
            best_card = random.choice(player_cards)

        return best_card

# Example Usage:
users = ['taylorswift13', 'piersmorgan', 'SabrinaAnnLynn', 'elonmusk', 'Simone_Biles']
user_sentiment_dict = dict(zip(user_sentiment['User'], user_sentiment['average_sentiment']))

# Simulate a game round
for user in users:
    player = SentimentNGramPlayer(user, user_sentiment_dict[user], tweets_df)
    target_adjective = 'happy'  # Example target adjective
    hand = ['day', 'party', 'life', 'moment', 'dream']  # Example nouns
    
    chosen_card = player.choose_card(target_adjective, hand)
    print(f"{user} chose the card '{chosen_card}' for the target adjective '{target_adjective}'.")

    player_cards = ['day', 'party', 'life']
    judged_card = player.judge_card(target_adjective, player_cards)
    print(f"{user} judged the card '{judged_card}' as the best match for the target adjective '{target_adjective}'.\n")

