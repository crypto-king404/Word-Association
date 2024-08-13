#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:41:27 2024

@author: rohan1809
"""
import random
import pandas as pd
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RefinedNGramPlayer:
    PLAYER_NAME = "Refined NGram-Based Player"

    def __init__(self, user, user_sentiment, user_emotion):
        self.user = user
        self.user_sentiment = user_sentiment
        self.user_emotion = user_emotion

        # Load adjectives and nouns from CSV files
        adjectives_file_path = '~/Desktop/EAAI25-main/Adjectives.csv'
        nouns_file_path = '~/Desktop/EAAI25-main/nounlist.csv'
        self.adjectives = self.load_words_from_csv(adjectives_file_path, 'Word')
        self.nouns = self.load_words_from_csv(nouns_file_path, 'ATM')
        
        print(f"Loaded {len(self.adjectives)} adjectives and {len(self.nouns)} nouns.")
        
        # Generate bigrams from adjectives and nouns
        self.bigrams = self.generate_bigrams()
        print(f"Generated {len(self.bigrams)} bigrams from adjectives and nouns.")
        
        # Load pre-trained GloVe vectors
        self.glove_vectors = self.load_glove_vectors()

        # Initialize VADER sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
    
    def load_words_from_csv(self, file_path, column_name):
        df = pd.read_csv(file_path)
        words = df[column_name].dropna().tolist()
        return words

    def generate_bigrams(self):
        bigram_dict = {}
        for adj in self.adjectives:
            bigram_dict[adj] = [f"{adj} {noun}" for noun in self.nouns]
        return bigram_dict

    def load_glove_vectors(self, glove_file='glove.6B.50d.txt'):
        # Load pre-trained GloVe vectors (50-dimensional embeddings)
        glove_path = os.path.expanduser(f'~/Desktop/EAAI25-main/{glove_file}')
        glove_vectors = {}
        with open(glove_path, 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_vectors[word] = vector
        print("GloVe vectors loaded.")
        return glove_vectors

    def get_glove_vector(self, word):
        return self.glove_vectors.get(word.lower())

    def semantic_similarity(self, word1, word2):
        vector1 = self.get_glove_vector(word1)
        vector2 = self.get_glove_vector(word2)
        if vector1 is not None and vector2 is not None:
            return cosine_similarity([vector1], [vector2])[0][0]
        return 0

    def get_word_sentiment(self, word):
        # Use VADER to get sentiment of the word
        sentiment_scores = self.sid.polarity_scores(word)
        return sentiment_scores['compound']  # Return compound score as sentiment

    def validate_bigram(self, bigram):
        # POS Tagging Validation
        words = bigram.split()
        pos_tags = nltk.pos_tag(words)
        if pos_tags[0][1] != 'JJ' or pos_tags[1][1] != 'NN':
            return False  # Ensure first word is an adjective and second is a noun
        
        # Synonym Matching using WordNet
        adj_synsets = wn.synsets(words[0], pos=wn.ADJ)
        noun_synsets = wn.synsets(words[1], pos=wn.NOUN)
        if adj_synsets and noun_synsets:
            for adj_synset in adj_synsets:
                if words[1] in adj_synset.lemma_names():
                    return True
        
        return False

    def dynamic_weighting(self, similarity_score, sentiment_influence):
        if similarity_score > 0.5:
            return sentiment_influence * 0.3  # Reduce impact if similarity is high
        return sentiment_influence * 2  # Amplify impact if similarity is low

    def choose_card(self, target_adjective, hand):
        print(f"Choosing card for target: {target_adjective}")
        
        target_bigrams = self.bigrams.get(target_adjective, [])
        
        best_card = None
        best_score = -float('inf')

        for card in hand:
            score = 0
            
            # Semantic similarity
            similarity_score = self.semantic_similarity(target_adjective, card)
            score += 2 * similarity_score  # Increase weight of similarity score
            
            # Word sentiment
            word_sentiment = self.get_word_sentiment(card)
            if self.user_sentiment > 0 and word_sentiment > 0:
                score += word_sentiment * 1.5  # Positive user with positive word
            elif self.user_sentiment < 0 and word_sentiment < 0:
                score += abs(word_sentiment) * 1.5  # Negative user with negative word
            else:
                score -= abs(word_sentiment) * 1.5  # Mismatch between user sentiment and word sentiment

            # Validate bigram
            if card in target_bigrams and self.validate_bigram(f"{target_adjective} {card}"):
                score += 1  # Bigram match

            # Emotion influence
            if 'delight' in self.user_emotion or 'ecstasy' in self.user_emotion:
                if word_sentiment > 0:
                    score += 0.7  # Boost for positive emotions and positive words
            elif 'enthusiasm' in self.user_emotion or 'serenity' in self.user_emotion:
                if word_sentiment > 0:
                    score += 0.5  # Moderate boost for positive words
            elif 'terror' in self.user_emotion or 'loathing' in self.user_emotion:
                if word_sentiment < 0:
                    score += 0.7  # Boost for negative emotions and negative words
                else:
                    score -= 0.5  # Penalization for mismatch with negative emotions

            print(f"Card: {card}, Similarity Score: {similarity_score}, Word Sentiment: {word_sentiment}, Total Score: {score}")

            if score > best_score:
                best_score = score
                best_card = card

        if best_card is None:
            best_card = random.choice(hand)

        print(f"Chosen card: {best_card} with score: {best_score}")
        return best_card

# Simulate the game for each user with the same target adjective and hand
def simulate_game_for_users_same_target_hand(users, user_sentiment_emotion_df, target_adjective, hand):
    results = []
    for user in users:
        user_data = user_sentiment_emotion_df[user_sentiment_emotion_df['User'] == user]
        user_sentiment = user_data['average_vader_sentiment'].values[0]
        user_emotion = user_data['dominant_sentic_emotion'].values[0]

        player = RefinedNGramPlayer(user, user_sentiment, user_emotion)
        
        chosen_card = player.choose_card(target_adjective, hand)
        print(f"{user} chose the card '{chosen_card}' for the target adjective '{target_adjective}'.")

        results.append({
            'User': user,
            'Target Adjective': target_adjective,
            'Chosen Card': chosen_card
        })

    return pd.DataFrame(results)

# Example Usage:
users = ['taylorswift13', 'piersmorgan', 'SabrinaAnnLynn', 'elonmusk', 'Simone_Biles']
user_sentiment_emotion_df = pd.read_excel('~/Desktop/EAAI25-main/UserSentimentEmotionData.xlsx')

# Define a common target adjective and hand for all users
target_adjective = 'beautiful'  # Example target adjective
hand = ['girl', 'boulder', 'leaf', 'disaster', 'rock']  # Example nouns hand

results_df = simulate_game_for_users_same_target_hand(users, user_sentiment_emotion_df, target_adjective, hand)

# Display results
print(results_df)


