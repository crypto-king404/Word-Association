#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:35:45 2024

@author: rohan1809
"""
import random
import pandas as pd
from Player import Player
from nltk.corpus import wordnet as wn
from nltk import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import os

class NGramBasedPlayer(Player):
    PLAYER_NAME = "NGram-VADER-NRC-Based Player"

    def __init__(self):
        super().__init__(self.PLAYER_NAME)
        
        # Load processed tweet data
        csv_file_path = os.path.expanduser('~/Desktop/EAAI25-main/training3.csv')
        self.data = pd.read_csv(csv_file_path)
        print(f"Loaded tweet data with {len(self.data)} entries.")
        
        # Initialize VADER sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
        
        # Generate bigrams from processed data
        self.bigrams = self.generate_bigrams_from_data()
        print(f"Generated bigrams from data: {len(self.bigrams)} bigrams created.")
    
    def vader_sentiment_analysis(self, text):
        sentiment_scores = self.sid.polarity_scores(text)
        return sentiment_scores['compound']

    def nrc_emotion_analysis(self, text):
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores
    
    def load_words_from_data(self, column_name):
        words = set()
        for entry in self.data[column_name]:
            words.update(eval(entry))
        print(f"Loaded {len(words)} words from column {column_name}.")
        return list(words)

    def generate_bigrams_from_data(self):
        bigram_dict = {}
        adjectives = self.load_words_from_data('adjectives')
        nouns = self.load_words_from_data('nouns')
        
        for adj in adjectives:
            bigram_dict[adj] = [f"{adj} {noun}" for noun in nouns]
        
        return bigram_dict
    
    def is_meaningful_bigram(self, bigram):
        words = bigram.split()
        if len(words) != 2:
            return False
        synsets1 = wn.synsets(words[0])
        synsets2 = wn.synsets(words[1])
        if not synsets1 or not synsets2:
            return False
        similarity_score = max((s1.wup_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2)
        return similarity_score > 0.5 

    def evaluate_bigram(self, bigram):
        sentiment_score = self.vader_sentiment_analysis(bigram)
        emotion_scores = self.nrc_emotion_analysis(bigram)
        combined_score = sentiment_score + sum(emotion_scores.values())
        return combined_score

    def choose_card(self, target, hand):
        print(f"Choosing card for target: {target} from hand: {hand}")
        try:
            target_bigrams = self.bigrams.get(target, [])
            best_card = None
            best_score = -1
            
            for card in hand:
                for bigram in target_bigrams:
                    if card in bigram:
                        combined_score = self.evaluate_bigram(bigram)
                        print(f"Evaluating bigram: {bigram} with score: {combined_score}")
                        if combined_score > best_score:
                            best_card = card
                            best_score = combined_score
            
            if best_card is None:
                print(f"No meaningful bigram found for target: {target}. Choosing randomly.")
                best_card = random.choice(hand)
            return hand.index(best_card)
        except Exception as e:
            print(f"Error in choose_card: {e}")
            return random.randrange(0, len(hand))

    def judge_card(self, target, player_cards):
        print(f"Judging cards for target: {target} from player_cards: {player_cards}")
        try:
            target_bigrams = self.bigrams.get(target, [])
            best_card = None
            best_score = -1
            
            for card in player_cards:
                for bigram in target_bigrams:
                    if card in bigram:
                        combined_score = self.evaluate_bigram(bigram)
                        print(f"Evaluating bigram: {bigram} with score: {combined_score}")
                        if combined_score > best_score:
                            best_card = card
                            best_score = combined_score
            
            if best_card is None:
                print(f"No meaningful bigram found for target: {target}. Choosing randomly.")
                best_card = random.choice(player_cards)
            return best_card
        except Exception as e:
            print(f"Error in judge_card: {e}")
            return random.choice(player_cards)

    def process_results(self, result):
        print("Result", result)

if __name__ == '__main__':
    download('wordnet')
    download('omw-1.4')
    
    player = NGramBasedPlayer()
    player.run()
