#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:59:09 2024

@author: rohan1809
"""
import random
from nltk.corpus import wordnet as wn
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

class NGramBasedPlayer:
    PLAYER_NAME = "NGram-Based Player"

    def __init__(self):
        self.adjectives = self.load_words("adjectives.txt")
        self.nouns = self.load_words("nouns.txt")
        self.bigrams = self.generate_bigrams(self.adjectives, self.nouns)
        self.sid = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")

    def load_words(self, filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file]

    def generate_bigrams(self, adjectives, nouns):
        bigram_dict = {}
        for adj in adjectives:
            bigram_dict[adj] = [f"{adj} {noun}" for noun in nouns if self.is_meaningful_bigram(f"{adj} {noun}")]
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

    def calculate_combined_score(self, bigram):
        words = bigram.split()
        synsets1 = wn.synsets(words[0])
        synsets2 = wn.synsets(words[1])
        if not synsets1 or not synsets2:
            semantic_score = 0
        else:
            semantic_score = max((s1.wup_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2)
        
        sentiment_score = self.sid.polarity_scores(bigram)['compound']
        
        semantic_weight = 0.7
        sentiment_weight = 0.3
        
        combined_score = (semantic_weight * semantic_score) + (sentiment_weight * sentiment_score)
        print(f"Bigram: {bigram}, Semantic Score: {semantic_score}, Sentiment Score: {sentiment_score}, Combined Score: {combined_score}")
        return combined_score

    def choose_card(self, target, hand):
        target_bigrams = self.bigrams.get(target, [])
        print(f"Target Bigrams for '{target}': {target_bigrams}")
        best_card = None
        best_score = -float('inf')
        for card in hand:
            for bigram in target_bigrams:
                if card in bigram:
                    combined_score = self.calculate_combined_score(bigram)
                    if combined_score > best_score:
                        best_score = combined_score
                        best_card = card
        if best_card is not None:
            return hand.index(best_card)
        return random.choice(range(len(hand)))

    def judge_card(self, target, player_cards):
        target_bigrams = self.bigrams.get(target, [])
        print(f"Target Bigrams for '{target}': {target_bigrams}")
        best_card = None
        best_score = -float('inf')
        for card in player_cards:
            for bigram in target_bigrams:
                if card in bigram:
                    combined_score = self.calculate_combined_score(bigram)
                    if combined_score > best_score:
                        best_score = combined_score
                        best_card = card
        if best_card is not None:
            return best_card
        return random.choice(player_cards)

    def process_results(self, result):
        print("Result", result)

if __name__ == '__main__':
    download('wordnet')
    download('omw-1.4')
    
    player = NGramBasedPlayer()
    # Example usage
    target_word = "handsome"
    hand = ["boy", "dumb", "water", "ice", "fire"]
    chosen_card_index = player.choose_card(target_word, hand)
    chosen_card = hand[chosen_card_index]
    print(f"Chosen card for target '{target_word}': {chosen_card}")

    # Judge cards
    player_cards = ["boy", "dumb", "water", "ice", "fire"]
    judged_card = player.judge_card(target_word, player_cards)
    print(f"Judged card for target '{target_word}': {judged_card}")
