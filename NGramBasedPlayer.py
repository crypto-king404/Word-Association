#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:28:35 2024

@author: rohan1809
"""
import random
from Player import Player
from nltk.corpus import wordnet as wn
from nltk import download
from nltk.util import bigrams
from itertools import product

class NGramBasedPlayer(Player):
    PLAYER_NAME = "NGram-Based Player"

    def __init__(self):
        super().__init__(self.PLAYER_NAME)
        self.adjectives = self.load_words("adjectives.txt")
        self.nouns = self.load_words("nouns.txt")
        self.bigrams = self.generate_bigrams(self.adjectives, self.nouns)

    def load_words(self, filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file]

    def generate_bigrams(self, adjectives, nouns):
        bigram_dict = {}
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

    def choose_card(self, target, hand):
        target_bigrams = self.bigrams.get(target, [])
        for card in hand:
            for bigram in target_bigrams:
                if card in bigram:
                    return hand.index(card)
        return random.choice(range(len(hand)))

    def judge_card(self, target, player_cards):
        target_bigrams = self.bigrams.get(target, [])
        for card in player_cards:
            for bigram in target_bigrams:
                if card in bigram:
                    return card
        return random.choice(player_cards)

    def process_results(self, result):
        print("Result", result)

if __name__ == '__main__':
    download('wordnet')
    download('omw-1.4')
    
    player = NGramBasedPlayer()
    player.run()
