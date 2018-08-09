from nltk.tokenize import TweetTokenizer as Tokenizer_NLTK
from nltk.tokenize.casual import remove_handles
from nltk.stem.snowball import GermanStemmer as Stemmer_NLTK

import numpy as np

def get_train_data(filename):
    X  = []
    y_task1 = []
    y_task2 = []
    
    with open(filename) as file:
        for line in file:
            tweet = line.rstrip('\n').split('\t')
            X.append(tweet[0])
            y_task1.append(tweet[1])
            y_task2.append(tweet[2])
    
    return np.asarray(X), np.asarray(y_task1), np.asarray(y_task2)

def get_test_data(filename):
    X  = []
    
    with open(filename) as file:
        for line in file:
            tweet = line.rstrip('\n')
            X.append(tweet)

    return np.asarray(X)

class Tokenizer:
    def __init__(self, preserve_case=True, use_stemmer=False, join=False):
        self.preserve_case = preserve_case
        self.use_stemmer = use_stemmer
        self.join = join
        
    def tokenize(self, tweet):
        tweet = remove_handles(tweet)
        
        tweet = tweet.replace('#', ' ')
        tweet = tweet.replace('&lt;', ' ')
        tweet = tweet.replace('&gt;', ' ')
        tweet = tweet.replace('&amp;', ' und ')
        tweet = tweet.replace('|LBR|', ' ')
        tweet = tweet.replace('-', ' ')
        tweet = tweet.replace('_', ' ')
        tweet = tweet.replace("'s", ' ')
        tweet = tweet.replace(",", ' ')
        tweet = tweet.replace(";", ' ')
        tweet = tweet.replace(":", ' ')
        tweet = tweet.replace("/", ' ')
        tweet = tweet.replace("+", ' ')
        
        tknzr = Tokenizer_NLTK(preserve_case=self.preserve_case, reduce_len=True)
        
        if self.join:
            return " ".join(tknzr.tokenize(tweet))
        elif self.use_stemmer: 
            stmmr = Stemmer_NLTK()
            return [stmmr.stem(token) for token in tknzr.tokenize(tweet)]
        else:
            return tknzr.tokenize(tweet)

def find_subtoken(unfound, word2vec_model, mode='initial'):
    found_affix = None
    
    if mode == 'initial':
        chunk = len(unfound) - 1
        while chunk > 2:
            try:
                word2vec_model[unfound[:chunk]]
                found_affix = unfound[:chunk]
                break
            except:
                chunk -= 1
    elif mode == 'final':
        chunk = 1
        while len(unfound) - chunk > 2:
            try:
                word2vec_model[unfound[chunk:]]
                found_affix = unfound[chunk:]
                break
            except:
                chunk += 1   
                
    return found_affix