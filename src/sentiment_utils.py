import nltk
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import little_mallet_wrapper as lmw
from little_mallet_wrapper import process_string
import redditcleaner
import re
import compress_json
from scipy import stats
from scipy.stats import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#set up sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

#Splits stories into 10 sections and runs sentiment analysis on them
def split_story_10_sentiment(lst):
    sentiment_story = []
    if isinstance(lst, float) == True:
        lst = str(lst)
    for sentence in lst:
        if len(tokenize.word_tokenize(sentence)) >=5:
            analyzed = sentiment_analyzer_scores(sentence)
            sentiment_story.append(analyzed)
    rounded = round(len(lst)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(lst) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story_sents = []
    for i in ind:
        if i == ind[-1]:
            split_story_sents.append(sentiment_story[i:i+remainder])
            return split_story_sents
        split_story_sents.append(sentiment_story[i:i+rounded])
    return split_story_sents

#Creates list of the story sentiment values per section of the story 
def group(story, num, val):
    compound_scores = []
    sentences = []
    for sent in story[num]:
        if val == 'compound' or val == 'pos' or val == 'neg':
            dictionary = sent[1]
            compound_score = dictionary[val]
            compound_scores.append(compound_score)
        else:
            sen = sent[0]
            sentences.append(sen)
    if val == 'sentences': 
        return " ".join(sentences)
    else:
        return compound_scores


#Groups together the stories per section in a dictionary
def per_group(story, val):
    group_dict = {} 
    for i in np.arange(10):
        group_dict[f"0.{str(i)}"] = group(story, i, val)
    return group_dict