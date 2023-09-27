import pandas as pd
import little_mallet_wrapper as lmw
import os
import nltk
from nltk import ngrams
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
import itertools
from itertools import chain, zip_longest
from little_mallet_wrapper import process_string
import seaborn
import redditcleaner
import re
import warnings
import itertools
import compress_json
import argparse
from text_utils import load_data_bf
from date_utils import get_post_month, pandemic 
from topic_utils import average_per_story, top_5_keys, topic_distributions

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--topic_key_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_keys.50")
    parser.add_argument("--topic_dist_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_distributions.50")
    parser.add_argument("--topic_sample", default="../data/Samples/topic_sample_", help="path to sample of topics", type=str)
    args = parser.parse_args()
    return args

def combine_topics_and_months(birth_stories_df, story_topics_df):
    #makes it even
    birth_stories_df.drop(birth_stories_df.head(3).index, inplace=True)

    #combines story dates with topic distributions
    birth_stories_df.reset_index(drop=True, inplace=True)
    dates_topics_df = pd.concat([birth_stories_df['created_utc'], birth_stories_df['title'], birth_stories_df['selftext'], story_topics_df], axis=1)

    #converts the date into datetime object for year and month
    dates_topics_df['Date Created'] = dates_topics_df['created_utc'].apply(get_post_month)
    dates_topics_df['date'] = pd.to_datetime(dates_topics_df['Date Created'])

    dates_topics_df = dates_topics_df.set_index('date')
    return dates_topics_df

def get_post_covid_posts(df):
    df = df.sort_values(by = 'Date Created')
    df['Pre-Covid'] = df['Date Created'].apply(pandemic)
    df.drop(columns=['created_utc', 'Date Created'], inplace=True)
    post_df = (df.get(df['Pre-Covid'] == False))
    return post_df

def get_samples(post_df, topics):
    args = get_args()
    for topic in topics:
        post_df_sorted = post_df.sort_values(by = topic)
        topic_df_highest = post_df_sorted.get([topic, 'title', 'selftext']).tail(10)
        topic_df_lowest = post_df_sorted.get([topic, 'title', 'selftext']).head(10)
        topic_df_highest.to_excel(f'{args.topic_sample}{topic}_high.xlsx')
        topic_df_lowest.to_excel(f'{args.topic_sample}{topic}_low.xlsx')

def main():
    args = get_args()
    birth_stories_df = load_data_bf(args.birth_stories_df)
    story_topics_df = topic_distributions(args.topic_dist_path, args.topic_key_path)
    dates_topics_df = combine_topics_and_months(birth_stories_df, story_topics_df)
    post_covid_posts = get_post_covid_posts(dates_topics_df)
    get_samples(post_covid_posts, ['cervix hours pitocin started dilated', 'mom husband family time sister', 'milk breastfeeding feeding formula baby', 'baby skin cord husband born', 'contractions minutes apart around started'])

if __name__ == "__main__":
    main()