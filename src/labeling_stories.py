import pandas as pd
import nltk
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
import seaborn
import redditcleaner
import re
import warnings
import compress_json
warnings.filterwarnings("ignore")
from text_utils import get_post_date, pandemic, create_df_label_list, load_data_bf
import argparse
import json

#Read all relevant dataframe jsons

def get_args():
    parser = argparse.ArgumentParser()

    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--labels_ngrams", default="../data/labels_ngrams.json", help="path to dictionary with list of labels and the ngrams mapping to them", type=str)
    parser.add_argument("--covid_ngrams", default="../data/covid_ngrams.json", help="path to dictionary with all the ngrams that map to the covid label", type=str)

    #for labeling_stories.py
    parser.add_argument("--pre_covid_posts_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_posts_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--label_counts_output", default="../data/label_counts_df.csv", help="path for where to save csv for counts of different labels", type=str)
    parser.add_argument("--labels_df", default= "relevant_jsons/labeled_df.json.gz", help="path to json with stories labeled accordingly")

    args = parser.parse_args()
    return args

def create_dict(df, labels, not_labels, descriptions):
    args = get_args()

    counts = create_df_label_list(df, 'title', labels, not_labels)

    labels_dict = { 'Labels': list(labels),
    'Description': descriptions,
    'N-Grams': [labels['Positive'], labels['Negative'], labels['Unmedicated'], labels['Medicated'],
             labels['Home'], labels['Hospital'], labels['First'], labels['Second'], labels['C-Section'], labels['Vaginal']],
    'Number of Stories': counts}

    #turn dictionary into a dataframe
    label_counts_df = pd.DataFrame(labels_dict, index=np.arange(10))
    label_counts_df.set_index('Labels', inplace = True)
    label_counts_df.to_csv(args.label_counts_output)

    return labels_dict 

#splitting into pre and post pandemic corpuses based on post date
def split_pre_post(df, labeled_df, covid_labels):
    args = get_args()

    df['date created'] = df['created_utc'].apply(get_post_date)
    df = df.sort_values(by = 'date created')

    labeled_df['Pre-Covid'] = df['date created'].apply(pandemic)

    covid = create_df_label_list(labeled_df, 'selftext', covid_labels, [])
    labeled_df['Date'] = labeled_df['created_utc'].apply(get_post_date)

    #Subreddits before pandemic 
    pre_covid_posts_df = labeled_df.get(labeled_df['Pre-Covid']==True).get(list(labeled_df.columns))
    print(f"Subreddits before pandemic: {len(pre_covid_posts_df)}")

    #Subreddits after pandemic 
    post_covid_posts_df = labeled_df.get(labeled_df['Pre-Covid']==False).get(list(labeled_df.columns))
    print(f"Subreddits during/after pandemic: {len(post_covid_posts_df)}")

    #Read dataframes to compressed jsons so we can reference later
    labels_df = labeled_df.to_json()
    compress_json.dump(labels_df, args.labels_df)
    
    post_covid_posts_df = post_covid_posts_df.to_json()
    compress_json.dump(post_covid_posts_df, args.post_covid_posts_df)

    pre_covid_posts_df = pre_covid_posts_df.to_json()
    compress_json.dump(pre_covid_posts_df, args.pre_covid_posts_df)

def main():

    args = get_args()

    birth_stories_df = load_data_bf(args.birth_stories_df)

    with open(args.labels_ngrams, 'r') as fp:
        labels_and_n_grams = json.load(fp)

    with open(args.covid_ngrams, 'r') as fp:
        Covid = json.load(fp)

    labels_df = birth_stories_df[['title', 'selftext', 'created_utc', 'author']]

    disallows = ['Positive', 'Unmedicated', 'Medicated']

    des = ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
    'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
    'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births']

    labels_dict = create_dict(labels_df, labels_and_n_grams, disallows, des)

    split_pre_post(birth_stories_df, labels_df, Covid)

if __name__ == "__main__":
    main()