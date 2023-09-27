import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import compress_json
import argparse
import nltk
from nltk import tokenize
from scipy import stats
from scipy.stats import norm
from text_utils import load_data_bf
from stats_utils import compute_confidence_interval

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default= "birth_stories_df.json.gz", help="path to birth stories", type=str)
    parser.add_argument("--LIWC_df", default= "LIWC2015_results_birth_stories_and_ids.csv", help="path to csv with birth story LIWC scores", type=str)
    parser.add_argument("--LIWC_t_tests", default= "../results/LIWC_Results/LIWC_t_tests.csv", help="path to csv with birth story LIWC pre-post t-tests", type=str)
    parser.add_argument("--LIWC_CI_df", default="../results/LIWC_Results/LIWC_CI_df.csv", help="path to 95 percent confidence intervals CSVs", type=str)

    args = parser.parse_args()
    return args

def get_pre_post(birth_stories_df, df):
    args = get_args()
    df.rename({'Source (B)': 'id'}, axis=1, inplace=True)
    combined = pd.merge(birth_stories_df, df, on = 'id')
    pre_covid_posts = combined.get(combined['Pre-Covid'] == True)
    pre_covid_posts = pre_covid_posts.drop(['id', 'author', 'title', 'selftext', 'story length', 'created_utc','Pre-Covid', 'Valid', 'Source (A)'], axis =1)
    post_covid_posts = combined.get(combined['Pre-Covid'] == False)
    post_covid_posts = post_covid_posts.drop(['id', 'author', 'title', 'selftext', 'story length', 'created_utc','Pre-Covid', 'Valid', 'Source (A)'], axis = 1)
    return (pre_covid_posts, post_covid_posts) 

def t_tests(cols, pre_df, post_df, puncts):
    stat = []
    p_value = []
    sig_cols = []
    for col in cols:
        if col not in puncts:
            pre_scores = pre_df[col]
            post_scores = post_df[col]
            t_test = stats.ttest_ind(post_scores, pre_scores)
            if t_test.pvalue < .05:
                p_value.append(t_test.pvalue)
                stat.append(t_test.statistic)
                sig_cols.append(col)
    label_frame = pd.DataFrame(data = {'T-test Statistic': stat, 'P-Values': p_value}, index = sig_cols)
    label_frame = label_frame.dropna()
    label_frame = label_frame.sort_values(by=['P-Values'])
    return label_frame

def main():
    args = get_args()
    birth_stories_df = load_data_bf(args.birth_stories_df)
    LIWC_df = pd.read_csv(args.LIWC_df)
    pre_covid_posts, post_covid_posts = get_pre_post(birth_stories_df, LIWC_df)
    cols = list(pre_covid_posts.columns)
    cols.remove('Source (C)')
    puncts =  ['AllPunc', 'OtherP', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'WC', 'WPS']
    t_tests(cols, pre_covid_posts, post_covid_posts, puncts).to_csv(args.LIWC_t_tests)
    LIWC_CI_df = compute_confidence_interval(cols, pre_covid_posts, post_covid_posts, puncts)
    LIWC_CI_df.to_csv(args.LIWC_CI_df)

if __name__ == '__main__':
    main()