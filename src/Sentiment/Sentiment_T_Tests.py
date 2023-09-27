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
from scipy import stats
import argparse
from text_utils import load_data
from sentiment_utils import split_story_10_sentiment, per_group
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()

    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_posts_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_posts_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labels_df", default="relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--overall_labels", default="../data/Sentiment_T_Tests/overall_labels.csv", help="path to the t-test folder for all story labels", type=str)
    parser.add_argument("--pairs", default="../data/Sentiment_T_Tests/pairs.csv", help="path to the t-test folder for all story pair labels", type=str)
    parser.add_argument("--chunks", default="../data/Sentiment_T_Tests/chunks_labels_", help="path to the t-test folder for each story label, broken up into chunks", type=str)

    args = parser.parse_args()
    return args

#Groups together all the raw sentiment scores--not the averages per section 
def group_raw_scores(df, l):
	new_df = df[['title', 'selftext']].get(df[l] == True)
	new_df['tokenized sentences'] = new_df['selftext'].apply(tokenize.sent_tokenize)
	new_df['sentiment groups'] = new_df['tokenized sentences'].apply(split_story_10_sentiment)
	new_df['comp sent per group'] = new_df['sentiment groups'].apply(per_group, args = ('compound',)) 
	compressed = pd.DataFrame(list(new_df['comp sent per group'])).to_dict(orient='list')
	raw_score_dict = {} 
	for key in compressed:
		raw_score_dict[key] = list(itertools.chain.from_iterable(compressed[key])) 
	return raw_score_dict

#Runs the t-test for all labels pre and post COVID-19 to see which are significant OVERALL 
def t_test(df_pre, df_post, labels):
	stat = []
	p_value = []
	for label in labels:
		label_pre = group_raw_scores(df_pre, label)
		label_post = group_raw_scores(df_post, label)
		pre = [value for values in label_pre.values() for value in values]
		post = [value for values in label_post.values() for value in values]
		t_test = stats.ttest_ind(pre, post)
		stat.append(t_test.statistic)
		p_value.append(t_test.pvalue)
	label_frame = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = labels)
	print(label_frame)
	return label_frame

#Runs the t-test for all labels pre and post COVID-19 for each CHUNK to see which chunks are significant 
def t_test_chunks(df_pre, df_post, labels):
	args = get_args()

	for label in labels:
		label_pre = group_raw_scores(df_pre, label)
		label_post = group_raw_scores(df_post, label)
		stat = []
		p_value = []
		for key in list(label_pre.keys()):
			t_test = stats.ttest_ind(label_pre[key], label_post[key])
			stat.append(t_test.statistic)
			p_value.append(t_test.pvalue)
		label_frame = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = list(label_pre.keys()))
		label_frame.index.name = f"{label}: Pre-Post Covid"
		sig_vals = label_frame.get(label_frame['P-Values'] < .05)
		label_frame.to_csv(f'{args.chunks}{label}.csv')
		print(label_frame)

#Runs the t-test between each pair of labels pre and post COVID-19 to see how the differences changed in significance
def t_test_two_labels(df_1, df_2, tuples):
	stats_pre = []
	stats_post = []

	p_values_pre = []
	p_values_post = []

	for tup in tuples: 
		label_dc_pre_1 = group_raw_scores(df_1, tup[0])
		label_dc_pre_2 = group_raw_scores(df_1, tup[1])

		label_dc_post_1 = group_raw_scores(df_2, tup[0])
		label_dc_post_2 = group_raw_scores(df_2, tup[1])

		pre_1 = [value for values in label_dc_pre_1.values() for value in values]
		pre_2 = [value for values in label_dc_pre_2.values() for value in values]
		t_test_pre = stats.ttest_ind(pre_1, pre_2)
		stats_pre.append(t_test_pre.statistic)
		p_values_pre.append(t_test_pre.pvalue)

		post_1 = [value for values in label_dc_post_1.values() for value in values]
		post_2 = [value for values in label_dc_post_2.values() for value in values]
		t_test_post = stats.ttest_ind(post_1, post_2)
		stats_post.append(t_test_post.statistic)
		p_values_post.append(t_test_post.pvalue)
	label_frame = pd.DataFrame(data = {'Statistics: Pre-Covid': stats_pre, 'Statistics: Post-Covid': stats_post, 'P-Values: Pre-Covid': p_values_pre, 'P-Values: Post-Covid': p_values_post}, index = tuples)
	sig_vals_pre = label_frame.get(label_frame['P-Values: Pre-Covid'] < .05)
	sig_vals_post = label_frame.get(label_frame['P-Values: Post-Covid'] < .05)
	print(label_frame)
	return label_frame

def main():
	args = get_args()

	labels_df, pre_covid_posts_df, post_covid_posts_df, birth_stories_df = load_data(args.birth_stories_df, args.pre_covid_posts_df, args.post_covid_posts_df, args.labels_df)

	labels = list(labels_df.columns)
	labels.remove('title')
	labels.remove('created_utc')
	labels.remove('Covid')
	labels.remove('Pre-Covid')
	labels.remove('Date')
	labels.remove('selftext')
	labels.remove('author')

	#Ran t-tests and loaded these into CSVs
	t_test(pre_covid_posts_df, post_covid_posts_df, labels).to_csv(args.overall_labels)
	t_test_chunks(pre_covid_posts_df, post_covid_posts_df, labels)
	tuples = [('Positive', 'Negative'), ('Medicated', 'Unmedicated'), ('Home', 'Hospital'), ('Birth Center', 'Hospital'), ('First', 'Second'), ('C-Section', 'Vaginal')]
	t_test_two_labels(pre_covid_posts_df,post_covid_posts_df, tuples).to_csv(args.pairs)

if __name__ == '__main__':
	main()