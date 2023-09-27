"""
Loads topic distributions, groups scores by post month, trains a Prophet forecast model on posts before COVID,
projects probabilities for COVID-era months, compares forecasted data to actual data with a z-test,
plots forecasted data (with 95% confidence interval) compared to actual data for topics that were 
statistically significant in their differences.
"""

import pandas as pd
import compress_json
import os
import numpy as np 
import argparse

import little_mallet_wrapper as lmw
from prophet import Prophet
from scipy import stats
from scipy.stats import norm, pearsonr

from matplotlib import pyplot as plt

from date_utils import combine_topics_and_months, pre_covid_posts
from topic_utils import predict_topic_trend_and_plot_significant_differences, topic_distributions

def get_args():
    parser = argparse.ArgumentParser("Load topic distributions, train Prophet model for projection, apply z-test for statistical significance, plot topics that are statistically significant.")
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
    parser.add_argument("--topic_key_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_keys.50")
    parser.add_argument("--topic_dist_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_distributions.50")
    parser.add_argument("--topic_forecasts_data_output", default="../data/Topic_Modeling_Data/topic_forecasts", help="path to where topic forecast data is saved")
    parser.add_argument("--topic_forecasts_plots_output", default="../data/Topic_Modeling_Data/Topic_Forecasts", help="path to where topic forecast plots are saved")
    parser.add_argument("--birth_stories_topics", default="../data/Topic_Modeling_Data/birth_stories_df_topics.csv")
    parser.add_argument("--ztest_output", default="../data/Topic_Modeling_Data/Z_Test_Stats.csv")
    args = parser.parse_args()
    return args
	
def main():
	args = get_args()

	#1. load topic model
	story_topics_df = topic_distributions(args.topic_dist_path, args.topic_key_path)
	dates_topics_df = combine_topics_and_months(args.birth_stories_df, story_topics_df, period="M")

	#2. for every topic:
		#train a model
		#project the model on held-out data
		#compare the projections to the held-out data
		#compute statistical tests
		#make figures if it's statistically significant

	if not os.path.exists(args.topic_forecasts_plots_output):
		os.mkdir(args.topic_forecasts_plots_output)

	if not os.path.exists(args.topic_forecasts_data_output):
		os.mkdir(args.topic_forecasts_data_output)

	pre_covid = pre_covid_posts(dates_topics_df)
	predict_topic_trend_and_plot_significant_differences(pre_covid, dates_topics_df, args.topic_forecasts_plots_output, args.ztest_output)

if __name__ == "__main__":
    main()