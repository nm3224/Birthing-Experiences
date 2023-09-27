import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import compress_json
import argparse
import nltk
from nltk import tokenize
from scipy import stats
from scipy.stats import norm
from stats_utils import compute_confidence_interval

def get_args():
    parser = argparse.ArgumentParser()

    #output path for plots
    parser.add_argument("--pre_covid_mentions", default="/home/daphnaspira/birthing_experiences/data/Personas_Data/persona_csvs/pre_covid_persona_mentions.csv", help="path to pre covid persona mentions", type=str)
    parser.add_argument("--post_covid_mentions", default="/home/daphnaspira/birthing_experiences/data/Personas_Data/persona_csvs/post_covid_persona_mentions.csv", help="path to post covid persona mentions", type=str)
    parser.add_argument("--persona_CI_df", default="../results/Personas_Results/persona_CI_df.csv", help="path to 95 percent confidence intervals CSVs", type=str)

    args = parser.parse_args()
    return args

def read_csvs(path_pre, path_post):
    pre_covid_persona_mentions = pd.read_csv(path_pre)
    post_covid_persona_mentions = pd.read_csv(path_post)

    return pre_covid_persona_mentions, post_covid_persona_mentions 

def main():
    args = get_args()
    pre_covid_persona_mentions, post_covid_persona_mentions = read_csvs(args.pre_covid_mentions, args.post_covid_mentions)
    personas = list(pre_covid_persona_mentions.columns)
    persona_CI_df = compute_confidence_interval(personas, pre_covid_persona_mentions, post_covid_persona_mentions, [])
    persona_CI_df.to_csv(args.persona_CI_df)

if __name__ == '__main__':
    main()