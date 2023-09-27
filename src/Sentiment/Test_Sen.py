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
from sentiment_utils import split_story_10_sentiment, per_group, group
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()

    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_posts_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_posts_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labels_df", default="relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--mar_june_2020_df", default="relevant_jsons/mar_june_2020_df.json.gz", help="path to df of the stories from COVID era 1", type=str)
    parser.add_argument("--june_nov_2020_df", default="relevant_jsons/june_nov_2020_df.json.gz", help="path to df of the stories from COVID era 2", type=str)
    parser.add_argument("--nov_2020_apr_2021_df", default="relevant_jsons/nov_2020_apr_2021_df.json.gz", help="path to df of the stories from COVID era 3", type=str)
    parser.add_argument("--apr_june_2021_df", default="relevant_jsons/apr_june_2021_df.json.gz", help="path to df of the stories from COVID era 4", type=str)

    #where to save
    parser.add_argument("--compound_sent_output", default="../data/Sentiment_Plots/Compound_Sentiment_Plot_", help="path to save png plot with compound sentiment of stories", type=str)
    parser.add_argument("--pos_neg_sent_output", default="../data/Sentiment_Plots/Pos_Neg_Sentiment_Plot_", help="path to save png plot with pos/neg sentiment of stories", type=str)
    parser.add_argument("--all", default="../data/Sentiment_Plots/All_Plot_", help="path to save png plot with sentiment of all stories", type=str)
    parser.add_argument("--pre_post", default="../data/Sentiment_Plots/Pre_Post_Covid/Pre_Post_Plot_", help="path to save png plot with sentiment of stories per label pair, 4 lines per graph", type=str)
    parser.add_argument("--four_sects", default="../data/Sentiment_Plots/4_Section_Sentiment_Plots/4_Sects_Pre_Post_Plot_", help="path to save png plot with sentiment of stories per label with 4 COVID eras", type=str)
    parser.add_argument("--label", default="../data/Sentiment_Plots/Singular_Labels/Pre_Post_Plot_", help="path to save png plot with sentiment of stories per label", type=str)
    parser.add_argument("--diff", default="../data/Sentiment_Plots/Differences_Plotted/Diff_Plot_", help="path to save png plot with differences of sentiment of stories betweem label pairs", type=str)
    
    args = parser.parse_args()
    return args

#Function to read all dataframes 
def load_data(path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_labels, path_to_e1, path_to_e2, path_to_e3, path_to_e4):
    labels_df = compress_json.load(path_to_labels)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)
    
    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)
    
    mar_june_2020_df = compress_json.load(path_to_e1)
    mar_june_2020_df = pd.read_json(mar_june_2020_df)

    june_nov_2020_df = compress_json.load(path_to_e2)
    june_nov_2020_df = pd.read_json(june_nov_2020_df)

    nov_2020_apr_2021_df = compress_json.load(path_to_e3)
    nov_2020_apr_2021_df = pd.read_json(nov_2020_apr_2021_df)

    apr_june_2021_df = compress_json.load(path_to_e4)
    apr_june_2021_df = pd.read_json(apr_june_2021_df)

    return labels_df, birth_stories_df, pre_covid_posts_df, post_covid_posts_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df    

# **Figure 2: Sentiment Analysis**

#Computes all story lengths
def story_lengths(lst):
    return len(lst)

#Converts the dictionary of values into a dataframe with only one value per section (the average of the sentiments)
def dict_to_frame(lst):
    compressed = pd.DataFrame(list(lst)).to_dict(orient='list')
    group_dict = {} 
    for key in compressed:
        group_dict[key] = np.mean(list(itertools.chain.from_iterable(compressed[key])))
    return(pd.DataFrame.from_dict(group_dict, orient='index', columns = ['Sentiments']))

#Plots Compound sentiment ONLY
def comp_sents(dfs, name):
    args = get_args()
    for df in dfs:
        #tokenize stories by sentence
        df['tokenized sentences'] = df['selftext'].apply(tokenize.sent_tokenize)
        df['sentiment groups'] = df['tokenized sentences'].apply(split_story_10_sentiment)
        df['comp sent per group'] = df['sentiment groups'].apply(per_group, args = ('compound',))
        sentiment_over_narrative = dict_to_frame(df['comp sent per group'])
        sentiment_over_narrative.index.name = 'Sections'

        plt.plot(sentiment_over_narrative['Sentiments'], label = f'{df.name} Compound Sentiment')
        plt.xlabel('Story Time')
        plt.ylabel('Sentiment')
        plt.legend()
    plt.savefig(f'{args.compound_sent_output}{name}.png')
    plt.clf()

#Plots Positive vs. Negative Sentiment 
def pos_neg_sents(dfs):
    args = get_args()

    for df in dfs:
        #tokenize stories by sentence
        df['tokenized sentences'] = df['selftext'].apply(tokenize.sent_tokenize)

        df['sentiment groups'] = df['tokenized sentences'].apply(split_story_10_sentiment)
        df['lengths'] = df['sentiment groups'].apply(story_lengths)

        df['Pos sent per group'] = df['sentiment groups'].apply(per_group, args = ('pos',))
        df['Neg sent per group'] = df['sentiment groups'].apply(per_group, args = ('neg',))

        sentiment_over_narrative_t1 = dict_to_frame(df['Pos sent per group'])
        sentiment_over_narrative_t1.index.name = 'Sections'

        sentiment_over_narrative_t2 = dict_to_frame(df['Neg sent per group'])
        sentiment_over_narrative_t2.index.name = 'Sections'

        #Plotting over narrative time
        plt.plot(sentiment_over_narrative_t1['Sentiments'], label = f'Pos Sentiment: {df.name}')
        plt.plot(sentiment_over_narrative_t2['Sentiments'], label = f'Neg Sentiment: {df.name}')
        plt.xlabel('Story Time')
        plt.ylabel('Sentiment')
        plt.title('Positive vs. Negative Sentiment')
        plt.legend()
        plt.savefig(f'{args.pos_neg_sent_output}{df.name}.png')
        plt.clf()

#Plots two labels (ex. medicated vs. unmedicated) 
def label_frames(dfs, tuples, x):
    args = get_args()

    for tup in tuples:
        for df in dfs:
            label_one = df[['title', 'selftext']].get(df[tup[0]] == True)
            label_two = df[['title', 'selftext']].get(df[tup[1]] == True)

            label_one['tokenized sentences'] = label_one['selftext'].apply(tokenize.sent_tokenize)    
            label_two['tokenized sentences'] = label_two['selftext'].apply(tokenize.sent_tokenize)    

            label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)
            label_two['sentiment groups'] = label_two['tokenized sentences'].apply(split_story_10_sentiment)

            label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))
            label_two['comp sent per group'] = label_two['sentiment groups'].apply(per_group, args = ('compound',))

            sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
            sentiment_over_narrative_one.index.name = 'Sections'

            sentiment_over_narrative_two = dict_to_frame(label_two['comp sent per group'])
            sentiment_over_narrative_two.index.name = 'Sections'

            if tup[0] == 'Negative' or 'Second' or 'Birth Center' or tup[1] == 'Negative' or 'Second' or 'Birth Center':
                sentiment_over_narrative_two['Sentiments']*=-1

            #Plotting each again over narrative time
            plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{tup[0]} Births: {df.name}')
            plt.plot(sentiment_over_narrative_two['Sentiments'], label = f'{tup[1]} Births: {df.name}')
            plt.xlabel('Story Time')
            plt.ylabel('Sentiment')
            plt.title(f'{tup[0]} vs. {tup[1]} Birth Sentiments')
            plt.legend()
            if x == True:
                plt.savefig(f'{args.all}{tup[0]}_{tup[1]}.png')
            else:
                plt.savefig(f'{args.pre_post}{tup[0]}_{tup[1]}.png')
        plt.clf()

#Plots a single label
def label_frame(dfs, labels, t):
    args = get_args()

    for label in labels:
        for df in dfs:
            df_one = df[['title', 'selftext']].get(df[label] == True)

            df_one['tokenized sentences'] = df_one['selftext'].apply(tokenize.sent_tokenize)     

            df_one['sentiment groups'] = df_one['tokenized sentences'].apply(split_story_10_sentiment)

            df_one['comp sent per group'] = df_one['sentiment groups'].apply(per_group, args = ('compound',))

            sentiment_over_narrative_one = dict_to_frame(df_one['comp sent per group'])
            sentiment_over_narrative_one.index.name = 'Sections'

            #Plotting each again over narrative time
            plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{label} Births: {df.name}')
            plt.xlabel('Story Time')
            plt.ylabel('Sentiment')
            plt.title(f'{label} Birth Sentiments')
            plt.legend()
            if t == True:
                plt.savefig(f'{args.four_sects}{label}.png')
            else:
                plt.savefig(f'{args.label}{label}.png')
        plt.clf()

#Plots only the difference between pre and post COVID-19 between the two labels 
def difference_pre_post(dfs, tuples):
    args = get_args()

    for tup in tuples:
        for df in dfs:
            label_one = df[['title', 'selftext']].get(df[tup[0]] == True)
            label_two = df[['title', 'selftext']].get(df[tup[1]] == True)

            label_one['tokenized sentences'] = label_one['selftext'].apply(tokenize.sent_tokenize)    
            label_two['tokenized sentences'] = label_two['selftext'].apply(tokenize.sent_tokenize)    

            label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)
            label_two['sentiment groups'] = label_two['tokenized sentences'].apply(split_story_10_sentiment)

            label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))
            label_two['comp sent per group'] = label_two['sentiment groups'].apply(per_group, args = ('compound',))

            sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
            sentiment_over_narrative_one.index.name = 'Sections'

            sentiment_over_narrative_two = dict_to_frame(label_two['comp sent per group'])
            sentiment_over_narrative_two.index.name = 'Sections'

            if tup[0] == 'Negative' or 'Second' or 'Birth Center' or tup[1] == 'Negative' or 'Second' or 'Birth Center':
                sentiment_over_narrative_two['Sentiments']*=-1

            #Plotting each difference over narrative time
            d = sentiment_over_narrative_one['Sentiments'] - sentiment_over_narrative_two['Sentiments']
            
            plt.plot(d, label = df.name)
            plt.xlabel('Story Time')
            plt.ylabel('Difference between Sentiments')
            plt.title(f'{tup[0]} vs. {tup[1]} Birth Sentiments')
            plt.legend()
            plt.savefig(f'{args.diff}{tup[0]}_{tup[1]}.png')
        plt.clf()

#Samples the split stories from a dataframe 
#Start is the starting section and end is the ending section, ex. sec1-7
def sample(df, start, end, size):
    sample = df.get(['title', 'selftext'])
    sample['tokenized sentences'] = sample['selftext'].apply(tokenize.sent_tokenize)     
    sample['sentiment groups'] = sample['tokenized sentences'].apply(split_story_10_sentiment)
    sample['sentences per group'] = sample['sentiment groups'].apply(per_group, args = ('sentences',))
    sampled = sample.sample(size)
    col = []
    titles = []
    for dt in sampled['sentences per group']:
        col.append(list(dt.items())[start:end])
    dic = {'title': sampled['title'], 'stories': col}
    new_df = pd.DataFrame(dic)
    return new_df

def main():

    args = get_args()

    dfs = load_data(args.birth_stories_df, args.pre_covid_posts_df, args.post_covid_posts_df, args.labels_df, args.mar_june_2020_df, args.june_nov_2020_df, args.nov_2020_apr_2021_df, args.apr_june_2021_df)
    birth_stories_df, pre_covid_posts_df, post_covid_posts_df, labels_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df = dfs

    labels = list(labels_df.columns)
    labels.remove('title')
    labels.remove('created_utc')
    labels.remove('Covid')
    labels.remove('Pre-Covid')
    labels.remove('Date')
    labels.remove('selftext')
    labels.remove('author')

    pre_covid_posts_df.name = 'Pre-Covid'
    post_covid_posts_df.name = 'Post-Covid'
    birth_stories_df.name = 'Birth_Stories'
    labels_df.name = 'Labeled_Stories'
    mar_june_2020_df.name = 'Mar-June 2020'
    june_nov_2020_df.name = 'June-Nov 2020'
    nov_2020_apr_2021_df.name = 'Nov 2020-April 2021'
    apr_june_2021_df.name = 'April-June 2021'

    tuples = [('Positive', 'Negative'), ('Medicated', 'Unmedicated'), ('Home', 'Hospital'), ('Birth Center', 'Hospital'), ('First', 'Second'), ('C-Section', 'Vaginal')]
    
    #Compound sentiment--entire dataset 
    comp_sents([birth_stories_df], "Overall")

    #Plots per COVID era
    label_frame([pre_covid_posts_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df], labels, True)
    
    #Plots per label
    label_frame([pre_covid_posts_df, post_covid_posts_df], labels, False)
    
    #Plots per COVID era
    label_frame([pre_covid_posts_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df], labels, True)
    
    #Plots per pair of labels (4 lines)
    label_frames([pre_covid_posts_df, post_covid_posts_df], tuples, False)
    
    #Plots differences between pairs 
    difference_pre_post([pre_covid_posts_df, post_covid_posts_df], tuples)

    #Compound sentiment--entire dataset 
    comp_sents([birth_stories_df], "Overall")

    #Split based on positive vs. negative sentiment--entire dataset 
    pos_neg_sents([birth_stories_df])

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines
    pos_neg_sents([pre_covid_posts_df, post_covid_posts_df])

    #Comparing labels--entire dataset--only relevant for pos vs. neg framed stories
    label_frames([labels_df], tuples, True)

    #Pre and Post Covid Sentiments
    #Starting with Compound Sentiment
    comp_sents([pre_covid_posts_df, post_covid_posts_df], "Pre_Post")

    #For the 4 time frames of Covid + pre-covid
    comp_sents([pre_covid_posts_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df], "4_Sects")

    #Stories mentioning Covid vs. Not
    #Compound Sentiment

    covid_df = pd.DataFrame()
    covid_df = labels_df.get(labels_df['Covid'] == True)
    covid_df.name = 'Covid_Mention'

    no_covid_df = pd.DataFrame()
    no_covid_df = labels_df.get(labels_df['Covid'] == False)
    no_covid_df.name = 'No_Covid_Mention'

    comp_sents([covid_df, no_covid_df], 'Covid')

if __name__ == '__main__':
    main()