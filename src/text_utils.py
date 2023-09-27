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
import os
from date_utils import pandemic, get_post_date

#Function to read all dataframes 
def load_data(path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_labels):
    labels_df = compress_json.load(path_to_labels)
    labels_df = pd.read_json(labels_df)
    
    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    return labels_df, pre_covid_posts_df, post_covid_posts_df, birth_stories_df

#Function for story length
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

#to find the average story length between pre and post covid
def avg_story_length(dfs):
    avg_lengths = []
    for df in dfs: 
        df['story length'] = df['selftext'].apply(story_lengths)

        story_lens = list(df['story length'])
        avg_story_length = np.round(np.mean(story_lens),2)
        avg_lengths.append(avg_story_length)
        print(f'Average story length {df.name}: {avg_story_length}')
    return avg_lengths


#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = nltk.word_tokenize(story)
    n = 100
    for i in range(0, len(s), n):
        sentiment_story.append(' '.join(s[i:i + n]))
    return sentiment_story

#splits story into ten equal chunks
def split_story_10(string):
    tokenized = tokenize.word_tokenize(string)
    rounded = round(len(tokenized)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(tokenized) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story = []
    for i in ind:
        if i == ind[-1]:
            split_story.append(' '.join(tokenized[i:i+remainder]))
            return split_story
        split_story.append(' '.join(tokenized[i:i+rounded]))
    return split_story

#processes the story using little mallet wrapper process_string function
def process_s(s, stpwrds=True):
    stop = stopwords.words('english')
    if stpwrds==True:
        new = lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
        return new
    else:
        new = lmw.process_string(s,lowercase=True,remove_punctuation=True, remove_short_words=False, remove_stop_words=False)
        return new

#removes all emojis
def remove_emojis(s):
    regrex_pattern = re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',s)

#functions to assign labels to posts based on their titles
def findkey(title, labels):
    x = False
    for label in labels:
        if label in title:
            x = True
    return x

def findkeydisallow(title, labels, notlabels):
    x = False
    for label in labels:
        if label in title:
            for notlabel in notlabels:
                if notlabel in title:
                    return x
                else:
                    x = True
    return x

def create_df_label_list(df, column, dct, disallows):
    label_counts = []
    for label in list(dct):
        if not disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label]))
            label_counts.append(df[label].value_counts()[1])
        elif label not in disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label][0]))
            label_counts.append(df[label].value_counts()[1]) 
        else:
            df[label] = df[column].apply(lambda x: findkeydisallow(x, dct[label][0], dct[label][1]))
            label_counts.append(df[label].value_counts()[1]) 
    return label_counts

#Function to read all dataframes 
def load_data_bf(path_to_birth_stories):

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    return birth_stories_df
  
#cleans up the training_data file
def clean_training_text(row):
    cleaned = row.replace(to_replace=r"[0-9]+ no_label ", value='', regex=True)
    return list(cleaned)

def prepare_data(df, stopwords=True):
    #load in data
    birth_stories_df = compress_json.load(df)
    birth_stories_df = pd.read_json(birth_stories_df)

    if stopwords==True:
        #remove emojis, apply redditcleaner, process string with remove stop words
        birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)
    else:
        #remove emojis, apply redditcleaner, process string WITHOUT remove stop words
        birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s, args=(False))
    #replace urls with ''
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

    #remove numbers
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

    #remove any missing values
    birth_stories_df = birth_stories_df.dropna()
    return birth_stories_df

def process_df(birth_stories_df):
    #label stories as pre or post covid (March 11, 2020)
    birth_stories_df['date created'] = birth_stories_df['created_utc'].apply(get_post_date)
    birth_stories_df = birth_stories_df.sort_values(by = 'date created')
    birth_stories_df['Pre-Covid'] = birth_stories_df['date created'].apply(pandemic)
    return birth_stories_df

def get_first_comment(row, subreddit):
    curr_id, author = row.id, row.author
    if not os.path.exists(f"../data/original-reddit/subreddits/{subreddit}/comments/{curr_id}.json.gz"):
        return 
    comments_df = pd.read_json(f"../data/original-reddit/subreddits/{subreddit}/comments/{curr_id}.json.gz", compression='gzip')
    if comments_df.shape[0] == 0:
        return
    match_df = comments_df[(comments_df['parent_id'].map(lambda x: curr_id in x)) & (comments_df['author'] == author)].sort_values('created_utc',ascending=True)
    if match_df.shape[0] == 0:
        return 
    return match_df.iloc[0]['body']

def missing_text(birth_stories_df, subreddit):
    missing_text_df = birth_stories_df[birth_stories_df['selftext'].map(lambda x: not x)]
    missing_id_author_df = missing_text_df[['id', 'author', 'Pre-Covid']]
    print(missing_id_author_df.shape)
    missing_id_author_df['body'] = missing_id_author_df.apply(get_first_comment, args=(subreddit,), axis=1)
    missing_id_author_df['body'].map(lambda x: x == None).value_counts()

    missing_id_author_df[missing_id_author_df['body'] == None]
    print(missing_id_author_df.shape)
    print(birth_stories_df['selftext'].map(lambda x: not x).value_counts())
    for idx, row in missing_id_author_df.iterrows():
        birth_stories_df.at[idx, 'selftext'] = row.body

    birth_stories_df['selftext'].map(lambda x: not x).value_counts()

    birth_stories_df['selftext'].map(lambda x: x != None).value_counts()

    birth_stories_df[birth_stories_df['selftext'].map(lambda x: not not x)]['selftext'].shape

    birth_stories_df = birth_stories_df[birth_stories_df['selftext'].map(lambda x: not not x)]
    birth_stories_df.shape

    birth_stories_df['selftext'].map(lambda x: x != '[removed]' or x != '[deleted]').value_counts()

    birth_stories_df = birth_stories_df[birth_stories_df['selftext'] != '[removed]']
    birth_stories_df = birth_stories_df[birth_stories_df['selftext'] != '[deleted]']

    nan_value = float("NaN")
    birth_stories_df.replace("", nan_value, inplace=True)
    birth_stories_df.dropna(subset=['selftext'], inplace=True)

    return birth_stories_df

#gets rid of posts that have no content or are invalid 
def clean_posts(all_posts_df):

    warning = 'disclaimer: this is the list that was previously posted'
    all_posts_df['Valid'] = [findkeyword(sub, warning) for sub in all_posts_df['selftext']]
    all_posts_df = all_posts_df.get(all_posts_df['Valid'] == False)

    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[removed]']
    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[deleted]']

    return all_posts_df

def load_subreddits(BabyBumps, beyond_the_bump, BirthStories, daddit, predaddit, pregnant, Mommit, NewParents, InfertilityBabies):
    BabyBumps_df = compress_json.load(BabyBumps)
    BabyBumps_df = pd.read_json(BabyBumps_df)

    beyond_the_bump_df = compress_json.load(beyond_the_bump)
    beyond_the_bump_df = pd.read_json(beyond_the_bump_df)

    BirthStories_df = compress_json.load(BirthStories)
    BirthStories_df = pd.read_json(BirthStories_df)

    daddit_df = compress_json.load(daddit)
    daddit_df = pd.read_json(daddit_df)

    predaddit_df = compress_json.load(predaddit)
    predaddit_df = pd.read_json(predaddit_df)

    pregnant_df = compress_json.load(pregnant)
    pregnant_df = pd.read_json(pregnant_df)

    Mommit_df = compress_json.load(Mommit)
    Mommit_df = pd.read_json(Mommit_df)

    NewParents_df = compress_json.load(NewParents)
    NewParents_df = pd.read_json(NewParents_df)

    InfertilityBabies_df = compress_json.load(InfertilityBabies)
    InfertilityBabies_df = pd.read_json(InfertilityBabies_df)
    return BabyBumps_df, beyond_the_bump_df, BirthStories_df, daddit_df, predaddit_df, pregnant_df, Mommit_df, NewParents_df, InfertilityBabies_df