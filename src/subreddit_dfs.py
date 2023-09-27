import pandas as pd
import os
import numpy as np
import nltk
import compress_json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
import warnings
warnings.filterwarnings("ignore")
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm
from text_utils import story_lengths, process_df, missing_text, get_first_comment, clean_posts
from date_utils import get_post_date, pandemic
import argparse

def get_args():
    parser = argparse.ArgumentParser("Create initial dataframe of birth stories")
    parser.add_argument("--path", default="../data/original-reddit/subreddits")
    parser.add_argument("--output_each_subreddit", default="../data/subreddit_json_gzs")
    args = parser.parse_args()
    print(args)
    return args

def birthstories(series):
    lowered = series.lower()
    if 'birth story' in lowered:
        return True
    if 'birth stories' in lowered:
        return True
    if 'graduat' in lowered:
        return True
    else:
        return False

#return if a story is 500+ words long or not
def long_stories(series):
    if series >= 500:
        return True
    else:
        return False

def findkeyword(word, key):
    if word.find(key) == -1:
        return False
    return True

def create_dataframe(path, output_each_subreddit):
    birth_stories_df = pd.DataFrame()
    subreddits = ("BabyBumps", "beyondthebump", "BirthStories", "daddit", "Mommit", "predaddit", "pregnant", "NewParents", "InfertilityBabies")
    for subreddit in subreddits:
        df = f"{subreddit}_df"
        df = pd.DataFrame()
        for file in os.listdir(f"{path}/{subreddit}/submissions/"):
            post = f"{path}/{subreddit}/submissions/{file}"
            if os.path.getsize(post) > 55:
                content = pd.read_json(post)
                df = df.append(content)

        df['birth story'] = df['title'].apply(birthstories)
        df = df[df['birth story'] == True]

        df = process_df(df)
        df = missing_text(df, subreddit)

        df.reset_index(drop=True, inplace=True)
        df_j = df.to_json()
        compress_json.dump(df_j, f"{output_each_subreddit}/{subreddit}_df.json.gz")
        birth_stories_df = birth_stories_df.append(df, ignore_index=True)


    return birth_stories_df

def only_useful_long_stories(birth_stories_df):
    #get story lengths
    birth_stories_df['story length'] = birth_stories_df['selftext'].apply(story_lengths)

    #only rows where the story is 500+ words long
    birth_stories_df['500+'] = birth_stories_df['story length'].apply(long_stories)
    birth_stories_df = birth_stories_df[birth_stories_df['500+'] == True]

    #only useful columns
    birth_stories_df = birth_stories_df[['id','author', 'title', 'selftext','story length','created_utc', 'Pre-Covid']]

    return birth_stories_df

def main():
    args = get_args()

    birth_stories_df = create_dataframe(args.path, args.output_each_subreddit)
    birth_stories_df = clean_posts(birth_stories_df)
    birth_stories_df = only_useful_long_stories(birth_stories_df)

    #Convert to compressed json 
    birth_stories_df = birth_stories_df.to_json()
    compress_json.dump(birth_stories_df, "birth_stories_df.json.gz")

if __name__ == "__main__":
    main()