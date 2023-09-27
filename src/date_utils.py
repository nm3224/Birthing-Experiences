import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from datetime import datetime
import compress_json

#translate created_utc column into dates
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

#turns utc timestamp into datetime object
def get_post_month(series):
    parsed_date = datetime.utcfromtimestamp(series)
    to_timestamp = pd.to_datetime(parsed_date, format="%m%Y")
    return to_timestamp

#Turns the date column into a year-month datetime object
def convert_datetime(post_covid_posts_df):
    post_covid_posts_df['Date Created'] = pd.to_datetime(post_covid_posts_df['Date'])
    post_covid_posts_df['year-month'] = post_covid_posts_df['Date Created'].dt.to_period('M')
    #import pdb; pdb.set_trace()
    #post_covid_posts_df['year-month'] = [month.to_timestamp() for month in post_covid_posts_df['year-month']]
    post_covid_posts_df.drop(columns=['Date', 'Date Created'], inplace=True)
    return post_covid_posts_df

#Checks what year
def this_year(date, y):
    start_date = datetime.strptime(y, "%Y")
    if date.year == start_date.year:
        return True
    else:
        return False

#todo look into why < 03-01 doesnt work
def pre_covid_posts(df):
    pre_covid = df[(df.index <= '2020-02-01')]
    return pre_covid

def posts_2019_on(df):
    recent_pre_covid = df[(df.index >= '2019-01-01')]
    return recent_pre_covid

#True/False column based on before and after pandemic 
def pandemic(date):
    start_date = datetime.strptime("11 March, 2020", "%d %B, %Y")
    if date > start_date:
        return False
    else:
        return True

#labels the dataframe with True or False based on whether the date the post was created falls within the inputed start and end date
def pandemic_eras(series, start_date, end_date):
    date = str(series)
    #date = date.split()[0]
    if end_date == '2021-06':
        if date >= start_date and date <= end_date:
            return True
        else:
            return False
    else:
        if date >= start_date and date < end_date:
            return True
        else:
            return False

def combine_topics_and_months(birth_stories_df, story_topics_df, period, drop=True):
    #load in data so that we can attach dates to stories
    birth_stories_df = compress_json.load(birth_stories_df)
    birth_stories_df = pd.read_json(birth_stories_df)

    if drop==True:
        #makes it even
        birth_stories_df.drop(birth_stories_df.head(3).index, inplace=True)

        #combines story dates with topic distributions
        birth_stories_df.reset_index(drop=True, inplace=True)
        dates_topics_df = pd.concat([birth_stories_df[['created_utc', 'id']], story_topics_df], axis=1)
    else:
        dates_topics_df = pd.merge(birth_stories_df[['created_utc', 'id']], story_topics_df, how='outer', left_on='id', right_on="Source (B)")

    #converts the date into datetime object for year and month
    dates_topics_df['Date Created'] = dates_topics_df['created_utc'].apply(get_post_month)
    dates_topics_df['date'] = pd.to_datetime(dates_topics_df['Date Created'])
    dates_topics_df['year-month'] = dates_topics_df['date'].dt.to_period(period)
    dates_topics_df['Date'] = [month.to_timestamp() for month in dates_topics_df['year-month']]
    dates_topics_df.drop(columns=['Date Created', 'created_utc', 'year-month', 'date'], inplace=True)

    dates_topics_df = dates_topics_df.set_index('Date')

    #groups stories by month and finds average
    dates_topics_df = pd.DataFrame(dates_topics_df.groupby(dates_topics_df.index).mean())
 
    return dates_topics_df