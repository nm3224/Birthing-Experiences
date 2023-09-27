import pandas as pd
import little_mallet_wrapper as lmw
import numpy as np
import compress_json
from datetime import datetime

import little_mallet_wrapper as lmw
from prophet import Prophet
from scipy import stats
from scipy.stats import norm, pearsonr

from matplotlib import pyplot as plt

from date_utils import get_post_month
from stats_utils import ztest

def get_all_chunks_from_column(series):
    #makes list of all chunks from all stories in the df
    training_chunks = []
    for story in series:
        for chunk in story:
            training_chunks.append(chunk)
    return training_chunks

#makes list of all the chunks for topic inferring
def get_chunks(series):
    testing_chunks = []
    for story in series:
        for chunk in story:
            testing_chunks.append(chunk)
    return testing_chunks

#finds average probability for each topic for each chunk of story
def average_per_story(df):
    dictionary = {}
    for i in range(len(df)//10):
        story = df[df['chunk_titles'].str.contains(str(i)+':')]
        means = story.mean()
        dictionary[i] = means
    return pd.DataFrame.from_dict(dictionary, orient='index')

#makes string of the top five keys for each topic
def top_6_keys(lst):
    top6_per_list = []
    for l in lst:
        joined = ' '.join(l[:6])
        top6_per_list.append(joined)
    return top6_per_list

def topic_distributions(file_path, topic_key_path):
    #makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(file_path)
    story_distributions =  pd.Series(topic_distributions)
    story_topics_df = story_distributions.apply(pd.Series)

    #goes through stories and names them based on the story number and chunk number
    chunk_titles = []
    for i in range(len(story_topics_df)//10):
        for j in range(10):
            chunk_titles.append(str(i) + ":" + str(j))

    story_topics_df['chunk_titles'] = chunk_titles

    #groups every ten stories together and finds the average for each story
    story_topics_df.groupby(story_topics_df.index // 10)
    story_topics_df = average_per_story(story_topics_df)

    #loads topic keys
    topic_keys = lmw.load_topic_keys(topic_key_path)
    six_keys = top_6_keys(topic_keys)

    #adds the keys as the names of the topic columns
    story_topics_df.set_axis(six_keys, axis=1, inplace=True)
    return story_topics_df

def prophet_projection(df, df2, topic_label, i, m, periods, frequency):
    topic = pd.DataFrame(df.iloc[:,i])
    topic.reset_index(inplace=True)
    topic.columns = ['ds', 'y']
    topic['ds'] = topic['ds'].dt.to_pydatetime()

    actual = pd.DataFrame(df2.iloc[:,i])
    actual.reset_index(inplace=True)
    actual.columns = ['ds', 'y']
    actual['ds'] = actual['ds'].dt.to_pydatetime()

    m.fit(topic)

    future = m.make_future_dataframe(periods=periods, freq=frequency)

    forecast = m.predict(future)
    return forecast

def projection_percent_outside_ci_and_ztest(forecast, df2, topic_label, pre_ztest_dict, post_ztest_dict):
    values = df2.loc[:, topic_label]

    #import pdb; pdb.set_trace()

    #finds values that are outside of the forecasted confidence interval
    inside_forecast = []
    for j in range(len(values)):
        inside_forecast.append(forecast["yhat_lower"][j] <= values[j] <= forecast["yhat_upper"][j])
    values_df = values.to_frame()
    values_df['inside_forecast'] = inside_forecast

    forecast_pre = forecast.get(forecast['ds'] <= '2020-02-01')
    forecast_post = forecast.get(forecast['ds'] > '2020-02-01')

    #splits up data pre and post covid and finds percentage of values that are outside of the CI for each
    values_df.reset_index(inplace=True)
    pre = values_df.get(values_df['Date'] <= '2020-02-01')
    post = values_df.get(values_df['Date'] > '2020-02-01')
    outside_ci_pre = pre.get(values_df['inside_forecast']==False)
    outside_ci_post = post.get(values_df['inside_forecast']==False)
    percent_pre = (len(outside_ci_pre)/len(pre))
    percent_post = (len(outside_ci_post)/len(post))

    #z-test
    ztest_vals_pre = ztest(pre[topic_label], forecast_pre['yhat'], percent_pre)
    pre_ztest_dict[topic_label] = ztest_vals_pre

    ztest_vals_post = ztest(pre[topic_label], forecast_pre['yhat'], percent_post)
    post_ztest_dict[topic_label] = ztest_vals_post
    return ztest_vals_post, pre_ztest_dict[topic_label], post_ztest_dict[topic_label]

def predict_topic_trend_and_plot_significant_differences(df, df2, topic_forecasts_plots_output, ztest_output, periods=16, frequency="MS", timestamp='2020-03-01'):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    pre_ztest_dict = {}
    post_ztest_dict = {}
    for i in range(df.shape[1]):
        ax.clear()
        topic_label = df.iloc[:, i].name
        #train a prophet model
        m = Prophet()
        forecast = prophet_projection(df, df2, topic_label, i, m, periods, frequency)
        #do statistical analysis (find percent of values outside the CI and do a z-test on the forecasted values compared to actual values)
        ztest_vals_post, pre_ztest_dict[topic_label], post_ztest_dict[topic_label] = projection_percent_outside_ci_and_ztest(forecast, df2, topic_label, pre_ztest_dict, post_ztest_dict)

        if ztest_vals_post[1] < 0.05:
            fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
            ax.plot(df2.iloc[:, i], color='k')
            ax = fig.gca()
            ax.set_title(f'{topic_label} Forecast', fontsize=20)
            plt.axvline(pd.Timestamp(timestamp),color='r')
            fig1.savefig(f'{topic_forecasts_plots_output}/{topic_label}_Prediction_Plot.png')

    pre_ztest_df = pd.DataFrame.from_dict(pre_ztest_dict, orient='index', columns=['Z Statistic Pre', 'P-Value Pre'])
    post_ztest_df = pd.DataFrame.from_dict(post_ztest_dict, orient='index', columns=['Z Statistic Post', 'P-Value Post'])
    ztest_df = pd.merge(pre_ztest_df, post_ztest_df, left_index=True, right_index=True)
    ztest_df = ztest_df[['Z Statistic Pre', 'Z Statistic Post', 'P-Value Pre', 'P-Value Post']]
    ztest_df.to_csv(ztest_output)