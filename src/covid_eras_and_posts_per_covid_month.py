import pandas as pd
import compress_json
import json
import argparse
from date_utils import pandemic_eras, get_post_month, convert_datetime
from text_utils import story_lengths, load_data
from plots_utils import plot_bar_graph

def get_args():
    parser = argparse.ArgumentParser()

    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_posts_df", default="/home/nm3224/birthing_experiences/src/relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_posts_df", default="/home/nm3224/birthing_experiences/src/relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labels_df", default="/home/nm3224/birthing_experiences/src/relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)

    #New data to create
    parser.add_argument("--mar_june_2020_df", default="../data/covid_era_jsons/mar_june_2020_df.json.gz", help="path to df of the stories from COVID era 1", type=str)
    parser.add_argument("--june_nov_2020_df", default="../data/covid_era_jsons/june_nov_2020_df.json.gz", help="path to df of the stories from COVID era 2", type=str)
    parser.add_argument("--nov_2020_apr_2021_df", default="../data/covid_era_jsons/nov_2020_apr_2021_df.json.gz", help="path to df of the stories from COVID era 3", type=str)
    parser.add_argument("--apr_june_2021_df", default="../data/covid_era_jsons/apr_june_2021_df.json.gz", help="path to df of the stories from COVID era 4", type=str)
    parser.add_argument("--bar_graph_output", default="../data/Corpus_Stats_Plots/Posts_per_Month_Covid_bar.png", help="bar graph of number of posts made each month of the pandemic", type=str)
    args = parser.parse_args()
    return args
    
#Splits df into four eras of covid
def four_eras(post_covid_posts_df):
    post_covid_posts_df['Mar 11-June 1 2020'] = post_covid_posts_df['year-month'].apply(pandemic_eras, args = ('2020-03', '2020-06'))
    post_covid_posts_df['June 1-Nov 1 2020'] = post_covid_posts_df['year-month'].apply(pandemic_eras, args =('2020-06', '2020-11'))
    post_covid_posts_df['Nov 1 2020-Apr 1 2021'] = post_covid_posts_df['year-month'].apply(pandemic_eras, args = ('2020-11', '2021-04'))
    post_covid_posts_df['Apr 1-June 24 2021'] = post_covid_posts_df['year-month'].apply(pandemic_eras, args = ('2021-04', '2021-06'))

    mar_june_2020_df = post_covid_posts_df.get(post_covid_posts_df['Mar 11-June 1 2020']==True)
    june_nov_2020_df = post_covid_posts_df.get(post_covid_posts_df['June 1-Nov 1 2020']==True)
    nov_2020_apr_2021_df = post_covid_posts_df.get(post_covid_posts_df['Nov 1 2020-Apr 1 2021']==True)
    apr_june_2021_df = post_covid_posts_df.get(post_covid_posts_df['Apr 1-June 24 2021']==True)

    print(len(mar_june_2020_df), len(june_nov_2020_df), len(nov_2020_apr_2021_df), len(apr_june_2021_df))
    return mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df

def save_jsons(mj, jn, na, aj, mar_june_output, june_nov_output, nov_apr_output, apr_june_output):
    #Loads into Jsons
    mar_june_2020_df = mj.to_json()
    compress_json.dump(mar_june_2020_df, mar_june_output)

    june_nov_2020_df = jn.to_json()
    compress_json.dump(june_nov_2020_df, june_nov_output)

    nov_2020_apr_2021_df = na.to_json()
    compress_json.dump(nov_2020_apr_2021_df, nov_apr_output)

    apr_june_2021_df = aj.to_json()
    compress_json.dump(apr_june_2021_df, apr_june_output)

def main():
    args = get_args()

    labels_df, birth_stories_df, pre_covid_posts_df, post_covid_posts_df = load_data(args.birth_stories_df, args.pre_covid_posts_df, args.post_covid_posts_df, args.labels_df)

    pre_covid_posts_df.name = 'pre-covid'
    post_covid_posts_df.name = 'post-covid'

    post_covid_posts_df = convert_datetime(post_covid_posts_df)

    plot_bar_graph(post_covid_posts_df['year-month'], title="Posts per Month of COVID", bar_graph_output=args.bar_graph_output)

    mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df = four_eras(post_covid_posts_df)
    save_jsons(mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df, args.mar_june_2020_df, args.june_nov_2020_df, args.nov_2020_apr_2021_df, args.apr_june_2021_df)

if __name__ == '__main__':
    main()
