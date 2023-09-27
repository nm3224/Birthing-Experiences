import argparse
import pandas as pd
import os
from date_utils import combine_topics_and_months, pre_covid_posts, posts_2019_on
from topic_utils import predict_topic_trend_and_plot_significant_differences

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--liwc_scores", default="/home/daphnaspira/birthing_experiences/data/LIWC2015_results_birth_stories_and_ids.csv", help="path to df with liwc scores for each story")
	parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
	parser.add_argument("--topic_forecasts_plots_output_all", default="../data/LIWC_Data/LIWC_Forecasts_2011_2021", help="path to where liwc forecast plots are saved")
	parser.add_argument("--ztest_output_all", default="../data/LIWC_Data/Z_Test_Stats_LIWC_2011_2021.csv", help="path to where ztest scores are saved")
	args = parser.parse_args()
	return args

def main():
	args=get_args()

	liwc_df = pd.read_csv(args.liwc_scores)
	dates_topics_df = combine_topics_and_months(args.birth_stories_df, liwc_df, period='M', drop=False)
	
	if not os.path.exists(args.topic_forecasts_plots_output_all):
		os.mkdir(args.topic_forecasts_plots_output_all)

	pre_covid = pre_covid_posts(dates_topics_df)

	predict_topic_trend_and_plot_significant_differences(pre_covid, dates_topics_df, args.topic_forecasts_plots_output, args.ztest_output)

if __name__ == "__main__":
    main()