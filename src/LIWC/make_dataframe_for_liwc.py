import pandas as pd
import argparse
import xlsxwriter
from text_utils import prepare_data

def get_args():
    parser = argparse.ArgumentParser()
    #df with all birth stories
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    args = parser.parse_args()
    print(args)
    return args

def main():
	args = get_args()
	birth_stories_df = prepare_data(args.birth_stories_df)
	birth_stories_df = birth_stories_df[['id', 'Cleaned Submission']]
	print(birth_stories_df)

	#Convert to excel
	birth_stories_df.to_excel("birth_stories_and_ids.xlsx", engine='xlsxwriter')

if __name__ == "__main__":
    main()