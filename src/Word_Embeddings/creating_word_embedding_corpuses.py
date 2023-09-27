import pandas as pd
import compress_json
import os 
import argparse
from text_utils import process_df, missing_text, clean_posts

def get_args():
    parser = argparse.ArgumentParser("Create dataframes of each subreddit")
    parser.add_argument("--path", default="/home/daphnaspira/birthing_experiences/data/original-reddit/subreddits")
    parser.add_argument("--output_each_subreddit", default="../data/subreddits_word_embeddings")
    args = parser.parse_args()
    print(args)
    return args

def create_dataframe(path, output_each_subreddit):
    subreddits = ("BabyBumps", "beyondthebump", "BirthStories", "daddit", "Mommit", "predaddit", "pregnant", "NewParents", "InfertilityBabies")
    for subreddit in subreddits:
        df = f"{subreddit}_df"
        df = pd.DataFrame()
        for file in os.listdir(f"{path}/{subreddit}/submissions/"):
            post = f"{path}/{subreddit}/submissions/{file}"
            if os.path.getsize(post) > 55:
                content = pd.read_json(post)
                df = df.append(content)

        df = process_df(df)
        df = missing_text(df, subreddit)

        df.reset_index(drop=True, inplace=True)
        df_j = df.to_json() 
        compress_json.dump(df_j, f"{output_each_subreddit}/{subreddit}_all_df.json.gz")

def main():
	args = get_args()

	if not os.path.exists(args.output_each_subreddit):
		os.mkdir(args.output_each_subreddit)

	create_dataframe(args.path, args.output_each_subreddit)

if __name__ == "__main__":
    main()