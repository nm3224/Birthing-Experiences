import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import redditcleaner
import argparse 
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from text_utils import load_subreddits, load_data_bf

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--path", default="/home/daphnaspira/birthing_experiences/data/subreddits_word_embeddings/", help="path to where the subreddits are stored", type=str)
    parser.add_argument("--subreddit", default="BabyBumps_all_df.json.gz", help="name of the file with the posts of the subreddit to clean and model", type=str)
    parser.add_argument("--model", default="../data/word2vec_models", help="path to where to save model", type=str)
    parser.add_argument("--tmpfile", default="../data/word2vec.model", help="path to temp file for gensim model", type=str)
    args = parser.parse_args()
    return args  

def clean_text(df):
	corpus = []
	for story in df['selftext']:
		lower = str(story).lower()
		text = redditcleaner.clean(lower)
		tokenized = nltk.sent_tokenize(text)
		for string in tokenized:
			word_tokenized = nltk.word_tokenize(string)
			corpus.append(word_tokenized)
	return corpus

def train_model(corpus, df_name, tmpfile, model_path):
	path = get_tmpfile(tmpfile)
	model = Word2Vec(sentences=corpus, min_count=2, workers=4)
	model.save(f"{model_path}/{df_name}_word2vec.model")

def main():
	args = get_args()

	df = load_data_bf(f"{args.path}{args.subreddit}")
	df_name = args.subreddit.split("_")[0]

	if not os.path.exists(args.model):
		os.mkdir(args.model)

	corpus = clean_text(df)
	train_model(corpus, df_name, args.tmpfile, args.model)

if __name__ == '__main__':
	main()
