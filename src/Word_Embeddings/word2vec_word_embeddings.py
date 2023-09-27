import gensim
from gensim.models import Word2Vec
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--model", default="/home/daphnaspira/birthing_experiences/data/word2vec_models/", help="path to where to save model", type=str)
    args = parser.parse_args()
    return args

def compare_words(word, model):
	model1 = Word2Vec.load(f"{model}/BabyBumps_word2vec.model")
	#similar1 = model1.wv.most_similar(word, topn=10)
	#print(similar1)

	model2 = Word2Vec.load(f"{model}/beyondthebump_word2vec.model")


	import pdb; pdb.set_trace()
	#similar2 = model2.wv.most_similar(word, topn=10)
	#print(similar2)

	len(set(model1.wv.index_to_key).intersection(set(model2.wv.index_to_key)))

def main():
	args = get_args()

	compare_words('baby', args.model)

if __name__ == '__main__':
	main()