# Word Embeddings in Different Subreddits
To replicate results, first run `creating_word_embedding_corpuses.py` to generate corpuses of all posts in each subreddit. Clean corpuses and train Word2Vec models with `word2vec_clean_and_train_model.py` and compare words across different models with `word2vec_word_embeddings.py`. Run all code from the `src` directory (full instructions in `src` readme).
- `creating_word_embedding_corpuses.py`: creates corpuses of all posts for each of 9 subreddits.
- `word2vec_clean_and_train_model.py`: takes in the name of a corpus file as a command line argument and cleans the corpus, trains a word2vec model for that corpus, and saves the model.
- `word2vec_word_embeddings.py`: compares most similar words for vocabulary between different subreddit models.
