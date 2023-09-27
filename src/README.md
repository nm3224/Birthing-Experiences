# Python Files

## Instructions to run files:
Run python files from `src`, not inner directories. Call `python -m` and the path to the `.py` file with "." instead of "/" in between directories. This allows the files to reference the utility functions in `src`.

**Example**: call `persona_stats.py` by running `python -m Personas.persona_stats` in the `src` directory.

**Path to corpus:**`birthing_experiences/src/birth_stories_df.json.gz`: compressed json file containing the dataframe of our corpus.

**Path to pre- and during COVID corpuses:** `birthing_experiences/src/pre_covid_posts_df.json.gz` and `birthing_experiences/src/post_covid_posts_df.json.gz` are compressed json files containing dataframes of the posts made before and after (respectively) March 11, 2020.

### Folders for each analysis method:
- `Corpus_Information/`: Statistics about the corpus, including data about the subreddits and statistics comparing number of posts made over time.
- `LIWC/`: Analyzing appearance of LIWC features in our corpus.
- `Personas/`: Analyzing persona frequency pre- and post-COVID.
- `Sentiment/`: Analyzing post sentiment pre- and post-COVID across several different categories of birthing experiences.
- `Topic_Modeling/`: Analyzing topic probability over time for 50 topics and comparing the forecasted probability during COVID to actual probability trends during COVID.
- `Word_Embeddings/`: Using Gensim's Word2Vec to create models and compare differences in vocabulary of each subreddit.
- `notebooks/`: Jupyter notebooks go here.

### Files creating and labeling the various corpuses:
- `covid_eras_and_posts_per_covid_month.py`: plots bar graph of number of posts made during each month of COVID and generates four dataframes of posts made during each of four pandemic "eras":
  -   March 11, 2020-June 1, 2020 (first wave)
  -   June 1, 2020-November 1, 2020 (dip in cases)
  -   November 1, 2020-April 1, 2021 (second wave)
  -   April 1, 2021-June 24, 2021 (widespread vaccine availability in US, dip in cases)
 - `labeling_stories.py`: re-implements Maria's code for Table 3: assigns labels to stories based on lexicon of key words, finds number of stories assigned each label. Also assigns "COVID" label to posts made after March 11, 2020, when COVID-19 was declared a pandemic by WHO, and separates the pre- and post-pandemic stories into two dataframes. Requires both the `labels_ngrams.json` and `covid_ngrams.json` dictionaries saved in `data/`.
 - `subreddit_dfs.py`: compiles all the submissions about birth stories that are 500+ words from all nine subreddits into one dataframe called birth_stories_df, incorporates author's first comment for empty submissions, and saves it as a compressed json file. We incorporated the author's first comment for submissions where the text field was empty because we noticed that in recent years, many birth story submissions will just contain a picture of the baby and the author will add the story as a comment on their own post.
 
### Utility files:
- `date_utils.py `: functions used to access date information about posts.
- `plots_utils.py`: functions used to make figures.
- `sentiment_utils.py`: functions used for sentiment analysis.
- `stats_utils.py`: functions for computing confidence intervals, Z-test scores and t-test scores.
- `text_utils.py`: functions used to load jsons and process text.
- `topic_utils.py`: functions used for topic modeling and FB Prophet forecast projection.
