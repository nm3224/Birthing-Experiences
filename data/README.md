# Data
## Structure:

### Folder containing the original downloaded Reddit data:
- `original-reddit/`: the original reddit data collected using Keith Harrigian's [Reddit Retriever](https://github.com/kharrigian/retriever) is saved here (full path: `original-reddit/subreddits/{subreddit_name}`). Do not push the data from that directory to the remote server.

### Folders for data for each analysis method:
- `Personas_Data/`: contains data related to persona analysis.
- `Topic_Modeling_Data/`: contains data related to topic modeling.

### Dictionaries of terms used to label posts and data on the number of posts with each label:
- `covid_ngrams.json`: dictionary with the n-grams that map to the "COVID" label used to label posts as mentioning COVID-19.
- `labels_ngrams.json`: dictionary with labels used to label the posts based on their titles and the n-grams that map to each label.
- `label_counts_df.csv`: table with number of counts for each label.
