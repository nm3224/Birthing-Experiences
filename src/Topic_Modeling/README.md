# Topic Modeling
To replicate results, first run `train_topics.py` to train LDA topic models on _k_ number of topics and then run `topic_projections_over_time.py` to forecast FB Prophet projections for each topic and compute z-tests to determine statistical significance (full instructions for running code in `src` readme).

`topic_projections_over_time.py`:
  - loads topic keys and dataframe of the distribution of topic probabilities for each topic for each story
  - combines topic probabilities with the date the story was posted
  - groups stories by month and finds the average topic probability for stories that month
  - for each topic:
    - trains a Prophet model on topic probabilities for each month until (not including) March 2020
    - uses the Prophet tool to forecast projections of topic probabilities for every month from March 2020 until June 2021
    - computes the percentage of data points that fall outside the projection's 95% confidence interval
    - runs a z-test on the pre-COVID and during-COVID data to determine if the difference between the forecasted data and the actual data is statistically significant
    - if the p-value of the z-test on the during-COVID data is less than 0.05, (difference is statistically significant) plots the forecasted data and the actual data over time.
    - saves the results of the z-tests for both the pre- and during-COVID datasets as csvs.

`train_topics.py`:
  - for each _k_ number of topics, trains an LDA topic model and computes the c_v coherence score for that number of topics
  - plots coherence scores and returns the number of topics that has the highest coherence score
