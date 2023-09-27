# Persona Mention Frequencies
To replicate these results, first run `personas_frequencies.py` and then run `persona_stats` from the `src` directory in terminal (full instructions in the `src/` readme). Requires the `personas_ngrams.json` file saved in `data/Personas_Data/`.

`personas_frequencies.py`:
  - Takes in birth stories and a dictionary of n-grams mapped to specific personas. 
  - Iterates for different time periods:
      -  Before COVID
      -  During COVID
      -  Each of four different "eras" within COVID
    -  Splits the stories into ten equal "chunks."
    -  Counts mentions of each persona in each chunk.
    -  Computes statistics about the mentions of personas in the corpus. 
  -  Normalizes the before-COVID numbers to account for a higher overall average story length in the pre-COVID dataset
  -  Compares the mentions before and during COVID using a t-test to determine if the differences for each persona are statistically significant/
  -  We use a significance cutoff of .05.
  -  Plots the mention frequency for each persona over the course of the average story
    -  One set of plots compares before and during COVID
    -  The other set of plots compares before COVID and each of the four "eras" of COVID
 
`persona_stats.py`:
- Computes 95% confidence intervals for all the personas using Welch's t-interval.
- If 0 exists between the upper and lower bounds of the confidence interval, the difference is not significant.  
- If the lower and upper bounds are negative--this means the pre-COVID-19 persona frequency on average was LESS than the post-COVID-19 persona frequency.
- If the lower and upper bounds are positive--this means the pre-COVID-19 persona frequency on average was MORE than the post-COVID-19 persona frequency. 
