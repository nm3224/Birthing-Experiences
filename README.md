# Discovering Changes in Birthing Narratives during COVID-19

Set up and activate the conda environment by running the following lines:
```
conda env create -f new_environment.yml
conda activate new_environment
```
### Structure of repo

- `data/` - all data is saved here. Our original Reddit data is saved in the `original-reddit/` folder.
- `results/` - all results and figures are saved here. We have 4 results folders: `Corpus_Stats_Plots/`, `LIWC_Results/`, `Personas_Results/`, and `Topic_Modeling_Results/`. Information about these folders are contained in their readmes. 
- `src/` - all coding files are saved here. Instructions to run the files to replicate results are contained in their respective readmes. Our methods included counting persona frequencies, conducting sentiment analysis, running LIWC on our corpus, and running statistical analyses such as t-tests and z-tests. We also computed 95% confidence intervals for some of our data to further support the statistical significance of our results.
- `poster-winlp.pdf` & `Poster_Presentation.png` - this work was presented as a poster at the Widening Natural Language Processing Workshop at the 2021 Empirical Methods in Natural Language Processing Conference, in the Dominican Republic.
- `arXiv paper` can be found [**here:**](https://arxiv.org/abs/2204.11742)

<br> 

<img src="reddit-logo.png" width="200" height="70">
