import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

def make_plots(pre_df, post_df=None, throughout=False, m_j_df=None, j_n_df=None, n_a_df=None, a_j_df=None, pre_post_plot_output_folder=None, throughout_covid_output_folder=None):
    if throughout==False:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid")
        if throughout==False:
            ax.plot(post_df.iloc[:,i], label = f"During Covid")
        else:
            ax.plot(m_j_df.iloc[:,i], label = f"March-June 2020")
            ax.plot(j_n_df.iloc[:,i], label = f"June-Nov. 2020")
            ax.plot(n_a_df.iloc[:,i], label = f"Nov. 2020-April 2021")
            ax.plot(a_j_df.iloc[:,i], label = f"April-June 2021")
        ax.set_title(f"{persona_label}", fontsize=22)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        if throughout==False:
            fig.savefig(f'{pre_post_plot_output_folder}{persona_label}_pre_post_frequency.png')
        else:
            fig.savefig(f'{throughout_covid_output_folder}{persona_label}_throughout_covid_frequency.png')

#Generates bar graph of number of posts made each month of the pandemic
def plot_bar_graph(series, name=None, title=None, bar_graph_output=None, path_output=None):
    posts_per_month = series.value_counts()
    fig = plt.figure(figsize=(20,10))
    posts_per_month.sort_index().plot.bar()
    if bar_graph_output != None:
        fig.suptitle(title)
        fig.savefig(bar_graph_output)
    if path_output != None:
        fig.suptitle(f'Number of posts in r/{str(name)} per year')
        fig.savefig(f'{path_output}{str(name)}_years.png')