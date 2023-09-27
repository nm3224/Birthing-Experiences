import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, pearsonr
import warnings
warnings.filterwarnings("ignore")

def ztest(actual, forecast, percent):
    residual = actual - forecast
    residual = list(residual)

    #compute correlation between data points for pre-covid data (using pearsonr)
    corr = pearsonr(residual[:-1], residual[1:])[0]

    #plug correlation as r into z test (function from biester)
    #calculate z test for pre and post covid data
    z = (percent - 0.05) / np.sqrt((0.05*(1-0.05))/len(actual))

    #find p-value
    pval = norm.sf(np.abs(z))
    return z, pval

#performs the t-test
def ttest(df, df2, chunks=False, persona_chunk_stats_output=None, persona_stats_output=None):
    stat=[]
    p_value=[]
    index = []
    if chunks==True:
        for i in range(df.shape[1]):
            chunk = i
            pre_chunk = df[i::10]
            post_chunk = df2[i::10]
            for j in range(df.shape[1]):
                persona_name = pre_chunk.iloc[:, j].name
                pre_chunk1 = pre_chunk.iloc[:, j]
                post_chunk1 = post_chunk.iloc[:, j]
                ttest = stats.ttest_ind(pre_chunk1, post_chunk1)
                stat.append(ttest.statistic)
                p_value.append(ttest.pvalue)
                index.append(persona_name)
        ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
        ttest_df.to_csv(persona_chunk_stats_output)
    else:
        for k in range(df.shape[1]):
            persona_name = df.iloc[:, k].name
            pre_covid = df.iloc[:, k]
            post_covid = df2.iloc[:, k]
            ttest = stats.ttest_ind(pre_covid, post_covid)
            stat.append(ttest.statistic)
            p_value.append(ttest.pvalue)
            index.append(persona_name)
            print(f"{persona_name} t-statistic: {ttest.statistic}, p-value: {ttest.pvalue}")
        
        ttest_df = pd.DataFrame(data = {'T-test statistic': stat, 'P-Values': p_value}, index = index)
        ttest_df.to_csv(persona_stats_output)

def compute_confidence_interval(personas, pre_df, post_df, puncts):
    lowers = []
    uppers = []
    personas_sigs = []
    diff = []
    for persona in personas:
        if persona not in puncts:
            x1 = post_df[persona]
            x2 = pre_df[persona]

            alpha = 0.05                                                      
            n1, n2 = len(x1), len(x2)                                          
            s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)  

            #print(f'ratio of sample variances: {s1**2/s2**2}')

            s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))  
            t = stats.t.ppf(1 - alpha/2, df)  

            d = (np.mean(x1) - np.mean(x2))
            lower = (np.mean(x1) - np.mean(x2)) - t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
            upper = (np.mean(x1) - np.mean(x2)) + t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s

            x = False 
            if lower < 0 and upper > 0:
                x = True 
            if x == False or len(puncts) == 0:
                lowers.append(lower)
                uppers.append(upper)
                personas_sigs.append(persona)
                diff.append(d)

    df = pd.DataFrame({'Lower Bound': lowers, 'Upper Bound': uppers, 'Difference in Sample Means: Post - Pre': diff}, index = personas_sigs)
    return df