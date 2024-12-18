#####
# This program calculates correlations between post-COVID Z scores from the adolescent behavioral (cognitive and
# affective) and post-COVID Z scores from brain structural or functional measures.
######
import os

import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, linregress
import statsmodels.stats.multitest as smt

brain_meas = 'MEG'
working_dir = os.getcwd()
Z_scores_behav_filename=working_dir+'/Z_scores_all_meltzoff_cogn_behav_visit2.csv'

Z_time2_behav = pd.read_csv(Z_scores_behav_filename)

all_behav_list = Z_time2_behav.columns.tolist()
all_behav_list.remove('participant_id')
all_behav_list.remove('gender')

if brain_meas == 'MEG':

    # Load Z scores for MEG resting state activity for the four bands for post covid subjects
    MEG_dir = '/home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling'
    Z_alpha=pd.read_csv(f'{MEG_dir}/avgmeg/AvgMEG_wholebrain_alpha.csv')
    Z_beta=pd.read_csv(f'{MEG_dir}/avgmeg/AvgMEG_wholebrain_beta.csv')
    Z_theta=pd.read_csv(f'{MEG_dir}/avgmeg/AvgMEG_wholebrain_theta.csv')
    Z_gamma=pd.read_csv(f'{MEG_dir}/avgmeg/AvgMEG_wholebrain_gamma.csv')

    # Calculate correlations between behavior z scores and MEG power in each band

    pvals_df = pd.DataFrame()
    counter=0

    for b in ['alpha', 'beta', 'gamma', 'theta']:

        if b=='alpha':
            Z_meg = Z_alpha.copy()
        elif b=='beta':
            Z_meg = Z_beta.copy()
        elif b=='gamma':
            Z_meg = Z_gamma.copy()
        elif b=='theta':
            Z_meg = Z_theta.copy()

        Z_meg.drop(columns=['gender'], inplace=True)

        Z2_merged_orig = Z_meg.merge(Z_time2_behav, on='participant_id', how='inner')

        Z2_merged=Z2_merged_orig.copy()

        for behavior in all_behav_list:
            p_values = pearsonr(Z2_merged['avgmeg_' + b], Z2_merged[behavior])

            new_pvalcorr_row = pd.DataFrame({'band': b, 'behavior': behavior , 'p_uncorr': p_values[1]}, index=[counter])
            pvals_df = pd.concat([pvals_df, new_pvalcorr_row], ignore_index=True)
            counter+=1

    reject, pvals_corr, a1, a2 = smt.multipletests(pvals_df['p_uncorr'], alpha=0.025, method='fdr_bh')
    pvals_corr = pd.DataFrame(pvals_corr, columns=['p_corr'])
    pvals_df = pd.concat([pvals_df, pvals_corr], axis=1)


    p_allbands = {}
    for b in ['alpha', 'beta', 'gamma', 'theta']:

        if b == 'alpha':
            Z_meg = Z_alpha.copy()
        elif b == 'beta':
            Z_meg = Z_beta.copy()
        elif b == 'gamma':
            Z_meg = Z_gamma.copy()
        elif b == 'theta':
            Z_meg = Z_theta.copy()

        Z_meg.drop(columns=['gender'], inplace=True)

        Z2_merged_orig = Z_meg.merge(Z_time2_behav, on='participant_id', how='inner')

        Z2_merged = Z2_merged_orig.copy()

        for behavior in all_behav_list:
            output_pearson = pearsonr(Z2_merged['avgmeg_'+b], Z2_merged[behavior])

            plt.figure()

            # plot col of interest against cols in df_cols
            plt.scatter(Z2_merged['avgmeg_'+b], Z2_merged[behavior], s=12, color='k')

            # calculate linear regression
            slope, intercept, r, p, std_err = linregress(Z2_merged['avgmeg_'+b], Z2_merged[behavior])
            trendline = slope * Z2_merged['avgmeg_'+b] + intercept

            # plot trendline
            plt.plot(Z2_merged['avgmeg_'+b], trendline, color='k', linewidth=1, label='trendline')

            plt.xlabel('avgmeg_'+b, fontsize=12)
            plt.ylabel(behavior, fontsize=12)

            # make title
            corrected_pvalue = float(pvals_df.loc[(pvals_df['behavior']==behavior) & (pvals_df['band']==b),'p_corr'])
            plt.title(f' Z score post-COVID avg {behavior} vs Zscore post-COVID avg meg {b} band \nr = : '
                      f'{output_pearson[0]: .2f}, corrp = {corrected_pvalue:.3f}', fontsize=12)

            plt.show()

mystop=1