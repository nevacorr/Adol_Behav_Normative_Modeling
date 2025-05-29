import scipy.stats as stats
import statsmodels.stats.multitest as smt
import numpy as np

def calc_ef_stats(z_time2):
    z_time2 = z_time2.dropna()
    t_statistic_Flanker, p_value_Flanker = stats.ttest_1samp(z_time2['FlankerSU'], popmean=0, nan_policy='raise')
    t_statistic_DCCS, p_value_DCCS = stats.ttest_1samp(z_time2['DCSU'], popmean=0, nan_policy='raise')

    p_array = np.array([p_value_Flanker, p_value_DCCS])
    reject_f, pvals_corrected, a1_f, a2_f = smt.multipletests(p_array, alpha=0.05, method='fdr_bh')
