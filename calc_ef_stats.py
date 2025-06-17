import scipy.stats as stats
import statsmodels.stats.multitest as smt
import numpy as np
from matplotlib import pyplot as plt

def calc_ef_stats(z_time2, working_dir):
    z_time2 = z_time2.copy()
    z_time2 = z_time2.dropna()
    t_statistic_Flanker, p_value_Flanker = stats.ttest_1samp(z_time2['FlankerSU'], popmean=0, nan_policy='raise')
    t_statistic_DCCS, p_value_DCCS = stats.ttest_1samp(z_time2['DCSU'], popmean=0, nan_policy='raise')

    p_array = np.array([p_value_Flanker, p_value_DCCS])
    reject_val_allsubjs, pvals_corrected_allsubjs, a1_f, a2_f = smt.multipletests(p_array, alpha=0.05, method='fdr_bh')

    # Add 'sex' column: 0 if participant_id even, 1 if odd
    z_time2['sex'] = z_time2['participant_id'] % 2

    # Split data by sex
    flanker_male = z_time2.loc[z_time2['sex'] == 1, 'FlankerSU']
    flanker_female = z_time2.loc[z_time2['sex'] == 0, 'FlankerSU']
    dcsu_male = z_time2.loc[z_time2['sex'] == 1, 'DCSU']
    dcsu_female = z_time2.loc[z_time2['sex'] == 0, 'DCSU']

    # Run independent t-tests
    t_flanker, p_flanker = stats.ttest_ind(flanker_male, flanker_female, nan_policy='omit')
    t_dcsu, p_dcsu = stats.ttest_ind(dcsu_male, dcsu_female, nan_policy='omit')

    # Collect p-values
    pvals = [p_flanker, p_dcsu]

    # Apply Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = smt.multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Print results
    print(f"Sex effect Flanker p-value: {p_flanker:.2f}, corrected p-value: {pvals_corrected[0]:.2f}, significant: {reject[0]}")
    print(f"Sex effect DCSU p-value: {p_dcsu:.2f}, corrected p-value: {pvals_corrected[1]:.2f}, significant: {reject[1]}")

    def plot_distributions(data_male, data_female, measure_name):
        combined_data = np.concatenate([data_male, data_female])

        # Compute common bins across all data
        num_bins = 20
        min_val = np.min(combined_data)
        max_val = np.max(combined_data)
        bins = np.linspace(min_val, max_val, num_bins + 1)  # ensures consistent bin edges

        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True)

        # plot male vs female
        axes[0].hist(data_female, bins=bins, alpha=1.0, label='female', color='crimson', edgecolor='black')
        axes[0].hist(data_male, bins=bins, alpha=0.7, label='male', color='blue', edgecolor='black')
        axes[0].axvline(x=0, color='k', linestyle='dotted', linewidth=1.5)
        axes[0].set_title(f"{measure_name}: Male vs. Female")
        axes[0].set_xlabel('Z-score')
        axes[0].set_ylabel('Count')
        handles, labels = axes[0].get_legend_handles_labels()
        # Reorder the handles and labels
        order = [1, 0]  # This assumes the 'female' is first and 'male' is second
        axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=10, loc='upper left')

        # plot combined
        axes[1].hist(combined_data, bins=20, alpha=0.5, label='All Subjects', color='gray', edgecolor='black')
        axes[1].axvline(x=0, color='k', linestyle='dotted', linewidth=1.5)
        axes[1].set_title(f"{measure_name}: All Subjects")
        axes[1].set_xlabel('Z-score')
        axes[1].set_ylabel('Count')

        plt.tight_layout()

        plt.savefig(f'{measure_name}_Distribution.png', dpi=300, bbox_inches='tight')

        plt.show()

    plot_distributions(flanker_male.values, flanker_female.values, 'Flanker')
    plot_distributions(dcsu_male.values, dcsu_female.values, 'Dimensional Change Card Sort')

    return reject_val_allsubjs, pvals_corrected_allsubjs
