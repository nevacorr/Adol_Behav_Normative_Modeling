import os
import pandas as pd
import numpy as np
from helper_functions_demographics import get_agegroup, summarize_visit, print_summaries

struct_var = 'fa_and_md_and_mpf'
working_dir = os.getcwd()
n_splits = 100
conv_days_to_years = 365.25

behav_score_cols = ['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']

all_subjects_filename = f'{working_dir}/visit1andvisit2_subjects_behav.csv'
all_subjects = pd.read_csv(all_subjects_filename)

nan_columns = all_subjects.columns[all_subjects.isna().any()].tolist()

# remove all rows that have nan values in all of the behav score cols
all_subjects.dropna(subset=behav_score_cols, how='all', inplace=True)
all_subjects.drop(columns=behav_score_cols, inplace=True)

all_subjects['agegroup'] = all_subjects.apply(get_agegroup, axis=1)

visit1_subjects = all_subjects[all_subjects['visit']==1]
visit2_subjects = all_subjects[all_subjects['visit']==2]

v1_agegroup_gender_summary = summarize_visit(visit1_subjects, conv_days_to_years)
v2_agegroup_gender_summary = summarize_visit(visit2_subjects, conv_days_to_years)

print_summaries(v1_agegroup_gender_summary, visit_name="Visit 1")
print_summaries(v2_agegroup_gender_summary, visit_name="Visit 2")

