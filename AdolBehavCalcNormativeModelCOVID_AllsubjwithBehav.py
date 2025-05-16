#####
# This program calculates normative models for the Meltzoff behavioral data for Genz study. It does not include
# the cognitive data which has missing values. This program does not have code to deal with missing values.
######

import pandas as pd
import matplotlib.pyplot as plt
from load_data_allbehav import load_data_aff_cog
import numpy as np
from calculate_normative_model import calculate_normative_model_behav
from helper_functions import makenewdir
from sklearn.model_selection import StratifiedShuffleSplit
import os
import shutil
from plot_and_compute_zdistributions import plot_by_gender_no_kde, plot_and_compute_zcores_by_gender


######Variables to specify#############
n_splits = 1
show_plots = 0  #set to 1 to show training and test data y vs yhat and spline fit plots. Set to 0 to save to file.
show_nsubject_plots = 0 #set to 1 to show number of subjects in analysis
spline_order = 1
spline_knots = 2
outputdirname = '/home/toddr/neva/PycharmProjects/AdolNormativeModelingCOVID'
struct_var = 'behav'
#######################################

############## Load brain structural measures and behavior data and reorganize  ############
all_data = load_data_aff_cog()
columns_to_keep_meltzoff = ['subject', 'visit', 'gender', 'agemonths', 'agedays', 'agegroup', 'peermindset', 'persmindset',
                   'needforapproval', 'needforbelonging', 'CDImean', 'RSQanxiety' ,'RSQanger',
                   'Rejection', 'RejectCoping', 'RejectCopingSad', 'Coping_mad', 'Coping_sad', 'Coping_worried',
                   'StateAnxiety', 'TraitAnxiety', 'FlankerSU','DCSU', 'VocabSU', 'WMemorySU']

#remove structural and social media data
dem_behav_data = all_data.loc[:, columns_to_keep_meltzoff].copy()

# Write subjects and nih toolkit scores to file
v1andv2_subjects_behav = dem_behav_data[['subject', 'visit', 'agedays', 'gender', 'FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']]
v1andv2_subjects_behav.to_csv('visit1andvisit2_subjects_behav.csv', index=False)

#replace gender codes 1=male 2=female with binary values (make male=1 and female=0)
dem_behav_data.loc[dem_behav_data['gender']==2, 'gender'] = 0

unique_subjects = dem_behav_data['subject'].value_counts()
unique_subjects = unique_subjects[unique_subjects == 1].index
subjects_with_one_dataset = dem_behav_data[dem_behav_data['subject'].isin(unique_subjects)]
subjects_visit1_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 1]
subjects_visit2_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 2]
subjects_v1_only = subjects_visit1_data_only['subject'].tolist()
subjects_v2_only = subjects_visit2_data_only['subject'].tolist()
all_subjects = dem_behav_data['subject'].unique().tolist()
all_subjects_2ts = [sub for sub in all_subjects if (sub not in subjects_v1_only and sub not in subjects_v2_only)]

#make separate dataframes for visit 1 and visit 2
dem_behav_data_v1 = dem_behav_data.loc[dem_behav_data['visit']==1]
dem_behav_data_v2 = dem_behav_data.loc[dem_behav_data['visit']==2]

#need to make cognitive numbers smaller or normative modeling function does not work
cog_columns = ['FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']
dem_behav_data_v1.loc[:,cog_columns] = dem_behav_data_v1.loc[:, cog_columns]/100.0
dem_behav_data_v2.loc[:,cog_columns] = dem_behav_data_v2.loc[:, cog_columns]/100.0

#check for columns with nan values
v1_has_nans = dem_behav_data_v1.isnull().sum()
v2_has_nans = dem_behav_data_v2.isnull().sum()

# Make n_splits train-test splits

df_for_train_test_split = dem_behav_data_v1.copy()

df_for_train_test_split['age_gender'] = (df_for_train_test_split['agegroup'].astype(str) + '_'
                                         + df_for_train_test_split['gender'].astype(str))

# For ttsplit, create a dataframe that has only visit 1 data and only subject number, visit, age and sex as columns
cols_to_keep = ['subject', 'visit', 'agegroup', 'gender', 'age_gender']
cols_to_drop = [col for col in df_for_train_test_split if col not in cols_to_keep]
df_for_train_test_split.drop(columns=cols_to_drop, inplace=True)
# keep only the subjects that have data at both time points
df_for_train_test_split = df_for_train_test_split[df_for_train_test_split['subject'].isin(all_subjects_2ts)]

# Initialize StratifiedShuffleSplit for equal train/test sizes
splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.54, random_state=42)

train_set_list = []
test_set_list = []
# Perform the splits
for i, (train_index, test_index) in enumerate(
        splitter.split(df_for_train_test_split, df_for_train_test_split['age_gender'])):
    train_set_list_tmp = df_for_train_test_split.iloc[train_index, 0].values.tolist()
    train_set_list_tmp.extend(subjects_v1_only)
    test_set_list_tmp = df_for_train_test_split.iloc[test_index, 0].values.tolist()
    test_set_list_tmp.extend(subjects_v2_only)
    train_set_list.append(train_set_list_tmp)
    test_set_list.append(test_set_list_tmp)

train_set_array = np.array(list(train_set_list))
test_set_array = np.array(list(test_set_list))

fname_train = '{}/visit1_subjects_train_sets_{}_splits_{}.txt'.format(outputdirname, n_splits, struct_var)
np.save(fname_train, train_set_array)

fname_test = '{}/visit1_subjects_test_sets_{}_splits_{}.txt'.format(outputdirname, n_splits, struct_var)
np.save(fname_test, test_set_array)

# make directories to store files for model creation
dirpath = os.path.join(outputdirname, 'data')
try:
    shutil.rmtree(dirpath)
    print(f"Directory '{dirpath}' and its contents have been removed.")
except FileNotFoundError:
    print(f"Directory '{dirpath}' does not exist.")
makenewdir('{}/data/'.format(outputdirname))
makenewdir('{}/data/plots'.format(outputdirname))

# make file diretories for model testing
dirpath = os.path.join(outputdirname, 'predict_files')
try:
    shutil.rmtree(dirpath)
    print(f"Directory '{dirpath}' and its contents have been removed.")
except FileNotFoundError:
    print(f"Directory '{dirpath}' does not exist.")
makenewdir('{}/predict_files/'.format(outputdirname))
makenewdir('{}/predict_files/plots'.format(outputdirname))

# Calculate Zscores for each response variable

Z_time1_behav, Z_time2_behav, behaviors = calculate_normative_model_behav(struct_var, dem_behav_data_v1, dem_behav_data_v2, train_set_array,
                                                test_set_array,show_plots, spline_order, spline_knots, outputdirname,
                                                n_splits)

Z_time1_avg_allsplits = Z_time1_behav.groupby(by=['participant_id']).mean().drop(columns=['split'])
Z_time1_avg_allsplits.reset_index(inplace=True)

Z_time2_avg_allsplits = Z_time2_behav.groupby(by=['participant_id']).mean().drop(columns=['split'])
Z_time2_avg_allsplits.reset_index(inplace=True)

plot_and_compute_zcores_by_gender(Z_time2_avg_allsplits, 'behavior', behaviors, outputdirname, n_splits)

Z_time1_avg_allsplits.to_csv(outputdirname+'/Z_scores_all_meltzoff_cogn_behav_visit1.csv')
Z_time2_avg_allsplits.to_csv(outputdirname+'/Z_scores_all_meltzoff_cogn_behav_visit2.csv')

plt.show()

mystop=1