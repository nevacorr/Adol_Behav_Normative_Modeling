
import pandas as pd

def load_data_aff_cog():
    # Load MPF, cortical thickness gm, affective behavior and social media data
    genz_data_combined = pd.read_csv(
        '/home/toddr/neva/PycharmProjects/data_dir/MPF_CT_GMV_AFFB_SM_data_combined_18Sep2023.csv')

    #remove row with useless index
    genz_data_combined.drop(columns=['Unnamed: 0'], inplace=True)

    # remove rows with no behavioral or brain structural data
    genz_data_combined = genz_data_combined.dropna(subset=['gender'])

    #load cognitive behavior data for both visits
    cogdat_v1 = pd.read_csv(
        '/home/toddr/neva/PycharmProjects/Adolescent_Brain_Behavior_Longitudinal_Analysis/'
        'T1_GenZ_Cognitive_Data_for_corr_with_behav.csv')
    cogdat_v1.rename({'Subject':'subject', 'AgeGrp':'agegroup', 'FlankerStandardUncorrected':
        'FlankerSU', 'DCStandardUncorrected':'DCSU',
        'VocabStandardUncorrected':'VocabSU', 'WMemoryStandardUncorrected':
        'WMemorySU'}, axis=1, inplace=True)
    cogdat_v1['visit'] = 1

    cogdat_v2 = pd.read_csv(
        '/home/toddr/neva/PycharmProjects/Adolescent_Brain_Behavior_Longitudinal_Analysis/'
        'T2_GenZ_Cognitive_Data_for_corr_with_behav.csv')
    cogdat_v2.rename({'Subject': 'subject', 'AgeGrp': 'agegroup', 'FlankerStandardUncorrected':
        'FlankerSU', 'DCStandardUncorrected':'DCSU',
        'VocabStandardUncorrected':'VocabSU', 'WMemoryStandardUncorrected':
        'WMemorySU'}, axis=1, inplace=True)
    cogdat_v2['visit'] = 2
    #remove raw and age corrected scores for cognitive data
    remove_cols = [x for x in cogdat_v2.columns if ('Raw' in x) or ( 'AgeCorrected' in x)]
    cogdat_v2.drop(columns=remove_cols, inplace=True)
    remove_cols = [x for x in cogdat_v1.columns if ('Raw' in x) or ( 'AgeCorrected' in x)]
    cogdat_v1.drop(columns=remove_cols, inplace=True)

    #concat cognitive data from two timepoints
    cogdat = pd.concat([cogdat_v1, cogdat_v2], ignore_index=True)

    #merge cognitive data with brain/affectbehav/socialmedia dataframe
    complete_df = pd.merge(genz_data_combined, cogdat, how='left', on=['subject', 'agegroup', 'visit'])

    # convert gender, agegroup and agemonths columns from float to int
    complete_df['gender'] = complete_df['gender'].astype('int64')
    complete_df['agegroup'] = complete_df['agegroup'].astype('int64')
    complete_df['agemonths'] = complete_df['agemonths'].astype('int64')
    complete_df['agedays'] = complete_df['agedays'].astype('int64')

    return complete_df
