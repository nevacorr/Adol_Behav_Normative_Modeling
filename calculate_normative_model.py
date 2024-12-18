from helper_functions import makenewdir, movefiles, create_design_matrix, create_dummy_design_matrix
from helper_functions import plot_data_with_spline, write_ages_to_file
import matplotlib.pyplot as plt
from predict_neva_from_normativepy import predict_neva
from pcntoolkit.normative import estimate
import shutil
import os
import pandas as pd
import numpy as np
from plot_z_scores import plot_and_compute_zcores_by_gender_with_nans
from apply_normative_model_time2 import apply_normative_model_time2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def calculate_normative_model_behav(struct_var, dem_behav_data_v1_orig, dem_behav_data_v2_orig, train_set_array,
     test_set_array, show_plots, spline_order, spline_knots, outputdirname, n_splits):


    Z2_all_splits = pd.DataFrame()

    for split in range(n_splits):

        subjects_train = train_set_array[split, :]
        subjects_test = test_set_array[split, :]

        dem_behav_data_v1 = dem_behav_data_v1_orig[dem_behav_data_v1_orig['subject'].isin(subjects_train)].copy()
        dem_behav_data_v2 = dem_behav_data_v2_orig[dem_behav_data_v2_orig['subject'].isin(subjects_test)].copy()
        dem_behav_data_v1.reset_index(drop=True, inplace=True)
        dem_behav_data_v2.reset_index(drop=True, inplace=True)

        makenewdir('{}/data/behav_models'.format(outputdirname))
        makenewdir('{}/data/covariate_files'.format(outputdirname))
        makenewdir('{}/data/response_files'.format(outputdirname))

        # create dataframes with subject age and gender to use as features
        feature_cols = ['agedays', 'agegroup', 'gender']
        X_train = dem_behav_data_v1[feature_cols].copy()
        X_train.reset_index(inplace=True, drop=True)
        X_test = dem_behav_data_v2[feature_cols].copy()
        X_test.reset_index(inplace=True, drop=True)

        # create response variables
        response_columns = ['peermindset', 'persmindset',
                            'needforapproval', 'needforbelonging', 'CDImean', 'RSQanxiety', 'RSQanger',
                            'Rejection', 'RejectCoping', 'RejectCopingSad', 'Coping_mad', 'Coping_sad', 'Coping_worried',
                            'StateAnxiety', 'TraitAnxiety', 'FlankerSU', 'DCSU', 'VocabSU', 'WMemorySU']

        y_train = dem_behav_data_v1[response_columns].copy()
        y_train.reset_index(inplace=True, drop=True)
        y_test = dem_behav_data_v2[response_columns].copy()
        y_test.reset_index(inplace=True, drop=True)

        agemin=X_train['agedays'].min()
        agemax=X_train['agedays'].max()
        write_ages_to_file(agemin, agemax, struct_var, outputdirname)

        # drop the age column because we want to use agedays as a predictor.
        X_train.drop(columns=['agegroup'], inplace=True)
        X_test.drop(columns=['agegroup'], inplace=True)

        ##########
        # Set up output directories. Save each behavior to its own text file, organized in separate directories,
        # because fpr each response variable Y (behavior) we fit a separate normative model
        ##########

        train_feature_indexes_with_nans = {}

        for c in y_train.columns:

            y_train_save = y_train.copy()
            X_train_save = X_train.copy()

            # If target column has nans, remove nans before saving to file
            if y_train_save[c].isnull().any():
                # Find index of nans
                ind_of_nans = y_train_save[c].loc[y_train_save[c].isna()].index
                # Save subject numbers that have nans
                train_feature_indexes_with_nans[c] = ind_of_nans
                # Remove nan values
                y_train_save.drop(index=ind_of_nans, inplace=True, axis=1)
                X_train_save.drop(index=ind_of_nans, inplace=True, axis=1)
                y_train_save.reset_index(drop=True, inplace=True)
                X_train_save.reset_index(drop=True, inplace=True)

            # Save only the data for this target to file in project directory. The filename has the target name in it.
            # The number of rows in this file depends on whether the target had nans in it
            y_train_save[c].to_csv(f'{outputdirname}/resp_tr_'+c+'.txt', header=False, index=False)
            # Save the data for the features to file in project directory. The filename has the target name in it.
            # The number of rows in this file depends on whether the target in this iteration had nans in it
            X_train_save.to_csv(f'{outputdirname}/cov_tr_'+c+'.txt', sep='\t', header=False, index=False)
            # Save features to file in project directory. The number of rows in this file depends on whether
            # the target for this iteration had nans in it.
            # Note that this file gets overwritten on every iteration through the targets. The number of rows is different
            # across different iterations. It will only save the data for the last iteration.
            # X_train_save.to_csv(f'{outputdirname}/cov_tr.txt', sep='\t', header=False, index=False)
            # Save the entire target dataframe which includes all behaviors to project directory.
            # NOTE: This overwrites itself on every loop. the number of rows in this file will depend on whether the
            # target for this loop iteration had nans in it.
            # Note that since some targets have nans, the number of rows will not be correct in this datafile for all
            # behaviors. It will only save the data for the last iteration.
            # y_train_save.to_csv(f'{outputdirname}/resp_tr.txt', sep='\t', header=False, index=False)

        for i in y_train.columns:
            # Make directory for model for this target
            behavdirname = '{}/data/behav_models/{}'.format(outputdirname,  i)
            makenewdir(behavdirname)
            # Recreate the filename for the target data that includes the target name. This is a
            # filename for this target that already exists and was created and saved to the project directory above.
            resp_tr_filename = "{}/resp_tr_{}.txt".format(outputdirname, i)
            # Make a new filename for this target that does not include the target name and has the path to the target model
            resp_tr_filepath = behavdirname + '/resp_tr.txt'
            # Copy the target data for this target to the model directory for this behavior and change the filename to not
            # include the target name
            shutil.copyfile(resp_tr_filename, resp_tr_filepath)
            # Recreate the filename for the feature data that does not include the target name. This is a filename
            # for this feature that already exists and was created and saved to the project directory above.
            cov_tr_filename = "{}/cov_tr_{}.txt".format(outputdirname, i)
            # Make a new filename for this feature matrix that does not include the target name and has the path to the
            # target model
            cov_tr_filepath = behavdirname + '/cov_tr.txt'
            # Copy the feature data for this target to the model directory for this target and change the filename to no
            # include the target name.
            shutil.copyfile(cov_tr_filename, cov_tr_filepath)

        movefiles(f"{outputdirname}/resp_*.txt", "{}/data/response_files/".format(outputdirname))
        movefiles(f"{outputdirname}/cov_t*.txt", "{}/data/covariate_files/".format(outputdirname))

        #  this path is where behavior models folders are located
        data_dir='{}/data/behav_models/'.format(outputdirname)

        behaviors = y_train.columns.tolist()

        # Create Design Matrix and add in spline basis and intercept
        create_design_matrix('train', agemin, agemax, spline_order, spline_knots, behaviors, data_dir)

        # Create pandas dataframes with header names to save evaluation metrics
        blr_metrics=pd.DataFrame(columns=['behav', 'MSLL', 'EV', 'SMSE','RMSE', 'Rho'])
        blr_site_metrics=pd.DataFrame(columns=['behav', 'y_mean','y_var', 'yhat_mean','yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

        # create dataframe with subject numbers to put the Z scores  in
        subjects_train = subjects_train.reshape(-1,1)
        Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

        # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few specific arguments that are worth commenting on:
        # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
        # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
        # ●savemodel=True: do not write out the final estimated model to disk
        # ●saveoutput=False: return the outputs directly rather than writing them to disk
        # ●standardize=False: do not standardize the covariates or response variable

        # Loop through behavs

        for behav in behaviors:
            print('Running behav:', behav)
            behav_dir=os.path.join(data_dir, behav)
            model_dir = os.path.join(data_dir, behav, 'Models')
            os.chdir(behav_dir)

            # configure the covariates to use. Change *_bspline_* to *_int_*
            cov_file_tr=os.path.join(behav_dir, 'cov_bspline_tr.txt')

            # load train & test response files
            resp_file_tr=os.path.join(behav_dir, 'resp_tr.txt')

            # run a basic model on the training dataset and store the predicted response (yhat_tr), the variance of the
            # predicted response (s2_tr), the model parameters (nm), the  Zscores for the train data, and other
            #various metrics (metrics_tr)
            yhat_tr, s2_tr, nm_tr, Z_tr, metrics_tr = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_tr,
                                                        testcov=cov_file_tr, alg='blr', optimizer='powell',
                                                        savemodel=True, saveoutput=False,standardize=False)
            Rho_tr=metrics_tr['Rho']
            EV_tr=metrics_tr['EXPV']

            #create dummy design matrices
            dummy_cov_file_path_female, dummy_cov_file_path_male = \
                create_dummy_design_matrix(behav, agemin, agemax, cov_file_tr, spline_order, spline_knots, outputdirname)

            #compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
            plot_data_with_spline('Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                                  dummy_cov_file_path_male, model_dir, behav, show_plots, outputdirname)

            plt.show()

        Z_time2 = apply_normative_model_time2(dem_behav_data_v2, outputdirname, struct_var, behaviors, spline_order, spline_knots,
                                    show_plots)

        Z_time2['split'] = split

        Z2_all_splits = pd.concat([Z2_all_splits, Z_time2], ignore_index=True)

    return Z2_all_splits, behaviors
