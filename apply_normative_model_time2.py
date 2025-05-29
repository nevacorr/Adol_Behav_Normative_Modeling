from helper_functions import makenewdir, read_ages_from_file, movefiles, create_design_matrix
from helper_functions import create_dummy_design_matrix, plot_data_with_spline
import shutil
import pandas as pd
import os
import numpy as np
from normative_edited import predict
import matplotlib.pyplot as plt
import time

def apply_normative_model_time2(dem_behav_data_v2, filepath, struct_var, behaviors, spline_order, spline_knots,
                                show_plots, splitno, start_time):

    ############################  Apply Normative Model to Post-COVID Data ####################

    dem_behav_data_v2 = dem_behav_data_v2[dem_behav_data_v2['subject'] < 400]

    # make file diretories for output
    makenewdir('{}/predict_files/behav_models'.format(filepath))
    makenewdir('{}/predict_files/covariate_files'.format(filepath))
    makenewdir('{}/predict_files/response_files'.format(filepath))

    #read agemin and agemax from file
    agemin, agemax = read_ages_from_file(struct_var, filepath)

    # specify paths
    training_dir = f'{filepath}/data/behav_models/'
    out_dir = (f'{filepath}/predict_files/behav_models/')
    #  this path is where behavior_models folders are located
    predict_files_dir = (f'{filepath}/predict_files/behav_models/')

    X_test = dem_behav_data_v2[['agedays', 'gender']]
    y_test = dem_behav_data_v2.loc[:, behaviors]

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########

    test_feature_indexes_with_nans = {}

    for c in behaviors:

        y_test_save = y_test.copy()
        X_test_save = X_test.copy()

        # If target column has nans, remove nans before saving to file
        if y_test_save[c].isnull().any():
            # find indexes of nans
            ind_of_nans = y_test_save[c].loc[y_test_save[c].isna()].index
            # Save subject numbers that have nans
            test_feature_indexes_with_nans[c] = ind_of_nans
            # Remove nan values
            y_test_save.drop(index=ind_of_nans, inplace=True, axis=1)
            X_test_save.drop(index=ind_of_nans, inplace=True, axis=1)
            y_test_save.reset_index(drop=True, inplace=True)
            X_test_save.reset_index(drop=True, inplace=True)

        y_test_save[c].to_csv(f'{filepath}/resp_te_' + c + '.txt', header=False, index=False)
        X_test_save.to_csv(f'{filepath}/cov_te_' + c + '.txt', sep='\t', header=False, index=False)
        X_test_save.to_csv(f'/{filepath}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{filepath}/resp_te.txt', sep='\t', header=False, index=False)

    for i in behaviors:
        behavdirname = (f'{filepath}/predict_files/behav_models/{i}')
        makenewdir(behavdirname)
        resp_te_filename = f'{filepath}/resp_te_{i}.txt'
        resp_te_filepath = behavdirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filename = "{}/cov_te_{}.txt".format(filepath, i)
        cov_te_filepath = behavdirname + '/cov_te.txt'
        shutil.copyfile(cov_te_filename, cov_te_filepath)

    movefiles(f"{filepath}/resp_*.txt", f"{filepath}/predict_files/response_files/")
    movefiles(f"{filepath}/cov_t*.txt", f"{filepath}/predict_files/covariate_files/")

    # Create Design Matrix and add in spline basis and intercept
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, behaviors, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['subject'] = dem_behav_data_v2['subject'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    # create design matrices for all regions and save files in respective directories
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, behaviors, predict_files_dir)

    for behav in behaviors:
        print(f'Running behav {behav} apply model split {splitno}')
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60.0

        print(f"Elapsed time: {elapsed_time:.4f} minutes")

        behav_dir = os.path.join(predict_files_dir, behav)
        model_dir = os.path.join(training_dir, behav, 'Models')
        os.chdir(behav_dir)

        # configure the covariates to use.
        cov_file_te = os.path.join(behav_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te = os.path.join(behav_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        if behav in test_feature_indexes_with_nans:
            Znew = Z.copy()
            for pos in sorted(test_feature_indexes_with_nans[behav]):
                Znew = np.insert(Znew, pos, np.nan)
            Z_time2[behav] = Znew
        else:
            Z_time2[behav] = Z


        # create dummy design matrices
        dummy_cov_file_path_female, dummy_cov_file_path_male = \
            create_dummy_design_matrix(behav, agemin, agemax, cov_file_te, spline_order, spline_knots, filepath)

        # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
        plot_data_with_spline('Test Data',struct_var, cov_file_te, resp_file_te, dummy_cov_file_path_female,
                              dummy_cov_file_path_male, model_dir, behav, show_plots, filepath)

    Z_time2.rename(columns={'subject': 'participant_id'}, inplace=True)

    plt.show()

    return Z_time2
