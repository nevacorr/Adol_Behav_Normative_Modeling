#########
# This module creates the design matrix plus adds the intercept and spline basis functions for use with both the
# precovid and postcovid data sets.
# Created April 28, 2023
##########
import os
import numpy as np
from matplotlib import pyplot as plt
from pcntoolkit.normative import predict
import pandas as pd
import seaborn as sns
import shutil
import glob
from pcntoolkit.util.utils import create_bspline_basis
from scipy import stats
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

def write_list_to_file(mylist, filepath):
   with open(filepath, 'w') as file:
       for item in mylist:
           file.write(item + '\n')

def plot_num_subjs(df, title, struct_var, timept, path):
    # sns.set(font_scale=2)
    sns.set_style(style='white')
    g = sns.catplot(x="agegrp", hue="gender", data=df, kind="count", legend=False, palette=sns.color_palette(['blue', 'green']))
    g.fig.suptitle(title, fontsize=10)
    g.fig.subplots_adjust(top=0.85) # adjust the Figure
    g.ax.set_xlabel("Age", fontsize=10)
    g.ax.set_ylabel("Number of Subjects", fontsize=8)
    g.ax.tick_params(axis='x', labelsize=8)
    g.ax.tick_params(axis='y', labelsize=8)
    # g.fig.suptitle(title)
    hue_labels = ['male', 'female']
    g.add_legend(legend_data={
        key: value for key, value in zip(hue_labels, g._legend_data.values())
        }, fontsize=10)
    g.ax.set(yticks=np.arange(0,20,2))
    # g.set(title=title)
    plt.show(block=False)
    plt.savefig('{}/data/{}/plots/NumSubjects_{}'.format(path, struct_var, timept))
    #plt.close(g.fig)

def makenewdir(path):
    isExist = os.path.exists(path)
    if isExist is False:
        os.mkdir(path)
        print('made directory {}'.format(path))

def movefiles(pattern, folder):
    files = glob.glob(pattern)
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, folder +  file_name)
        print('moved:', file)

def create_design_matrix(datatype, agemin, agemax, spline_order, spline_knots, roi_ids, data_dir):
    B = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    for roi in roi_ids:
        print('Creating basis expansion for ROI:', roi)
        roi_dir = os.path.join(data_dir, roi)
        os.chdir(roi_dir)
        # create output dir
        makenewdir(f'{roi_dir}/blr')
        # os.makedirs(os.path.join(roi_dir, 'blr'), exist_ok=True)

        # load train & test covariate data matrices
        if datatype == 'train':
            # X = np.loadtxt(os.path.join(roi_dir, 'cov_tr.txt'))
            X = np.loadtxt(f'{roi_dir}/cov_tr.txt')
        elif datatype == 'test':
            # X = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))
            X = np.loadtxt(f'{roi_dir}/cov_te.txt')

        # add intercept column
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        if datatype == 'train':
            # np.savetxt(os.path.join(roi_dir, 'cov_int_tr.txt'), X)
            np.savetxt(f'{roi_dir}/cov_int_tr.txt', X)
        elif datatype == 'test':
            # np.savetxt(os.path.join(roi_dir, 'cov_int_te.txt'), X)
            np.savetxt(f'{roi_dir}/cov_int_te.txt', X)

        # create Bspline basis set
        # This creates a numpy array called Phi by applying function B to each element of the first column of X
        Phi = np.array([B(i) for i in X[:, 0]])
        X = np.concatenate((X, Phi), axis=1)
        if datatype == 'train':
            # np.savetxt(os.path.join(roi_dir, 'cov_bspline_tr.txt'), X)
            np.savetxt(f'{roi_dir}/cov_bspline_tr.txt', X)
        elif datatype == 'test':
            # np.savetxt(os.path.join(roi_dir, 'cov_bspline_te.txt'), X)
            np.savetxt(f'{roi_dir}/cov_bspline_te.txt', X)

#this function creates a dummy design matrix for plotting of spline function
def create_dummy_design_matrix(struct_var, agemin, agemax, cov_file, spline_order, spline_knots, outputdir):
    # load predictor variables for region
    X = np.loadtxt(cov_file)

    # make dummy test data covariate file starting with a column for age
    dummy_cov = np.linspace(agemin, agemax, num=1000)
    ones = np.ones((dummy_cov.shape[0], 1))

    # add a column for gender for male and female data
    dummy_cov_female = np.concatenate((dummy_cov.reshape(-1, 1), ones * 0), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov.reshape(-1, 1), ones), axis=1)

    #add a column for intercept
    dummy_cov_female = np.concatenate((dummy_cov_female, ones), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov_male, ones), axis=1)

    # create spline features and add them to male and female predictor dataframes
    BAll = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    Phidummy_f = np.array([BAll(i) for i in dummy_cov_female[:, 0]])
    Phidummy_m = np.array([BAll(i) for i in dummy_cov_male[:, 0]])
    dummy_cov_female = np.concatenate((dummy_cov_female, Phidummy_f), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov_male, Phidummy_m), axis=1)

    # write these new created predictor variables with spline and response variable to file
    dummy_cov_file_path_female = os.path.join(outputdir, 'cov_file_dummy_female.txt')
    np.savetxt(dummy_cov_file_path_female, dummy_cov_female)
    dummy_cov_file_path_male = os.path.join(outputdir, 'cov_file_dummy_male.txt')
    np.savetxt(dummy_cov_file_path_male, dummy_cov_male)
    return dummy_cov_file_path_female, dummy_cov_file_path_male


# this function plots  data with spline model superimposed, for both male and females
def plot_data_with_spline(datastr, struct_var, cov_file, resp_file, dummy_cov_file_path_female,
                              dummy_cov_file_path_male, model_dir, roi, showplots, outputdir):

    output_f = predict(dummy_cov_file_path_female, respfile=None, alg='blr', model_path=model_dir)

    output_m = predict(dummy_cov_file_path_male, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy_m=output_m[0]
    yhat_predict_dummy_f=output_f[0]

    # load real data predictor variables for region
    X = np.loadtxt(cov_file)
    # load real data response variables for region
    y = np.loadtxt(resp_file)

    # create dataframes for plotting with seaborn facetgrid objects
    dummy_cov_female = np.loadtxt(dummy_cov_file_path_female)
    dummy_cov_male = np.loadtxt(dummy_cov_file_path_male)
    df_origdata = pd.DataFrame(data=X[:, 0:2], columns=['Age in Days', 'gender'])
    df_origdata[struct_var] = y.tolist()
    df_origdata['Age in Days'] = df_origdata['Age in Days'] / 365.25
    df_estspline = pd.DataFrame(data=dummy_cov_female[:, 0].tolist() + dummy_cov_male[:, 0].tolist(),
                                columns=['Age in Days'])
    df_estspline['Age in Days'] = df_estspline['Age in Days'] / 365.25
    df_estspline['gender'] = [0] * 1000 + [1] * 1000
    df_estspline['gender'] = df_estspline['gender'].astype('float')
    tmp = np.array(yhat_predict_dummy_f.tolist() + yhat_predict_dummy_m.tolist(), dtype=float)
    df_estspline[struct_var] = tmp
    df_estspline = df_estspline.drop(index=df_estspline.iloc[999].name).reset_index(drop=True)
    df_estspline = df_estspline.drop(index=df_estspline.iloc[1998].name)

    fig=plt.figure()
    colors = { 0: 'green', 1: 'blue'}
    sns.lineplot(data=df_estspline, x='Age in Days', y=struct_var, hue='gender', palette=colors, legend=False)
    sns.scatterplot(data=df_origdata, x='Age in Days', y=struct_var, hue='gender', palette=colors)
    plt.legend(title='')
    if roi == 'StateAnxiety':
        plt.ylim(20, 60)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["female", "male"]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(datastr +' ' + struct_var +  ' vs. Age\n' + roi.replace(struct_var+'-', ''))
    plt.xlabel('Age')
    plt.ylabel(datastr + struct_var)
    if showplots == 1:
        if datastr == 'Training Data':
            plt.show(block=False)
        else:
            plt.show()
    else:
        plt.savefig('{}/data/plots/{}_vs_age_withsplinefit_{}_{}'
            .format(outputdir, roi, roi, datastr))
        plt.close(fig)

def plot_y_v_yhat(cov_file, resp_file, yhat, typestring, struct_var, roi, Rho, EV, gendercol, outputdir):
    cov_data = np.loadtxt(cov_file)
    gender = cov_data[:,gendercol].reshape(-1,1)
    y = np.loadtxt(resp_file).reshape(-1,1)
    dfp = pd.DataFrame()
    gender=gender.flatten()
    y=y.flatten()
    yht=yhat.flatten()
    dfp['gender'] = gender
    dfp['y'] = y
    dfp['yhat'] = yhat
    print(dfp.dtypes)
    fig = plt.figure()
    colors = { 0: 'green',1: 'blue',}
    sns.scatterplot(data=dfp, x='y', y='yhat', hue='gender', palette=colors)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["female", "male"]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(typestring + ' ' + struct_var + ' vs. estimate\n'
              + roi +' EV=' + '{:.4}'.format(str(EV.item())) + ' Rho=' + '{:.4}'.format(str(Rho.item())))
    plt.xlabel(typestring + ' ' + struct_var)
    plt.ylabel(struct_var + ' estimate on ' + typestring)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # plots line y = x
    # plt.show(block=False)
    plt.savefig(
    '{}/data/{}/plots/{}_{}_vs_estimate_{}'
    .format(outputdir, struct_var, struct_var, typestring, roi.replace(struct_var + '-', '')))
    plt.close(fig)

def barplot_performance_values(struct_var, metric, df, spline_order, spline_knots, outputdir):
    colors = ['blue' if 'lh' in x else 'green' for x in df.ROI]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=df[metric], y=df['ROI'], orient='h', palette=colors)
    plt.subplots_adjust(left=0.4)
    plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(bottom=0.05)
    ax.set_title('Test Set ' + metric + ' for All Brain Regions')
    # plt.show(block=False)
    plt.savefig(
        '{}/data/{}/plots/Test_Set_{}_for_all_regions_splineorder{}, splineknots{}.png'
        .format(outputdir, struct_var, metric, spline_order, spline_knots))
    #plt.close(fig)

def write_ages_to_file(agemin, agemax, struct_var, outputdir):
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(outputdir, struct_var), "w") as file:
        # Write the values to the file
        file.write(str(agemin) + "\n")
        file.write(str(agemax) + "\n")

def read_ages_from_file(struct_var, outputdir):
    # Open the file in read mode
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(outputdir, struct_var), "r") as file:
        # Read all lines from the file
        lines = file.readlines()
    # Extract the values from the lines
    agemin = int(lines[0].strip())
    agemax = int(lines[1].strip())
    return agemin, agemax

def fit_regression_model_dummy_data(model_dir, dummy_cov_file_path_female, dummy_cov_file_path_male):
    # create dummy data to find equation for linear regression fit between age and structvar
    dummy_predictors_f = pd.read_csv(dummy_cov_file_path_female, delim_whitespace=True, header=None)
    dummy_predictors_m = pd.read_csv(dummy_cov_file_path_male, delim_whitespace=True, header=None)
    dummy_ages_f = dummy_predictors_f.iloc[:, 0]
    dummy_ages_m = dummy_predictors_m.iloc[:, 0]

    # calculate predicted values for dummy covariates for male and female
    output_f = predict(dummy_cov_file_path_female, respfile=None, alg='blr', model_path=model_dir)
    output_m = predict(dummy_cov_file_path_male, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy_f = output_f[0]
    yhat_predict_dummy_m = output_m[0]

    # remove last element of age and output arrays
    last_index = len(yhat_predict_dummy_f) - 1
    yhat_predict_dummy_f = np.delete(yhat_predict_dummy_f, -1)
    yhat_predict_dummy_m = np.delete(yhat_predict_dummy_m, -1)
    dummy_ages_f = np.delete(dummy_ages_f.to_numpy(), -1)
    dummy_ages_m = np.delete(dummy_ages_m.to_numpy(), -1)

    # find slope and intercept of lines
    slope_f, intercept_f, rvalue_f, pvalue_f, std_error_f = stats.linregress(dummy_ages_f, yhat_predict_dummy_f)
    slope_m, intercept_m, rvalue_m, pvalue_m, std_error_m = stats.linregress(dummy_ages_m, yhat_predict_dummy_m)

    # #plot dummy data with fit
    # plt.figure()
    # plt.plot(dummy_ages_f, yhat_predict_dummy_f, 'og', markersize=3, markerfacecolor='None')
    # plt.plot(dummy_ages_m, yhat_predict_dummy_m, 'ob', markersize=3, markerfacecolor='None')
    # plt.plot(dummy_ages_f, slope_f*dummy_ages_f+intercept_f, '-k', linewidth=1)
    # plt.plot(dummy_ages_m, slope_m*dummy_ages_f+intercept_m, '-k', linewidth=1)
    # # plt.title(roi)
    # plt.show(block=False)

    return slope_f, intercept_f, slope_m, intercept_m

def read_text_list(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    #remove newline characters and create list
    mylist = [line.strip() for line in lines]
    return mylist

def plot_scatter_with_trendline_corthick_MEGrs_byreg(df,band, cortthick_cols,band_cols, cortthick_str_to_remove, band_str_to_remove, mycolor):


    all_band_col_names = []
    allslope = []
    allintercept = []
    allr = []
    allp = []

    for i, ct_col_name in enumerate(cortthick_cols):
        cortthick_col_split = ct_col_name.split('-')
        band_col_name= band + '_' + cortthick_col_split[2] + '-' + cortthick_col_split[1]
        all_band_col_names.append(band_col_name)
        slope, intercept, r, p, std_error = linregress(df.loc[:, band_col_name], df.loc[:, ct_col_name])
        allslope.append(slope)
        allintercept.append(intercept)
        allr.append(r)
        allp.append(p)
        mysstop=1

    # correct the pvalues for multiple comparisons
    output = multipletests(allp, alpha=0.05, method='fdr_bh')
    correctedp = output[1]
    rejectho = output[0]

    #make a dataframe that only has regions with significant corrected p values
    # ct_cols_to_plot = cortthick_cols[rejectho.tolist()
    rejectho_list = rejectho.tolist()
    ct_cols_to_plot = [cortthick_cols[i] for i in range(len(cortthick_cols)) if True]
    # ct_cols_to_plot = [cortthick_cols[i] for i in range(len(cortthick_cols)) if rejectho_list[i]]
    ct_cols_to_plot_orig_ind = [i for i, x in enumerate(rejectho_list) if x]
    p_to_plot = [correctedp[i] for i in ct_cols_to_plot_orig_ind]
    r_to_plot = [allr[i] for i in ct_cols_to_plot_orig_ind]
    slope_to_plot = [allslope[i] for i in ct_cols_to_plot_orig_ind]
    intercept_to_plot = [allintercept[i] for i in ct_cols_to_plot_orig_ind]

    for i, ct_col_name in enumerate(ct_cols_to_plot):
        ct_col_split = ct_col_name.split('-')
        band_col_name= band + '_' + ct_col_split[2] + '-' + ct_col_split[1]

        #plot z score of band power against z score of cortical thickness
        plt.scatter(df.loc[:,band_col_name], df.loc[:,ct_col_name], s=12, color=mycolor)

        #calculate trendline
        trendline = slope_to_plot[i] * df.loc[:,band_col_name] + intercept_to_plot[i]

        #plot trendline
        plt.plot(df.loc[:,band_col_name], trendline, color=mycolor, linewidth=1, label='trendline')

        plt.xlabel('z {band} power', fontsize=8)
        plt.ylabel('z cortthick', fontsize=8)

        #make title
        region = ct_col_name.replace(cortthick_str_to_remove, '')
        plt.title(f'z corthick vs. z {band} power\n {region} \nr = : {r_to_plot[i]: .2f}, corrp = {p_to_plot[i]:.2f}', fontsize=8)

        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)

        plt.show()

        mystop=1

def plot_scatter_with_trendline_one_col_against_all_MEGrs(df, strMEGcolsofinterest, cortthick_cols, num_cols, cortthick_str_to_remove,
                                                     col_int_string, cortthick_str, mycolor):
    fig, ax= plt.subplots(1,1)

    df.reset_index(inplace=True, drop=True)

    new_df = df.iloc[:,2:].copy()
    melted_df = new_df.melt().drop('variable', axis=1).rename({'value':'z-score'}, axis=1)

    repeated_puberty = pd.concat([df.iloc[:,1]] * len(cortthick_cols), axis=0, ignore_index=True)

    ax.scatter(repeated_puberty, melted_df, s=12, color=mycolor)

    # calculate linear regression
    slope, intercept, r, p, std_err = linregress(repeated_puberty, melted_df['z-score'])
    trendline = slope * repeated_puberty + intercept

    # plot trendline
    ax.plot(repeated_puberty, trendline, color=mycolor, linewidth=1, label='trendline')

    ax.set_xlabel(col_int_string)
    ax.set_ylabel(cortthick_str)
    ax.set_title(f'{cortthick_str} vs {strMEGcolsofinterest} z across all regions')

    plt.show()

    # fig, ax = plt.subplots(1, 1)
    # for i, col in enumerate(cortthick_cols):
    #     #plot col of interest against cols in cortthick_cols
    #     ax.scatter(df[strMEGcolsofinterest], df[col], s=12, color=mycolor)
    #
    # ax.set_xlabel(col_int_string)
    # ax.set_ylabel(cortthick_str)
    # ax.set_title(f'{cortthick_str} vs {strMEGcolsofinterest} z across all regions')
    #
    # plt.show()

    mystop=1


