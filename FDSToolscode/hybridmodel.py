#!/usr/bin/env python3

#
# Copyright (C) 2021 Jerry Hoogenboom
#
# This file is part of FDSTools, data analysis tools for Massively
# Parallel Sequencing of forensic DNA markers.
#
# FDSTools is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# FDSTools is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FDSTools.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Train a hybrid prediction model using 2 person and 3 person mixtures. 

The model obtained from this tool can be used by hybridpredict to predict the 
number of reads for each sequence to be a hybrid artefact. 
"""
import argparse
import csv
import os
import pickle
import re
import numpy as np  # Only imported when actually running this tool.

from concurrent.futures import ProcessPoolExecutor
from errno import EPIPE
import time
from numpy import genfromtxt, mean, std
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

from hybrids import create_mito_features_dataset_from_dir, write_to_file

__version__ = "1.2.0"


# Default values for parameters are specified below.

# Default minimum R2 score.
# This value can be overridden by the -t command line option.
_DEF_MIN_R2 = 0.5


def nested_cross_validation_gridsearch(model, parameters, X_train, y_train):
    '''
    Performs the nested cross validation using a hyperparameter optimisation. It
    uses 5 inner loops where it performs the hyperparameter optimisation and 
    it uses 10 outer loops for cross validation. It outputs the best performing
    parameters.
    '''
    start_time = time.time()

    # initialise rsquared score
    prev_r_squared = 0
    best_perf_metrics = {}
    # enumerate splits
    outer_results = list()


    # configure the cross-validation procedure
    outer_cv = KFold(n_splits=10, shuffle=True)
    cv_inner = KFold(n_splits=5, shuffle=True)
    count = 0
    for train_ix, validate_ix in outer_cv.split(X_train):
        count +=1
        print(f"Outer loop number {count}")
        # split data
        train_data, val_data = X_train[train_ix, :], X_train[validate_ix, :]
        train_label, val_label = y_train[train_ix], y_train[validate_ix]

        # define search
        search = GridSearchCV(model, parameters, cv=cv_inner, refit=True, verbose=1)#, n_jobs=-1)
        # execute search
        result = search.fit(train_data, train_label)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(val_data)
        # evaluate the model
        r_squared = r2_score(val_label, y_pred)
        rmse = mean_squared_error(val_label, y_pred, squared=False)
        mae = mean_absolute_error(val_label, y_pred)

        # find the best performing parameters and rsquared score
        if r_squared > prev_r_squared:
            best_perf_metrics = result.best_params_
            prev_r_squared = r_squared
        
        # store the result
        outer_results.append(r_squared)

        # report progress
        print('>R2=%.3f, RMSE=%.3f, MAE=%s' % (r_squared, rmse, mae))
        # summarize the estimated performance of the model
        print('R2: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


    time_elapsed = time.time()-start_time
    print('Nested cross-validation done in {:.0f}m {:.0f}s'.format(time_elapsed\
                                                     // 60, time_elapsed % 60))

    if best_perf_metrics == {}:
        print("No model has been trained, due to the r2 score being negative!")
    return best_perf_metrics
#nested_cross_validation_gridsearch


def train_hybrid_model(model, X_train, y_train, pcr_kit, best_perf_metrics = {}, \
                check_model_performance=False):
    '''
    Trains the models random forest regression (random_forest  ), XGBoost 
    sklearn and XGBoost. 
    '''

    if model =='random_forest':
        regr_model = RandomForestRegressor(random_state=0, n_jobs=1)
        parameters = {
        'n_estimators': [50, 100, 150, 200, 250],
        'criterion': ["squared_error", "absolute_error", "friedman_mse", \
                      "poisson"],
        'max_depth': [2, 3, 5, 10, 6, 7, 8],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf' : [2, 3, 4],
        'max_leaf_nodes': [2, 3, 4, None]
        }  

    elif model == 'xgboost_sklearn':
        regr_model = GradientBoostingRegressor()
        parameters = {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'criterion': ['friedman_mse','squared_error'],
        'n_estimators': [100,200, 250, 300, 500],
        "max_depth": [3,4,5, 10],
        "learning_rate": [0.01, 0.05, 0.001]
        }  
        
    elif model =='xgboost':
        regr_model = XGBRegressor(seed=20, n_jobs=1)
        parameters = {
        'eta': [0.1, 0.2, 0.3, 0.5, 0.6, 1],
        'gamma': [0, 6, 8, 10, 15, 20, 25],
        'alpha' : [1, 2 ,3, 4, 5],
        'n_estimators': [100, 200, 250, 300,500],
        "max_depth": [3,4,5, 10],
        "learning_rate": [0.01, 0.05, 0.001]
        }

    # if model performance should be checked, perform a nested cross validation   
    if check_model_performance:
        best_perf_metrics = nested_cross_validation_gridsearch(regr_model, \
                                                parameters, X_train, y_train)

    # train a model with the best performing parameters
    regr_model = regr_model.set_params(**best_perf_metrics)
    regr_model.fit(X_train, y_train)
    
    # save model
    if not os.path.exists('Models'):
        os.mkdir('Models')

    filename = f'Models/{model}_{pcr_kit}.sav'
    #output file als argumetn meegeven als de filename
    pickle.dump(regr_model, open(filename, 'wb'))

    return regr_model
# train_hybrid_model

def test_model_on_test_set(regr_model, X_test, y_test):
    '''Provides evaluation metrics on a test set. Can be used for minor/allele 
    combinations or on the test set. '''

    # Check model on the test set
    r_sq = regr_model.score(X_test, y_test)
    y_pred = regr_model.predict(X_test)

    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print(f"coefficient of determination: {r_sq}")
#test_model

def fit_test_hybrid_model(dir, ml_model, pcr_kit, best_perf_metrics = {}, \
                                                check_model_performance=False):
    # Get all hybrids in given directory 
    feature_file, label_file = create_mito_features_dataset_from_dir(dir, pcr_kit)

    # Reads the data and labels
    data, labels, column_names = read_data_and_labels(feature_file, label_file, True)

    # Split data in train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, \
                                                test_size=0.33, random_state=1)

    # Write test set to a separate file, if future checks are needed
    test_features_name = f"test_sets/Test_set_features_{pcr_kit}.csv"
    test_labels_name = f"test_sets/Test_set_labels_{pcr_kit}.csv"
    write_to_file(test_features_name, X_test, column_names)
    labels_test = list(map(lambda el:[el], y_test))
    write_to_file(test_labels_name, labels_test, ["labels"])



    # Train model
    regr_model = train_hybrid_model(ml_model, X_train, y_train, pcr_kit, 
                best_perf_metrics, check_model_performance)

    # Test model on the test set
    test_model_on_test_set(regr_model, X_test, y_test)   

    # Test model on the Minor/hybrids in dataset
    mh_file = feature_file.replace("_all", "_minor")
    mh_labels = feature_file.replace("_all", "_minor")
    mh_data, mh_labels,_ = read_data_and_labels(mh_file, mh_labels, True)
    test_model_on_test_set(regr_model, mh_data, mh_labels)   
#fit_test_hybrid_model
    
def read_data_and_labels(feature_file, label_file, test_data=False):
    column_names = np.loadtxt(open(feature_file, "rb"), delimiter=",", max_rows=1, dtype="str").tolist()
    column_names = column_names[1:]
    data = np.loadtxt(open(feature_file, "rb"), delimiter=",", skiprows=1)
    labels = np.loadtxt(open(label_file, "rb"), delimiter=",", skiprows=1)
    if test_data:
        indices = np.where(data[:,0] >= 30)[0]
        data = data[indices]
        labels = indices
    true_reads = data[:,0]
    data = data[:,1:]
    return data, labels, column_names
# read_data_and_labels

'''

# This main function runs the code and outputs evaluation metrics for training, 
# testing and the hybrid_minor dataset
def main():
    dir = "../../Data/Mitodata/"
    ml_model = "xgboost"
    pcr_kit = "Mito-mini"

    fit_test_hybrid_model(dir, ml_model, pcr_kit, best_perf_metrics = {}, \
                                                check_model_performance=True)


if __name__ == "__main__":
    main()


#'''


# def add_arguments(parser):
#     add_input_output_args(parser)
#     add_allele_detection_args(parser)
#     parser.add_argument("-T", "--num-threads", metavar="THREADS", type=pos_int_arg, default=1,
#         help="number of worker threads to use (default: %(default)s)")
#     filtergroup = parser.add_argument_group("filtering options")
#     filtergroup.add_argument("-t", "--min-r2", type=float, default=_DEF_MIN_R2, metavar="N",
#         help="minimum required r-squared score (default: %(default)s)")
#     parser.add_argument("-r", "--raw-outfile", type=argparse.FileType("tw", encoding="UTF-8"),
#         metavar="RAWFILE",
#         help="write raw data points to this file, for use in stuttermodel "
#              "visualisations (specify '-' to write to stdout; normal output on "
#              "stdout is then suppressed)")
#     add_sequence_format_args(parser, default_format="raw", force=True)
# #add_arguments


# def run(args):
#     # Import numpy now.
#     global np
#     import numpy as np

#     files = get_input_output_files(args)
#     if not files:
#         raise ValueError("please specify an input file, or pipe in the output of another program")
#     try:
#         fit_test_hybrid_model(files[0], files[1], args.pcr_kit)
#     except IOError as e:
#         if e.errno == EPIPE:
#             return
#         raise
# #run