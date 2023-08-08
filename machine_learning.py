import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import visualisation as pp
import pickle

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def feature_selection(data, labels, dataset): 
    '''This function was used for testing the information in features for 
    explaining the hyrbid ratio. Eventually all the created features are used in 
    the predcitive model. The feature selections uses a scoring function of 
    mutual information.'''

    data = data.loc[:, (data != 0).any(axis=0)]
    labels = np.ravel(labels.astype('float'))

    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    fs.fit(data, labels)

    feature_names = []
    indices = np.argsort(fs.scores_)[::-1]
    for i in range(len(fs.scores_)):
        feature_names.append(data.columns[indices[i]])
        print('Feature %d: %f' % (i, fs.scores_[i]))


    high_fs = [i for i in fs.scores_ if i > 0.1] 
    num_features = len(high_fs)

    fs = SelectKBest(mutual_info_regression, k=num_features)
    fs.fit(data, labels)
    cols_idxs = fs.get_support(indices=True)
    features_df_new = data.iloc[:,cols_idxs]


    plt.barh(feature_names, fs.scores_[indices], color='turquoise')
    for index, value in enumerate(fs.scores_[indices]):
        plt.text(value, index, str(round(value, 3)))
    plt.xlabel("Mutual Information score")
    plt.tight_layout()

    plt.savefig("../Afbeeldingen/Feature_selection_" + str(dataset))
    print(list(features_df_new.columns.values))

    return features_df_new, labels, cols_idxs


def nested_cross_validation_gridsearch(model, parameters, X_train, y_train):
    '''
    Performs the nested cross validation using a hyperparameter optimisation.
    The model is trained after performing a k-fold of 5 splits with hyperparameter 
    optimisation. Then the nested cross validation starts, to check the models
    perfomances. 
    During research, the nested cross-validation was first used to check the 
    performances of the models, before pushing through with the used features. 
    It is now after the training of the model, to show the workings of the 
    nested-cross validation, but can be commented out.
    '''
    start_time = time.time()

    inner_cv = KFold(n_splits=5, shuffle=True)
    outer_cv = KFold(n_splits=10, shuffle=True)

    clf = GridSearchCV(model, param_grid=parameters, cv=inner_cv, verbose=1)
    clf.fit(X_train, y_train)
    non_nested_score = clf.best_score_
    print(non_nested_score)
    print(clf.best_params_)

    nested_score = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv).mean()
    print(nested_score)

    time_elapsed = time.time()-start_time
    print('Nested cross-validation done in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return clf


def train_model(model, X_train, X_test, y_train, y_test,cols_idxs,dataset):
    '''
    Trains the models random forest regression (random_forest  ), XGBoost 
    sklearn and XGBoost. 
    '''
    if model =='random_forest':
        model_name = "Random Forest Regression"
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
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
        model_name = "XGBoost Regression sklearn"
        regr = GradientBoostingRegressor()
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        parameters = {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'criterion': ['friedman_mse','squared_error'],
        'n_estimators': [100,200, 250, 300, 500],
        "max_depth": [3,4,5, 10],
        "learning_rate": [0.01, 0.05, 0.001]
        }  
        
    elif model =='xgboost':
        model_name = "XGBoost Regression"
        regr = XGBRegressor(seed=20)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        parameters = {
        'eta': [0.1, 0.2, 0.3, 0.5, 0.6, 1],
        'gamma': [0, 6, 8, 10, 15, 20, 25],
        'alpha' : [1, 2 ,3, 4, 5],
        'n_estimators': [100, 200, 250, 300,500],
        "max_depth": [3,4,5, 10],
        "learning_rate": [0.01, 0.05, 0.001]
        }
        
    regr = nested_cross_validation_gridsearch(regr, parameters, X_train, y_train)
    filename = f'Models/{model}_{dataset}.sav'
    pickle.dump(regr, open(filename, 'wb'))

    # the following lines of code will create a bargraph of the feature importances
    plt.title(f"{model_name} feature importance {dataset}")
    feature_names = []
    indices = np.argsort(regr.best_estimator_.feature_importances_)[::-1]

    #get the names of each feature in order of importance
    for i in range(len(regr.best_estimator_.feature_importances_)):
        feature_names.append(X_train.columns[indices[i]])
    plt.barh(feature_names, regr.best_estimator_.feature_importances_[indices], \
             color='turquoise')
    plt.xlabel("Feature importance")
    plt.tight_layout()

    plt.savefig("../Afbeeldingen/Feature_importance" + str(model_name))

    # get and print the evaluation metrics
    r_sq = regr.score(X_test, y_test)
    y_pred = regr.predict(X_test)
    print(len(np.unique(y_pred)))

    pp.scatter_predict_vs_true(y_pred, y_test, model_name, "test set")
    print(model_name)
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print(f"coefficient of determination: {r_sq}")
        
    # test the mino alleles for the trained model
    test_minor_alleles(regr,cols_idxs, model_name, dataset)


# -------------- Predict the Minor/Hybrid dataset -------------- #

def test_minor_alleles(regr, cols_idxs, model="", dataset="", raw_data=\
                       "Features_data_Mitodata_DSNAME/features_minor.pkl", \
                        labels="Features_data_Mitodata_DSNAME/labels_minor.pkl", \
                            extra = "hybrid-minor_allele_combi"):
    '''
    This function predicts the data in a given file. It only considers data 
    where the potential hybrid is at least 50. It outputs a scatter plot with
    the true ratio against the predicted ratio  per sampleand a bargraph with 
    the actual reads along with the predicted reads per sample in the file.
    '''    
    raw_data = pd.read_pickle(raw_data.replace('DSNAME', dataset))
    labels = pd.read_pickle(labels.replace('DSNAME', dataset))


    # uncomment for adding the hybrid/minor combi in E0155
    # minor_data = pd.read_pickle("Features_data_Mitodata_E0155/features_minor.pkl")
    # minor_label = pd.read_pickle("Features_data_Mitodata_E0155/labels_minor.pkl")

    # raw_data = pd.concat([raw_data, minor_data])
    # labels = pd.concat([labels, minor_label])


    # make data interpretable by model and only select data where the potential 
    # hybrid reads are > 30
    raw_data = raw_data.astype(float)
    drop_indices = raw_data.index[raw_data.hybrid_reads < 30].tolist()
    # drop_indices.extend([1,55]) # uncomment for removing T0320 hybrid/minor combi 

    raw_data = raw_data.drop(index=drop_indices)
    raw_data = raw_data.set_axis(range(len(raw_data)))
    labels = labels.drop(index=drop_indices)
    labels = labels.set_axis(range(len(labels)))

    data = raw_data.iloc[:,cols_idxs]

    y_pred = regr.predict(data)

    #plot the predicted ratios against the actual labels
    pp.scatter_predict_vs_true(y_pred, labels, model, dataset, extra)
    plt.clf()

    # ptint evaluation metrics
    r_sq = regr.score(data, labels)
    print("Test on Minor allele/hybrid combi's")
    print('RMSE: ', mean_squared_error(labels, y_pred, squared=False)) 
    print('MAE: ', mean_absolute_error(labels, y_pred))

    print(f"coefficient of determination: {r_sq}")
    pred = []
    count = 0 
    count2 = 0

    # Iterate through predictions and get actual reads vs predicted reads
    for i in range(len(y_pred)):
        new = y_pred[i]*(raw_data.loc[i, "lowest_reads_par"] + \
                         raw_data.loc[i, "highest_reads_par"])
        pred.append(new)
        if float(new) > float(raw_data.loc[i,"hybrid_reads"]):
            count+=1
        else:
            count2+=1
    # inspect if the model mostly under-predict or over-predicts
    print(f"There are {count} hybrids that predict higher than total")
    print(f"There are {count2} hybrids that predict lower than total")
    X_axis = np.arange(len(pred))

    #plot predicted reas vs actual reads
    pp.scatter_predict_vs_true(pred, raw_data.loc[:,"hybrid_reads"], model, \
                               dataset, extra, 'reads')

    fig = plt.figure(figsize=(40,7))
  
    # these last lines create the side-by-side histogram of actual reads vs
    # predicted reads. For bigger data sets, the histograms get difficult to 
    # interpret
    plt.grid(color='grey', linewidth=1, axis='y')
    plt.bar(X_axis - 0.2, pred, 0.4, label = 'predicted reads by model', \
            color="gold")
    plt.bar(X_axis + 0.2, raw_data.loc[:,"hybrid_reads"], 0.4, label = \
            'actual reads of the sequence', color="darkorange")

    plt.tick_params(axis='x', which='both', bottom=False)
    
    plt.xlabel("Different hybrids")
    plt.ylabel("Reads")
    plt.legend()
    plt.title('Side-by-side Histogram for predicted reads of a sequence vs. \
              actual reads of the sequence in the profile')

    plt.savefig("../Afbeeldingen/Side_by_side_pred_true_" + str(model) + \
                str(dataset), dpi=300)

def main():
    file = sys.argv[1]
    labels = sys.argv[2]
    data_set = file.split('_')[-2].replace('/features',"")

    data = pd.read_pickle(file)  
    labels = pd.read_pickle(labels)  

    data = data.drop(['hybrid_reads'], axis=1)

    data = data.astype(float)
    # data, labels, cols_idxs = feature_selection(data, labels, data_set)

    cols_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=1)
    X_test.to_csv(f"Test_set_{data_set}.csv")
    pd.DataFrame(y_test.T).to_csv(f"Test_set_labels_{data_set}.csv")

    train_model('random_forest', X_train, X_test, y_train, y_test, cols_idxs, data_set)
    train_model('xgboost', X_train, X_test, y_train, y_test, cols_idxs, data_set)
    train_model('xgboost_sklearn', X_train, X_test, y_train, y_test, cols_idxs, data_set)
    

if __name__ == "__main__":
    main()

