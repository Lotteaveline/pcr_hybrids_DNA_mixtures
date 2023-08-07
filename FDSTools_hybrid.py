import pickle
import sys
import pandas as pd
import obtain_hybrids as oh
import machine_learning as ml
import matplotlib.pyplot as plt
import numpy as np

def make_prediction(model, data, parentA_reads, parentB_reads):

    result = model.predict(data)
    print(result)    

def load_data(tssv_file, col_idxs, model_file='Models/random_forest_mito-mini.sav'):
    feat_names = ['highest_reads_par', 'lowest_reads_par', 'ratio_parents', 'len_hybrid', 'len_longest_par', 'len_shortest_par', 'begin_pos_overlap', 'end_pos_overlap', 'len_overlap', 'longest_c_stretch', 'len_overlap_parA', 'len_overlap_parB', 'hybrid_CG_counts', 'hybrid_AT_counts', 'overlap_AT_counts', 'hybrid_A_counts', 'hybrid_C_counts', 'hybrid_G_counts', 'hybrid_T_counts', 'overlap_A_counts', 'overlap_C_counts', 'overlap_G_counts', 'overlap_T_counts', 'MT_hybrid', 'MT_overlap', 'hybrid_mol_weight']

    loaded_model = pickle.load(open(model_file, 'rb'))

    marker_hybrid_dict, hybrid_features_dict, non_hybrid_features_dict = oh.obtain_all_hybrids_from_file(tssv_file, False)
    for marker, align_hybrid_parents in marker_hybrid_dict.items():
        for combi in align_hybrid_parents:

            sequence_name = combi[4]
            raw_features = oh.get_features_and_labels_from_hybrid([combi], marker)[0][0]
            true_label = raw_features[0]/(raw_features[1]+raw_features[2])
            selected_features = [raw_features[i] for i in col_idxs]

            features = np.array([selected_features])
            features = pd.DataFrame(features, columns = feat_names)
            y_pred = loaded_model.predict(features)

            r_sq = loaded_model.score(X_test, y_pred)
            print(y_pred)
            print(sequence_name)


def main():
    file = sys.argv[1]
    labels = sys.argv[2]
    model = sys.argv[3]
    dataset_name = file.split('_')[-2].replace('/features',"")

    model_name = "XGBoost"
    dataset = "Easymito"


    cols_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]


    # load_data(file, cols_idxs)

    # file = "Features_data_Mitodata_mito-mini/features_minor.pkl"
    # labels = "Features_data_Mitodata_mito-mini/labels_minor.pkl"
    # model = 
    # 'Models/random_forest_mito-mini.sav'

    data = pd.read_pickle(file)

    data = data.drop(['hybrid_reads'], axis=1)


    loaded_model = pickle.load(open(model, 'rb'))

    plt.title(f"{model_name} feature importance {dataset}")
    feature_names = []
    indices = np.argsort(loaded_model.best_estimator_.feature_importances_)[::-1]
    for i in range(len(loaded_model.best_estimator_.feature_importances_)):
        feature_names.append(data.columns[indices[i]])
    plt.barh(feature_names, loaded_model.best_estimator_.feature_importances_[indices], color='turquoise')
    plt.xlabel("Feature importance")
    # plt.show()

    if "320" in file or '155' in file:
        type_data = '3 person mixture'
    elif "216" in file or '122' in file:
        type_data = '2 person mixture'
    else:
        type_data = 'hybrid-minor_allele_combi'

    if "random" in model:
        model_name = "RandomForest"
    else: 
        model_name = "XGBoost"
    ml.test_minor_alleles(loaded_model, cols_idxs, model_name,dataset_name, file, labels, type_data)



if __name__ == "__main__":
    main()


    
