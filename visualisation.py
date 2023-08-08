import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#--------------- VISUALISATION ---------------#
def pretty_print_alignments(hybrid_parents,f):    
    '''This function pretty prints the hybrid with the parents. It is inspired 
    byt the code from the course Algorithms in Sequence Analysis (X_405050) 
    given at the VU. 
    It prints the parent with the longest common preffix first, then the hybrid 
    and then the parent with the longest common suffix. The places where the 
    hybrid and parents align, are markes by a |.'''
    sequences = list(hybrid_parents[0].keys())
    reads = list(hybrid_parents[0].values())
    hybrid = sequences[0]
    parentA = sequences[1]

    # the if-else is needed when there is one parent
    if len(sequences) == 3:
        parentB = sequences[2]
        reads_parentsA = reads[1]
        reads_parentsB = reads[2]
    else:
        parentB = sequences[1]
        reads_parentsA = reads[1]
        reads_parentsB = reads[1]
    matchA = ''
    matchB = ''
    for i in range(len(hybrid)):
        if i == len(parentA):
            break
        if hybrid[i] == parentA[i]:
            matchA += '|' 
        else:
            break

    matchA += ' ' * (len(parentA)-len(matchA))
    
    for i in range(len(hybrid)):
        if i == len(parentB):
            break
        if hybrid[::-1][i] == parentB[::-1][i]:
            matchB += '|' 
        else:
            break
    matchB += ' ' * (len(parentB)-len(matchB))
    if len(hybrid) > len(parentB):
        diff_hybrid_parent = " " * (len(hybrid) - len(parentB))
        hybrid =  "hybrid  = " + hybrid
        parentA = "parentA = " + parentA
        parentB = "parentB = " + diff_hybrid_parent + parentB
        matchA =  "          " + matchA
        matchB =  "          " + diff_hybrid_parent + matchB[::-1]
    else:
        diff_hybrid_parent = " " * (len(parentB) - len(hybrid))
        hybrid =  "hybrid  = " + diff_hybrid_parent + hybrid
        parentA = "parentA = " + diff_hybrid_parent +parentA
        parentB = "parentB = " + parentB
        matchA =  "          " + diff_hybrid_parent + matchA
        matchB =  "          " + matchB[::-1]
    true_all_hyb = "Hybrid allele = " + hybrid_parents[4]
    true_all_A = "Parent A allele = " + hybrid_parents[5]
    true_all_B = "parent B allele = " + hybrid_parents[6]
    h_reads = "Hybrid reads = " + str(reads[0])
    A_reads = "ParentA reads = " + str(reads_parentsA)
    B_reads = "ParentB reads = " + str(reads_parentsB)
    aligned_matches = '\n '.join([true_all_hyb, true_all_A, true_all_B, h_reads, \
                                  A_reads, B_reads, parentA, matchA, hybrid, matchB, parentB])
    print(aligned_matches)
    f.write(aligned_matches)

    # uncomment the following lines for writing the pretty print alignement to a 
    # file
    # with open('pretty_print_hybrid16-new.txt', 'a') as convert_file:
    #     string_list = [h_reads, A_reads, B_reads, parentA, matchA, hybrid, matchB, parentB]
    #     for i in string_list:
    #         convert_file.write(i + '\n')
        
    #     convert_file.write('-'*20 + '\n')

    return aligned_matches    

def histogram_ratios_per_class(hybrid_label_dd, non_hybrid_label_dd):
    # Create list of data according to different accessibility index
    hybrid_label_dd = [val for i in hybrid_label_dd.values() for val in i]
    non_hybrid_label_dd = [val for i in non_hybrid_label_dd.values() for val in i]

    plt.hist(hybrid_label_dd, bins=50, color='blue', label='hybrids')
    plt.hist(non_hybrid_label_dd, bins=50,color='red', label='non-hybrids')
    
    plt.legend()
    
    plt.title('Side-by-side Histogram for sequence/parents ratio')
    plt.show(block=True)

def scatter_predict_vs_true(y_pred, y_true, model="",dataset = "", train_or_test="", reads_ratio="ratio"):
    '''This models creates a scatter plot for a predicted value against its 
    actual value. It also plots a line for perfect prediction where x = y.'''
    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, y_true , s=5, c='deepskyblue')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, ls="--", c=".3")    
    plt.title(f"Predicted {reads_ratio} for {model} on {train_or_test}")
    plt.ylabel(f'True {reads_ratio}')
    plt.xlabel(f'Predicted {reads_ratio}')
    plt.tight_layout()
    

    plt.savefig(f"../Afbeeldingen/{model}_{dataset}_{train_or_test}_{reads_ratio}")
    plt.figure().clear()

# def plot_training_evaluation(model,X_test, y_test, y_pred):
#     test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
#     for i, y_pred in enumerate(model.staged_predict(X_test)):
#         test_score[i] = mean_squared_error(y_test, y_pred)

#     fig = plt.figure(figsize=(6, 6))
#     plt.subplot(1, 1, 1)
#     plt.title("Deviance")
#     plt.plot(
#         np.arange(params["n_estimators"]) + 1,
#         model.train_score_,
#         "b-",
#         label="Training Set Deviance",
#     )
#     plt.plot(
#         np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
#     )
#     plt.legend(loc="upper right")
#     plt.xlabel("Boosting Iterations")
#     plt.ylabel("Deviance")
#     fig.tight_layout()
#     plt.show()

def plot_ratio_to_feature(hybrid_dd, hybrid_label_dd, non_hybrid_dd, non_hybrid_label_dd, feature_index=1):
    hybrid_parent_ratios = []
    read_counts = []
    non_hybrid_parent_ratios = []
    non_read_counts = []
    for marker, features in hybrid_dd.items():
        hybrid_parent_ratios.extend(hybrid_label_dd[marker])
        read_counts.extend([item[feature_index] for item in features])

        # non_hybrid_parent_ratios.extend(non_hybrid_label_dd[marker])
        # non_read_counts.extend([item[feature_index] for item in non_hybrid_dd[marker]])
        
    read_counts = [float(i) for i in read_counts]

    non_read_counts = [float(i) for i in non_read_counts]
    x, y = zip(*sorted(zip(read_counts, hybrid_parent_ratios)))

    # x1, y1 = zip(*sorted(zip(non_hybrid_parent_ratios, non_read_counts)))
    plt.scatter(x,y , s=2, c='b')
    # plt.scatter(x1,y1 , s=2, c='r')
    plt.ylabel('Length overlap (nt)')
    plt.xlabel('Ratio hybrids to parent reads')
    plt.show(block=True)


def plot_feature_against_ratio(file, labels):
    data = pd.read_pickle(file)  
    labels = pd.read_pickle(labels)  
    labels = labels['labels']
    image_path = "/home/lotte/Documents/MSc_thesis/Afbeeldingen/Feature_against_ratio"

    for column in data:
        feature_data = data[column]
        name = feature_data.name
        feature_data = np.array(feature_data)
        norm = np.linalg.norm(feature_data)
        feature_data = feature_data/norm
        print(feature_data, labels)
        feature_data = -np.log(feature_data)
        x, y = zip(*sorted(zip(feature_data, labels)))
        plt.scatter(x,y , s=2, c='b')
        # plt.scatter(x1,y1 , s=2, c='r')
        plt.ylabel(f'{name}')
        plt.xlabel('Ratio hybrids to parent reads')
        # plt.show(block=True)
        plt.savefig(image_path + "/" + str(name))
        plt.clf()


def read_data_plot_relation_and_check_normal_distribution(file, labels):
    data = pd.read_pickle(file)  
    y = pd.read_pickle(labels)  

    X, y = zip(*sorted(zip(X, y)))
    plt.scatter(X ,X, s=2, c='r')

    plt.plot(data, norm.pdf(data,0,2))
    plt.ylabel('Length overlap (nt)')
    plt.xlabel('Ratio hybrids to parent reads')
    plt.show(block=True)


def main():
    data = "Features_data_Mitodata_Mito-mini/features_all.pkl"
    labels = "Features_data_Mitodata_Mito-mini/labels_all.pkl"

    plot_feature_against_ratio(data, labels)
    read_data_plot_relation_and_check_normal_distribution(data, labels)


if __name__ == "__main__":
    main()

