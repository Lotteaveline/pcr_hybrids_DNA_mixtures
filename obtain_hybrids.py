import itertools
import time
import os
import sys 
import argparse
from collections import defaultdict
import pandas as pd
import true_alleles as ta
from feature_creation import get_features_and_labels_from_hybrid
# import visualisation as pp

reads_threshold = 10.0 
COUNT = 0
minor_alleles_and_hybrids = defaultdict(list)

#----------- OBTAIN HYBRIDS AND ITS PARENTS FROM A FOLDER -----------#

def get_subfolder_form_all_mito(folder, dataset_name, train=True):
    '''
    This function creates feature datasets for hybrid noise, non-hybrid noise 
    (other noise) and minor/hybrid allele combinations. From the folder input it 
    fetches all the subfolders conatining output from FDSTools. It reads the 
    contents of it and compares it to the true alleles in the dataset, to find 
    possible alleles from a third contributor that can also be formed as hybrid.
    The output is written to CSV files over the complete dataset given and as 
    fragments, with the corresponding labels file with the hybrid ratio.
    '''
    global minor_alleles_and_hybrids
    columns = ['hybrid_reads', 'highest_reads_par', 'lowest_reads_par', \
              'ratio_parents', 'len_hybrid', 'len_longest_par', \
              'len_shortest_par', 'begin_pos_overlap', 'end_pos_overlap', \
              'len_overlap', 'longest_c_stretch', 'len_overlap_parA', \
              'len_overlap_parB', 'hybrid_CG_counts', 'hybrid_AT_counts', \
              'overlap_CG_counts', 'overlap_AT_counts', 'hybrid_A_counts', \
              'hybrid_C_counts', 'hybrid_G_counts', 'hybrid_T_counts', \
              'overlap_A_counts', 'overlap_C_counts', 'overlap_G_counts', \
              'overlap_T_counts', 'MT_hybrid', 'MT_overlap', 'hybrid_mol_weight']

    all_features = []
    all_labels = []
    minor_features =[]
    minor_labels =[]
    start_time = time.time()
    dirs = []

    kit = folder.split('/')[-1]
    hybrid_dd = defaultdict(list)
    hybrid_label_dd = defaultdict(list)

    # not used in thesis (other noise!)
    non_hybrid_dd = defaultdict(list)
    non_hybrid_label_dd = defaultdict(list)

    minor_dd = defaultdict(list)
    minor_label_dd = defaultdict(list)

    for (root,_,_) in os.walk(folder): 
        if "results_fdstools_" in root:
            dirs.append(root)

    for dir in dirs:
        # this statments excludes specific data sets and every thing that does
        # not have the dataset_name in it
        if "_mito-mini_low" in dir or 'Saliva' in dir or not dataset_name in dir.split('/')[3]:
            continue

        hybrid_features, non_hybrid_features = obtain_all_hybrids_from_folder(dir, dataset_name, train)

        # creates a label and feature dictionary from hybrids 
        for key, values in hybrid_features.items():
            for value in values:
                hybrid_dd[key].append(value[0])
                hybrid_label_dd[key].append(value[1])
                
        # creates a label and feature dictionary from other noise (not for thesis)
        for key, values in non_hybrid_features.items():
            for value in values:
                non_hybrid_dd[key].append(value[0])
                non_hybrid_label_dd[key].append(value[1])
        

        # creates a label and feature dictionary from the hybrid/minor combis
        for key, values in minor_alleles_and_hybrids.items():
            for value in values:
                minor_dd[key].append(value[0])
                minor_label_dd[key].append(value[1])

        minor_alleles_and_hybrids = defaultdict(list)

        
        time_elapsed = time.time()-start_time
        print('Obtained features of {:s} in {:.0f}m {:.0f}s'.format(dir, time_elapsed // 60, time_elapsed % 60))

    # Comment out to inspect linearity in data
    # pp.histogram_ratios_per_class(hybrid_label_dd, non_hybrid_label_dd)
    # pp.plot_ratio_to_reads(hybrid_dd,hybrid_label_dd, non_hybrid_dd, non_hybrid_label_dd)


    kit = 'Features_data_' + kit + "_" + dataset_name + '_new'

    for marker, hybrid_features in hybrid_dd.items():
        all_features.extend(hybrid_features)


        # non_hybrid_features = non_hybrid_dd[marker]
        # all_features.extend(non_hybrid_features)
        # hybrid_features.extend(non_hybrid_features)

        minor_features.extend(minor_dd[marker])
        minor_labels.extend(minor_label_dd[marker])


        hybrid_labels = hybrid_label_dd[marker]
        # non_hybrid_labels = non_hybrid_label_dd[marker]
        
        all_labels.extend(hybrid_labels)
        # all_labels.extend(non_hybrid_labels)
        
        hybrids_classes = len(hybrid_labels)
        print(f'{marker} has {hybrids_classes} hybrids')
        
        # non_hybrids_classes = len(non_hybrid_labels)
        # print(f'{marker} has {non_hybrids_classes} non_hybrids')

        # hybrid_labels.extend(non_hybrid_labels)
        # labels = hybrids_classes * [0] + non_hybrids_classes * [1]
        
        df_labels = pd.DataFrame(hybrid_labels, columns = ['labels'])
        df_features = pd.DataFrame(hybrid_features, columns=columns)
        
        pkl_file_name = f'{kit}/features_{marker}.pkl' 
        df_features.to_pickle(pkl_file_name)
        csv_file_name = f'{kit}/features_{marker}.csv'
        df_features.to_csv(csv_file_name)

        labels_file = f'{kit}/labels_{marker}.pkl' 
        df_labels.to_pickle(labels_file)
        csv_file_name = f'{kit}/labels_{marker}.csv'
        df_labels.to_csv(csv_file_name)


    df_labels = pd.DataFrame(minor_labels, columns = ['labels'])
    df_features = pd.DataFrame(minor_features, columns=columns)
    
    pkl_file_name = f'{kit}/features_minor.pkl' 
    df_features.to_pickle(pkl_file_name)
    csv_file_name = f'{kit}/features_minor.csv'
    df_features.to_csv(csv_file_name)

    labels_file = f'{kit}/labels_minor.pkl' 
    df_labels.to_pickle(labels_file)
    csv_file_name = f'{kit}/labels_minor.csv'
    df_labels.to_csv(csv_file_name)

    df_labels = pd.DataFrame(all_labels, columns = ['labels'])
    df_features = pd.DataFrame(all_features, columns=columns)
    
    pkl_file_name = f'{kit}/features_all.pkl' 
    df_features.to_pickle(pkl_file_name)
    csv_file_name = f'{kit}/features_all.csv'
    df_features.to_csv(csv_file_name)

    labels_file = f'{kit}/labels_all.pkl' 
    df_labels.to_pickle(labels_file)
    csv_file_name = f'{kit}/labels_all.csv'
    df_labels.to_csv(csv_file_name)

     

def obtain_all_hybrids_from_folder(folder,dataset_name, train=True):
    '''
    This function takes one FDSTools output folder and then obtains all the 
    potential hybrids in each file and outputs them as a default dictionaries 
    for the hybrid noise, non-hybrid noise and a list of hybird parents. This 
    last list was used for testing and might be obsolete.
    '''
    folder_name = folder.split('_')
    hybrid_feature_dict = defaultdict(list)
    non_hybrid_feature_dict = defaultdict(list)
    # creates dictionaries for hybrids, hybrid_features and non-hybrid features
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            if train:
                # Exclude the two files used for validation in the thesis. These  
                # are a 2-pers mix and 3-pers mix from 2 unseen individuals.
                if ("2023" in folder_name[0]):
                    if dataset_name == "Mito-mini" and (file == 'T0216-table.txt'\
                                                     or file == 'T0320-table.txt'):
                        continue
                    elif dataset_name == "Easymito" and (file == 'E0122-table.txt'\
                                                     or file == 'E0155-table.txt'):
                        continue
            # if "2018" in folder_name[0]:
            #     continue
            
            file_path = os.path.join(folder, file)
                    
            hybrid_features, non_hybrid_features = obtain_all_hybrids_from_file(file_path, train)

            hybrid_feature_dict = merge_defaultdicts(hybrid_feature_dict, hybrid_features)
            non_hybrid_feature_dict = merge_defaultdicts(non_hybrid_feature_dict, non_hybrid_features)

    return hybrid_feature_dict, non_hybrid_feature_dict, 

def merge_defaultdicts(d,d1):
    '''
    https://stackoverflow.com/questions/31216001/how-to-concatenate-or-combine-two-defaultdicts-of-defaultdicts-that-have-overlap
    '''
    for k,v in d1.items():
        if (k in d):
            d[k].extend(v)
        else:
            d[k] = d1[k]
    return d

def obtain_all_hybrids_from_file(filename, train=True, f=''):
    marker_hybrid_dict = {}
    global minor_alleles_and_hybrids
    hybrid_features_dict = defaultdict(list)
    non_hybrid_features_dict = defaultdict(list)
    filepath_list = filename.split('/')
    if train:
        dataset = filepath_list[3]
    else:
        dataset = ''
    barcode_table = filepath_list[-1].replace('-table.txt', "")

    marker_sequences_dict  = create_sequence_dict_from_file(filename)

    if marker_sequences_dict == {}:
        return {},{}

    for marker, values in marker_sequences_dict.items():
        ''' #comment this line if you want to pretty print a file
        f = open(f"{barcode_table}", "at")
        f.write('*'*40 + '\n' + f'{marker}' +   '\n')
        f = open(f"{filename}-{marker}", "at")
        #'''

        # create features for the  hybrids with parents and non-hybrids with parents
        align_hybrid_parents, align_non_hybrid_parents = get_hybrid_parents_combis_and_overlaps(values, train, f)
        if align_hybrid_parents != []:

            if marker not in hybrid_features_dict:
                marker_hybrid_dict[marker] = []
                hybrid_features_dict[marker] = []
            
            if train:
                minor_alleles = ta.get_minor_allele_hybrid_combi(align_hybrid_parents, marker, barcode_table, dataset)
                if minor_alleles != []:
                    if marker not in minor_alleles_and_hybrids:
                        minor_alleles_and_hybrids[marker] = []
                    minor_alleles_and_hybrids[marker].extend(get_features_and_labels_from_hybrid(minor_alleles, marker))
                    for i in minor_alleles:
                        align_hybrid_parents.remove(i)

            # marker_hybrid_dict was needed for testing, not feature creation
            # it contains alle hybrid parents combinations
            marker_hybrid_dict[marker].extend(align_hybrid_parents) 
            features = get_features_and_labels_from_hybrid(align_hybrid_parents, marker)
            hybrid_features_dict[marker].extend(features)

        # not used in thesis:
        if align_non_hybrid_parents != []:
            if marker not in non_hybrid_features_dict:
                non_hybrid_features_dict[marker] = []
            features = get_features_and_labels_from_hybrid(align_non_hybrid_parents, marker)
            non_hybrid_features_dict[marker].extend(features)

            
    return hybrid_features_dict, non_hybrid_features_dict


def create_sequence_dict_from_file(file_path):
    df = pd.read_csv(file_path, sep='\t', lineterminator='\n')

    # check if there is data in the file
    if df.iloc[1]['sequence'] == "No data":
        return {}

    # try-except for catching differences in versions of FDSTools data creation
    try:
        total_column = 'total_corrected'
        df = df[['marker','sequence', 'rawsequence', total_column]]
        df = df[df.sequence != "Other sequences"]
    except:
        total_column = 'total'
        if 'tssv' in file_path:
            df = df[['marker','sequence', total_column]]
            df = df.rename(columns={"sequence": "rawsequence"})
            df["sequence"] = ""
            df = df[df.rawsequence != "Other sequences"]
        else:
            df = df[['marker','sequence', 'rawsequence', total_column]]
            df = df[df.sequence != "Other sequences"]

    # alter dataframe and convert to nested dictionary
    df = df[df[total_column] > reads_threshold]
    df[total_column] = df[total_column].astype(str)
    df['sequence'] = df['sequence'].str.split('_').str[0]
    df = df.rename(columns={total_column: 'reads'})
    markers_sequences_dict = df.groupby('marker')[['rawsequence','sequence','reads']].apply(lambda x: x.set_index('rawsequence').to_dict(orient='index')).to_dict()

    return markers_sequences_dict


def get_hybrid_parents_combis_and_overlaps(sequence_reads_dict, train=True, f=''):
    global COUNT
    hybrid_parents_combis = []
    non_hybrid_parent_combis = []
    sequence_list = list(sequence_reads_dict.keys())

    # Consider each sequence  and check if it can be a hybrid
    for pos_hybrid in sequence_list:
    
        #creates a possible parent list without the possible hybrid
        parent_sets = []
        temp = sequence_list[:]
        temp.remove(pos_hybrid)

        # pick two parents from the left over sequences 
        # parent_sets = parent_sets + [(i,i) for i in temp] # used to get hybrids from one parent
        parent_sets = parent_sets + [i for i in itertools.combinations(temp,2)]

        for parent_set in parent_sets:
            hybrid_parents_read_list, hybrid_bool = get_hybrid_parents_list(pos_hybrid, parent_set, sequence_reads_dict, train) 
            if hybrid_parents_read_list == []:
                continue
            if hybrid_bool:
                COUNT += 1
                hybrid_parents_combis.append(hybrid_parents_read_list)
            else:
                non_hybrid_parent_combis.append(hybrid_parents_read_list)

    '''
    # comment the 3 ' above if you want to pretty print the hybrid/parents
    # and create its text file
    for i in hybrid_parents_combis:
        if isinstance(i, int):
            print("Length overlap = " + str(i))
            f.write("Length overlap = " + str(i))
            continue
        if isinstance(i, str):
            print("Allele hybrid: " + i)
            f.write("Allele hybrid: " + i)
            continue
        pp.pretty_print_alignments(i,f)
        print('\n')
        f.write('\n'))
    f.close()

    # '''
    return hybrid_parents_combis, non_hybrid_parent_combis

def get_hybrid_parents_list(pos_hybrid, parent_set, sequence_reads_dict, train=True):
    '''
    Returns the hybrid aligned with the two parents, overlapping pieces, allele 
    names of hybrid and the parents and checks if it is hybrid noise or other 
    noise. Also checks the assumption for training where the parent reads must 
    be higher than hybrid reads.
    '''
    reads_hybrid = float(sequence_reads_dict.get(pos_hybrid)['reads'])
    reads_parA = float(sequence_reads_dict.get(parent_set[0])['reads'])
    reads_parB = float(sequence_reads_dict.get(parent_set[1])['reads'])

    if train:
        # the hybrid reads must be lower than both parent reads for the training set
        if reads_hybrid >= reads_parA or reads_hybrid >= reads_parB:
            return [], False
    else:
        # the hybrid and parent reads must at least be 50 reads
        th_reads = 50
        if reads_hybrid < th_reads or reads_parA < th_reads  or reads_hybrid < th_reads:
            return [], False


    # align the hybrid with the parents
    align_hybrid_parents, overlap, overlap_A, overlap_B, hybrid_bool = align_3_sequences_prefix_suffix(pos_hybrid, parent_set)

    # add hybrid/parents combi and stop iterating with this hybrid and parent set
    if align_hybrid_parents != []:
        align_hybrid_parents_read_dict = {}
        if align_hybrid_parents[1] == parent_set[0]:
            align_hybrid_parents_read_dict = {align_hybrid_parents[0]: reads_hybrid,\
             align_hybrid_parents[1]: reads_parA, align_hybrid_parents[2]: reads_parB} 
        else:
            align_hybrid_parents_read_dict = {align_hybrid_parents[0]: reads_hybrid,\
             align_hybrid_parents[1]: reads_parB, align_hybrid_parents[2]: reads_parA} 
        
        name_hybrid = sequence_reads_dict.get(pos_hybrid)['sequence'].split('_')[0]
        name_parA = sequence_reads_dict.get(parent_set[0])['sequence'].split('_')[0]
        name_parB = sequence_reads_dict.get(parent_set[1])['sequence'].split('_')[0]
        return  [align_hybrid_parents_read_dict, overlap, overlap_A, overlap_B, \
                 name_hybrid, name_parA, name_parB], hybrid_bool

    return [], False

#--------------- ALIGNMENT OF SEQUENCES ---------------#

def align_3_sequences_prefix_suffix(hybrid,parent_set):
    '''
    This function will find the longest common prefix and suffix of parents with
    the hybrid in both ways and compares them with the hybrid to see if a hybrid
    might have formed or not. It returns the hybrid and parents in order; prefix
    parent, suffix parent and also returns the overlapping piece, the prefix
    and suffix and a bool if it is hybrid noise or other noise.
    '''

    parA, parB =  parent_set[0], parent_set[1]

    # get possible suffix and prefix from both parent sequences
    parA_prefix = longest_common_prefix_suffix([hybrid, parA])
    parB_prefix = longest_common_prefix_suffix([hybrid, parB])
    parA_suffix = longest_common_prefix_suffix([hybrid[::-1], parA[::-1]])
    parB_suffix = longest_common_prefix_suffix([hybrid[::-1], parB[::-1]])

    #dubbele code --> eruit slopen! --> ??? --> nog doen?
    if len(parA_prefix + parB_suffix) > len(hybrid):
        parB_suffix_rev = parB_suffix[::-1]
        overlap = find_overlapping_region(parA_prefix, parB_suffix_rev)
        parB_suffix_rev_no_overlap = parB_suffix_rev[len(overlap):]

        #check if hybrid and return hybrid bool as True for hybrid creation
        if parA_prefix + parB_suffix_rev_no_overlap == hybrid:
            return [hybrid,parA, parB], overlap, parA_prefix, parB_suffix[::-1], True
        else:
            return [hybrid,parA, parB], overlap, parA_prefix, parB_suffix[::-1], False

    if len(parB_prefix + parA_suffix) > len(hybrid):
        parA_suffix_rev = parA_suffix[::-1]
        overlap = find_overlapping_region(parB_prefix, parA_suffix_rev)
        if parB_prefix + parA_suffix_rev[len(overlap):] == hybrid:
            return [hybrid,parB,parA], overlap,  parB_prefix, parA_suffix[::-1], True
        else:
            return [hybrid,parB,parA], overlap,  parB_prefix, parA_suffix[::-1], False

    return [], 0,0,0, False
        

def find_overlapping_region(seq1, seq2):
    overlap = ''
    for nt in range(len(seq1)):
        if seq2.startswith(seq1[nt:]):
            overlap = seq1[nt:]
            break   
    # The overlap can not be the entire sequence
    if overlap == seq1 or overlap == seq2:
        return ''
    return overlap

def longest_common_prefix_suffix(two_sequences):
    two_sequences.sort()
    end = min(len(two_sequences[0]), len(two_sequences[1]))
    i = 0
    while i<end and two_sequences[0][i] == two_sequences[1][i]:
        i+=1
    prefix = two_sequences[0][0: i]
    return prefix


#--------------- MAIN FUNCTIONS ---------------#

def main():
    folder = sys.argv[1]
    dataset_name = sys.argv[2]
    get_subfolder_form_all_mito(folder, dataset_name)

    global COUNT
    print(f"Total number of hybrids: {COUNT}")


if __name__ == "__main__":
    main()

