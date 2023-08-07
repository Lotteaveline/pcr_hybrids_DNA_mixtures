import numpy as np
import pandas as pd

#--------------- FEATURES FORMULAS AND CONSTANTS ---------------#
MW_dATP = 313.2
MW_dCTP = 289.2
MW_dGTP = 329.2
MW_dTTP = 304.2
MW_addition = 79.0

NN_table = {"H": [-9.1, -8.6, -6.0, -5.8, -6.5, -7.8, -5.6, -11.9,-11.1,-11.0,-9.1,-11.0, -7.8, -6.5, -5.6],
"S": [-0.0240, -0.0239, -0.0169, -0.0129, -0.0173, -0.0208, -0.0135, -0.0278, -0.0267, -0.0266, -0.0240, -0.0266, -0.0208, -0.0173,-0.0135]}

df_MT = pd.DataFrame(data=NN_table, index=["AA", "AT", "TA","CA","GT","CT","GA","CG","GC","GG","TT", "CC", "AG", "AC", "TC"])

# pop_freq = pd.read_excel('True_alleles/Frequentiebestand_NL-donors_Forenseq.xlsx')
# pop_freq['Total_sequence'] = pop_freq['Total_sequence'].str.split('_')
# pop_freq['Total_sequence'] = pop_freq['Total_sequence'].str[0].str.strip('CE')

def calculate_molecular_weight(a_counts, c_counts, g_counts, t_counts):
    mol_weight = (a_counts*MW_dATP) + (t_counts*MW_dTTP) +(c_counts*MW_dCTP) + (g_counts*MW_dGTP) + MW_addition
    return mol_weight

def marmor_doty_formula(a_counts, c_counts, g_counts, t_counts):
    # Marmor-Doty forula (1962) / Wallace rule
    return 2 * (a_counts + t_counts) + 4 *(c_counts+g_counts) - 7

def nearest_neighbour_formula(sequence):
    # Nearest Neigbours melting temperatuur len(sequence >= 14)
    sum_H = 0.0
    sum_S = 0.0
    for i in range(1,len(sequence)):
        pair = sequence[i-1] + sequence[i]
        if pair in df_MT.index:
            sum_H += df_MT.at[pair,"H"]
            sum_S += df_MT.at[pair,"S"]

    # 0.0108 = constant for helix initation
    # 0.00199 = gas constant
    # 0.0000005 = oligonucleotide concentration in M
    return sum_H / (-0.0108 + sum_S + 0.00199 * np.log(0.0000005/4)) - 273.15 + 16.6*np.log10(0.05)


def find_longest_c_stretch(seq):
    maximum = count = 0
    for c in seq:
        if c == 'C':
            count += 1
        else:
            count = 0
        maximum = max(count,maximum)
    return maximum
 
#--------------- FEATURE DATASET CREATION ---------------#

def get_features_and_labels_from_hybrid(hybrid_parents_overlap_allele, marker):
    samples = []
    global pop_freq
    for i in hybrid_parents_overlap_allele:
        allele_freq = 0
        all_sequences = i[0]
        overlap = i[1]
        overlap_parA = i[2]
        overlap_parB = i[3]
        true_allele = i[4]

        sequences =  list(all_sequences.keys())
        reads =  list(all_sequences.values())

        hybrid_sequence = sequences[0]
        parA_sequence = sequences[1]
        parB_sequence = sequences[2]

        hybrid_reads = float(reads[0])
        parA_reads = float(reads[1])
        parB_reads = float(reads[2])

        float_reads = [parA_reads, parB_reads]

        highest_reads_par = max(float_reads)
        lowest_reads_par = min(float_reads)
        ratio_parents = lowest_reads_par/highest_reads_par

        len_hybrid = len(hybrid_sequence)
        len_longest_par= max(len(sequences[1]), len(sequences[2]))
        len_shortest_par = min(len(sequences[1]), len(sequences[2]))

        len_overlap = int(len(overlap))
        len_overlap_parA = int(len(overlap_parA))
        len_overlap_parB = int(len(overlap_parB))
        marker_pos = int(marker.split('_')[1].replace("mt","").split('-')[0])
        begin_pos_overlap = marker_pos + (len_hybrid - len_overlap_parB)
        end_pos_overlap = marker_pos + len_overlap_parA

        hybrid_A_counts = hybrid_sequence.count('A')
        hybrid_C_counts = hybrid_sequence.count('C')
        hybrid_G_counts = hybrid_sequence.count('G')
        hybrid_T_counts = hybrid_sequence.count('T')

        overlap_A_counts = overlap.count('A')
        overlap_C_counts = overlap.count('C')
        overlap_G_counts = overlap.count('G')
        overlap_T_counts = overlap.count('T')

        longest_c_stretch = find_longest_c_stretch(hybrid_sequence)

        hybrid_CG_counts = hybrid_sequence.count('CG') 
        hybrid_AT_counts = hybrid_sequence.count('AT')
        hybrid_GC_counts = hybrid_sequence.count('GC') 
        hybrid_TA_counts = hybrid_CG_counts = hybrid_sequence.count('TA') 
        overlap_CG_counts = overlap.count('CG') 
        overlap_AT_counts = overlap.count('AT')
        overlap_GC_counts = overlap.count('GC') 
        overlap_TA_counts = overlap.count('TA')


        MT_hybrid = marmor_doty_formula(hybrid_A_counts,hybrid_C_counts,hybrid_G_counts,hybrid_T_counts) if len_hybrid >= 14 else nearest_neighbour_formula(hybrid_sequence)
        MT_overlap = marmor_doty_formula(overlap_A_counts,overlap_C_counts,overlap_G_counts,overlap_T_counts) if len_overlap >= 14 else nearest_neighbour_formula(overlap)

        hybrid_mol_weight = calculate_molecular_weight(hybrid_A_counts, hybrid_C_counts,hybrid_G_counts, hybrid_T_counts)

        # used for detemining allele frequency --> not available for mtDNA
        # allele_freq = pop_freq[(pop_freq['marker']==marker) & (pop_freq['Total_sequence'] == true_allele)]
        # if allele_freq.empty:
        #     allele_freq = 0.0   
        # else:
        #     allele_freq = allele_freq
        
        # final feature list
        sample = [hybrid_reads, highest_reads_par, lowest_reads_par, ratio_parents, len_hybrid, len_longest_par, len_shortest_par, begin_pos_overlap, end_pos_overlap, len_overlap, longest_c_stretch, len_overlap_parA, len_overlap_parB, hybrid_CG_counts, hybrid_AT_counts, overlap_CG_counts, overlap_AT_counts, hybrid_A_counts, hybrid_C_counts, hybrid_G_counts, hybrid_T_counts, overlap_A_counts, overlap_C_counts, overlap_G_counts, overlap_T_counts, MT_hybrid, MT_overlap, hybrid_mol_weight]

        # label creation
        label = hybrid_reads/(parA_reads + parB_reads)
        sample = [sample,label]
        samples.append(sample)

    return samples

