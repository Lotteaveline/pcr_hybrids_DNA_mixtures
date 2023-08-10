import sys
import itertools
import pandas as pd


from obtain_hybrids import align_3_sequences_prefix_suffix


def calc_hybrid_potential_factor(file):
    '''This function calculates the hybrid protential factor between all given 
    sequences. It selects two possible parents and checks all other sequences 
    on their possibilty to form a hybrid. If no hybrid can be formed, the hybrid
    potential factor will not be calculated.
    This function was used for creating two person mixtures and three person 
    mixtures.'''
    df = pd.read_table(file,sep='\t')
    sequences = []
    hybrid_parents = {}
    print(df)
    df1 = (df[df['marker'].str.contains('Frag01')])
    df2 = (df[df['marker'].str.contains('Frag02')])
    df3 = (df[df['marker'].str.contains('Frag03')])
    df4 = (df[df['marker'].str.contains('Frag04')])
    df5 = (df[df['marker'].str.contains('Frag05')])
    df6 = (df[df['marker'].str.contains('Frag06')])
    df7 = (df[df['marker'].str.contains('Frag07')])
    df8 = (df[df['marker'].str.contains('Frag08')])
    df9 = (df[df['marker'].str.contains('Frag09')])
    df10 = (df[df['marker'].str.contains('Frag10')])

    sequences.append(df1['rawsequence'].to_list())
    sequences.append(df2['rawsequence'].to_list())
    sequences.append(df3['rawsequence'].to_list())
    sequences.append(df4['rawsequence'].to_list())
    sequences.append(df5['rawsequence'].to_list())
    sequences.append(df6['rawsequence'].to_list())
    sequences.append(df7['rawsequence'].to_list())
    sequences.append(df8['rawsequence'].to_list())
    sequences.append(df9['rawsequence'].to_list())
    sequences.append(df10['rawsequence'].to_list())

    for dataframe in sequences:
        for pos_hybrid in dataframe:
            parent_sets = []
            temp = dataframe[:]
            temp.remove(pos_hybrid)
            parent_sets = parent_sets + [i for i in itertools.combinations(temp,2)]
            for parent_set in parent_sets:
                align_hybrid_parents, overlap, overlap_A, overlap_B, hybrid_bool = align_3_sequences_prefix_suffix(pos_hybrid, parent_set)
                if align_hybrid_parents == []:
                    continue
                factor = 1 - (len(overlap_A) + len(overlap_B))/ min(len(align_hybrid_parents[1]), len(align_hybrid_parents[2]))
                if factor > 0.8:
                    hybrid = df.loc[df['rawsequence'] == pos_hybrid, 'sequence'].values[0]
                    parA = df.loc[df['rawsequence'] == parent_set[0], 'sequence'].values[0]
                    parB = df.loc[df['rawsequence'] == parent_set[1], 'sequence'].values[0]
                    marker = df.loc[df['rawsequence'] == parent_set[1], 'marker'].values[0]
                    
                    if marker not in hybrid_parents:
                        hybrid_parents[marker] = [[hybrid, parA, parB]]
                    else:
                        hybrid_parents[marker].append([hybrid, parA, parB, factor])
                
    # print(pd.DataFrame.from_dict(hybrid_parents))
    print(hybrid_parents)

def main():
    file = sys.argv[1]
    calc_hybrid_potential_factor(file)


if __name__ == "__main__":
    main()
