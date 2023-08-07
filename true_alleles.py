import pandas as pd
from collections import defaultdict
import re

sample_content_dict = {}

def read_mito_true_alleles(excel_file):
    '''
    This function reads in the excel file with the true alleles per sample for
    a given dataset. It returns all the true alleles per fragment per barcode.
    '''
    sample_content = pd.read_excel(excel_file,sheet_name=0, skiprows=3)
    sample_genotypes = pd.read_excel(excel_file, sheet_name=1, index_col=0)
    
    sample_genotypes_dict = sample_genotypes.to_dict()
    sample_content_dict = sample_content.set_index('tag').to_dict()['sample']
    sample_content_dict = {k: str(v).split('+') for k,v in sample_content_dict.items()}
    true_alleles_per_sample_dict = {}
    
    for barcode, donors in sample_content_dict.items():
        if barcode not in true_alleles_per_sample_dict:
            true_alleles_per_sample_dict[barcode] = {}
        true_alleles_list = []
        for donor, true_alleles in sample_genotypes_dict.items():  
            if str(donor) in donors:
                true_alleles_list.append(true_alleles)
        dd = defaultdict(list)
        for individual in true_alleles_list:
            for fragment, alleles in individual.items():
                dd[fragment].append(alleles)  
        true_alleles_per_sample_dict[barcode] = dd
    return true_alleles_per_sample_dict

def get_minor_allele_hybrid_combi(list_hybrid_parents, marker, tag, dataset):
    '''
    This function finds all minor alleles, that are a true allele from a 
    third contibutor as well.
    '''
    minor_allele_hybrid = []
    global sample_content_dict

    if sample_content_dict == {}:
        filename = f"True_alleles/Final_true_alleles/True-alleles_{dataset}.xlsx"
        sample_content_dict = read_mito_true_alleles(filename) 

    # get the markers with corresponding hybrid alleles
    for hybrid_allele in list_hybrid_parents:
        if tag in sample_content_dict:
            samples = sample_content_dict[tag]
            true_alleles = samples[marker]
            true_hybrid_allele = hybrid_allele[4]
            if true_hybrid_allele in true_alleles:
                # print(tag)
                # print(true_hybrid_allele)
                # print(true_alleles)
                minor_allele_hybrid.append(hybrid_allele)
    return minor_allele_hybrid

def get_true_alleles_from_range_per_fragment(file, kit="easymito"):
    '''
    This function reads a file with the true alleles per person. It is a very 
    specific input, so there is no generalisability. 
    '''
    df = pd.read_excel(file,sheet_name=3, skiprows=57)
    df = df[['SampleID', 'Input_Sample']]

    df_list = df.values.tolist()
    final_dict = {}
    for sample in df_list:
        if isinstance(sample[1], str):
            sample_name = sample[0]
            alleles = sample[1].split()
            true_alleles = defaultdict(list,{ k:"" for k in [1,2,3,4,5,6,7,8,9,10]})
            for i in alleles:
                i_pos = i.split(".")[0]
                position = int(re.sub("[^0-9]", "", i_pos))
                if 16009 <= position <= 16129:
                    true_alleles[1] += i + " "
                if 16113 <= position <= 16227:
                    true_alleles[2] += i + " "
                if 16222 <= position <= 16380:
                    true_alleles[3] += i + " "
                if 16381 <= position <= 16489:
                    true_alleles[4] += i + " "
                if 16471 <= position <= 16569 or 1 <= position <= 33:
                    true_alleles[5] += i + " "
                if 19 <= position <= 155:
                    true_alleles[6] += i + " "
                if 133 <= position <= 267:
                    true_alleles[7] += i + " "
                if 259 <= position <= 367:
                    true_alleles[8] += i + " "
                if 339 <= position <= 439:
                    true_alleles[9] += i + " "
                if 438 <= position <= 589 and kit == "easymito":
                    true_alleles[10] += i + " "
                if 428 <= position <= 589 and kit == "mitomini":
                    true_alleles[10] += i + " "
            for key, value in true_alleles.items():
                if value == "":
                    true_alleles[key] += "REF"
                else:
                    true_alleles[key] = value[:-1]
            alleles = list(true_alleles.values())
            final_dict[sample_name] = alleles

    final_df = pd.DataFrame.from_dict(final_dict)
    final_df.to_excel(f"True_alleles/True_alleles_new_dataset_3pers_mixtures_{kit}.xlsx")


def main():
    ''' 
    The following will create a true allele file that is readable by the 
    obtain_hybrids.py script. The file must have the specific overview as in the 
    given file.
    '''
    file = "True_alleles/overzicht donoren mito-profielen_clustered.xlsx"
    kit = 'mitomini'
    get_true_alleles_from_range_per_fragment(file, kit)




if __name__ == "__main__":
    main()
