import pandas as pd
import numpy as np

random_seed = 23

data = pd.read_excel('tcell_table_export_1739072612.xlsx')
data2 = data.copy()

data2 = data2[data2['Related Object - Name'].notna()] # remove empty samples
data2['Related Object - Name'] = data['Related Object - Name'].astype(str) # convert to string

### filtering peptides ###
df = data2[data2['Epitope - Name'].apply(lambda x: 8 <= len(x) <= 13)] # only keeping peptides with 8 to 13 amino acids
df = df[df['Related Object - Name'].apply(lambda x: 8 <= len(x) <= 13)]
df = df[df['MHC Restriction - Name'].str.startswith("HLA-")] # keep only samples that have relevant HLA format
df = df[df['Assay - Method'] == 'biological activity'] # assay type biological activity
df = df[['Epitope - Name','Related Object - Name','MHC Restriction - Name','Assay - Qualitative Measurement']] # columns to keep


### valid Amino Acids annotations ###
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
valid_amino_acids_set = set(AMINO_ACIDS)

def is_valid_sequence(sequence):
    return set(sequence).issubset(valid_amino_acids_set)

df = df[df['Epitope - Name'].apply(is_valid_sequence)] # only keeping peptides with relevant amino acids
df = df[df['Related Object - Name'].apply(is_valid_sequence)]


### annotating Qualitative Measurement values Postive, Postive-high and Positive-intermediete as 1 and Negative and Postive-low as 0 ###
df['Assay - Qualitative Measurement'] = df['Assay - Qualitative Measurement'].apply(lambda x: 0 if x.startswith('N') or x.endswith('w') else 1)


### matching hla column in neoantigen dataset to MHC sequences ###
hla_check_df = pd.read_csv('MHC_Alleles.csv')

def match_seq(hla_check,dataset,mhc_column):
    unique_mhc = dataset[mhc_column].unique()
    unique_mhc_df = pd.DataFrame(unique_mhc, columns=['hla'])
    unique_mhc_df['seq'] = np.nan
    
    unique_mhc_df = unique_mhc_df.merge(hla_check, left_on='hla', right_on='HLA Allele', how='left')
    unique_mhc_df['seq'] = unique_mhc_df['Sequence'].fillna(unique_mhc_df['seq']) 
    unique_mhc_df = unique_mhc_df.drop(columns=['HLA Allele', 'Sequence'])
    
    unique_mhc_df.rename(columns={"hla": "HLA Allele", "seq": "Sequence"}, inplace=True)
    return unique_mhc_df

assay_alleles = match_seq(hla_check_df,df,'MHC Restriction - Name')

df = df.merge(assay_alleles, left_on='MHC Restriction - Name', right_on='HLA Allele', how='left') # merge df with assay_alleles to match sequences with alleles
df = df.drop(columns=['HLA Allele', 'MHC Restriction - Name'])
df = df.dropna()

### merging peptide sequence and HLA sequence ###
df["merged_sequence"] = df["Related Object - Name"].str.ljust(12, "X") + ("|") + df["Epitope - Name"].str.ljust(12, "X") + ("|") + df["Sequence"]
df = df[["merged_sequence", "Assay - Qualitative Measurement"]]


### dealing with duplicates ###
duplicate_sequences = df["merged_sequence"].duplicated(keep=False) # finding sequences that have duplicates
df_filtered = df[(df["Assay - Qualitative Measurement"] == 1) | ~duplicate_sequences] # keeping rows where Assay = 1 OR the sequence is not duplicated
df = df_filtered.reset_index(drop=True)

### balancing classes ###
class_counts = df['Assay - Qualitative Measurement'].value_counts() # getting class counts
min_class_count = class_counts.min()

df_balanced = df.groupby('Assay - Qualitative Measurement', group_keys=False).apply(lambda x: x.sample(min_class_count, random_state=42)) # # undersample the majority class
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) # shuffeling the dataset
df_balanced.to_csv('CEDAR_biological_activity.csv', index=False) # exporting data