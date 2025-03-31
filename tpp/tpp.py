"""Functions for analyzing Thermal Proteome Profiling (TPP) data

Version: 1
Author: Camilla Johansson
"""
import os
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns; sns.set_style("white")
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
import matplotlib.offsetbox as osb
from matplotlib.patches import Patch
from matplotlib.cm import tab10
from matplotlib.colors import to_hex
import requests
import networkx as nx
from unipressed import IdMappingClient
import time
import random 
random.seed(111)
np.random.seed(111)

##### functions for processing TPP data #####

def extractSignifTargets2DTPP(path_2dtpp_excel, cutoff=2, 
                              sheet_name = 1, header=1,
                              passed_filter_columns=['passed_filter']
                             ):
    """Extracting affected proteins from 2DTPP results (excel).

    Parameters
    ---------
    path_2dtpp_excel: str
       Path and file name of results excel file for 2DTPP data.

    cutoff: int
       Minimum number of neighbouring temperature curves with
       good concentration curves. Default is 2

    sheet_name: int
       Specifies which excel sheet (in results file) that contain the
       data. Default is 1.

    header: int
       Specifies which excel row (in results file) that contain 
       the table header. Default is 1. 

    Output
    ------
    protein_all: list
       List of proteins which are either stabilized or destabilized 
       by compound.

    protein_stab: list
       List of proteins which are stabilized by compound.

    protein_destab: list
       List of proteins which are destabilized by compound.
    """
    #Importing 2D-TPP data (from excel results file)
    df_tpp = pd.read_excel(path_2dtpp_excel, 
                           sheet_name = sheet_name, 
                           header=header)
    
    #Filter cell extract file
    for passed_filter_column in passed_filter_columns:
        df_tpp = df_tpp[df_tpp[passed_filter_column] == 'Yes']
    
    #Stabilized proteins
    stab_col_name = 'protein_stabilized_neighb_temp_good_curves_count'
    destab_col_name = 'protein_destabilized_neighb_temp_good_curves_count'
    df_stab = df_tpp[df_tpp[stab_col_name] >= cutoff].copy()
    df_stab = df_stab[df_stab[destab_col_name] == 0].copy()
    
    #Destabilized proteins
    df_destab = df_tpp[df_tpp[destab_col_name] >= cutoff].copy()
    df_destab = df_destab[df_destab[stab_col_name] == 0].copy()

    #Generate output
    protein_stab = removeNoneGeneNames(
        list(set(list(df_stab["protein name"])))
    )
    protein_destab = removeNoneGeneNames(
        list(set(list(df_destab["protein name"])))
    )
    protein_all = list(set(protein_stab + protein_destab))

    return(protein_all, protein_stab, protein_destab)

def extractSignifTargetsTPPTR(path_tpptr_excel, 
                              sheet_name = 1, 
                              header=1, 
                              min_fdr=0.1, 
                              max_fdr=0.2,
                              vehicle_name='DMSO',
                              compound_name='Roscovitin',
                              translate_uniprotID_to_geneNames = True
                             ):
    """Extracting affected proteins from TPP-TR results (excel).
    Parameters
    ---------
    path_tpptr_excel: str
       Path and file name of results excel file for 2DTPP data.

    sheet_name: int
       Specifies which excel sheet (in results file) that contain
       the data. Default is 1.

    header: int
       Specifies which excel row (in results file) that contain 
       the table header. Default is 1. 

    Output
    ------
    protein_all: list
       List of proteins which are either stabilized or destabilized 
       by compound.

    protein_stab: list
       List of proteins which are stabilized by compound.

    protein_destab: list
       List of proteins which are destabilized by compound.
    """

    #Importing TPP-TR data (from excel results file)
    df_tpp = pd.read_excel(path_tpptr_excel, 
                           sheet_name = sheet_name, header=header)
    
    #Keep only instances where vehicle curve plateau < 0.3
    plateau_col1 = 'plateau_'+ vehicle_name + '_1'
    plateau_col2 = 'plateau_'+ vehicle_name + '_2'
    df_tpp_out = df_tpp.loc[df_tpp[plateau_col1] < 0.3, :]
    df_tpp_out = df_tpp_out.loc[df_tpp_out[plateau_col2] < 0.3, :]

    # Keep only instances where minimum slope in both control vs
    # treatment experiments is < -0.06.
    df_tpp_out = df_tpp_out.loc[df_tpp_out['minSlopes_less_than_0.06'] == 'Yes', :]

    # Keep only instances where melt point difference in control vs
    # treatment is larger than melt point difference between controls.
    df_tpp_out = df_tpp_out.loc[df_tpp_out['meltP_diffs_T_vs_V_greater_V1_vs_V2'] == 'Yes', :]
    
    # Keep only instances where melt point shift in both experiments
    # have the same sign.
    df_tpp_out = df_tpp_out.loc[df_tpp_out['meltP_diffs_have_same_sign'] == 'Yes', :]

    # Keep only instances where melting curves for both conditions
    # have R2 > 0.8. 
    R_sq_compound_col1 = 'R_sq_'+ compound_name +'_1'
    R_sq_compound_col2 = 'R_sq_'+ compound_name +'_2'
    R_sq_vehicle_col1 = 'R_sq_'+ vehicle_name +'_1'
    R_sq_vehicle_col2 = 'R_sq_'+ vehicle_name +'_2'
    df_tpp_out = df_tpp_out.loc[df_tpp_out[R_sq_compound_col1] > 0.8, :]
    df_tpp_out = df_tpp_out.loc[df_tpp_out[R_sq_compound_col2] > 0.8, :]
    df_tpp_out = df_tpp_out.loc[df_tpp_out[R_sq_vehicle_col1] > 0.8, :]
    df_tpp_out = df_tpp_out.loc[df_tpp_out[R_sq_vehicle_col2] > 0.8, :]

    
    # Keep only instances where melt curve shift FDR 
    # for replicate with lowest FDR < min_fdr
    # Keep only instances where melt curve shift FDR 
    # for replicate with highest FDR < max_fdr
    index_keep = []
    pVal_compound_vs_vehicle_col1 = 'pVal_adj_'+compound_name+'_1_vs_'+vehicle_name+'_1'
    pVal_compound_vs_vehicle_col2 = 'pVal_adj_'+compound_name+'_2_vs_'+vehicle_name+'_2'
    for i in df_tpp_out.index:
        fdr_rep1 = df_tpp_out.loc[i, pVal_compound_vs_vehicle_col1]
        fdr_rep2 = df_tpp_out.loc[i, pVal_compound_vs_vehicle_col2]
        if fdr_rep1 < fdr_rep2: 
            if fdr_rep1 <= min_fdr:
                if fdr_rep2 <= max_fdr:
                    index_keep = index_keep + [i]
        else: 
            if fdr_rep2 <= min_fdr:
                if fdr_rep1 <= max_fdr:
                    index_keep = index_keep + [i]
                
    df_tpp_out = df_tpp_out.loc[index_keep, :]

    
    # Map protein Uniprot IDs to gene names.
    diff_meltP_compound_vs_vehicle_col1 = 'diff_meltP_'+compound_name+'_1_vs_'+vehicle_name+'_1'
    proteins_stab = list(
        df_tpp_out.loc[df_tpp_out[diff_meltP_compound_vs_vehicle_col1] > 0,
        'Protein_ID']
    )
    proteins_destab = list(
        df_tpp_out.loc[df_tpp_out[diff_meltP_compound_vs_vehicle_col1] < 0,
        'Protein_ID']
    )

    if translate_uniprotID_to_geneNames == True:
        counter = 0
        proteins_stab_out = []
        proteins_destab_out = []
        
        for i in proteins_stab:
            i_list = i.split('.')
            if len(i_list) > 0:
                i_list = i_list[0]
                proteins_stab_out = proteins_stab_out + [i_list]
            counter += 1
        for i in proteins_destab:
            i_list = i.split('.')
            if len(i_list) > 0:
                i_list = i_list[0]
                proteins_destab_out = proteins_destab_out + [i_list]
            counter += 1
    
        #Retrieve gene names from uniprot
        request = IdMappingClient.submit(
            source="UniProtKB_AC-ID", dest="Gene_Name", ids=proteins_stab_out
        )
        time.sleep(5)
        proteins_stab_out = []
        for i in list(request.each_result()):
            gene = i['to']
            proteins_stab_out = proteins_stab_out + [gene]
    
        request = IdMappingClient.submit(
            source="UniProtKB_AC-ID", dest="Gene_Name", ids=proteins_destab_out
        )
        time.sleep(5)
        proteins_destab_out = []
        for i in list(request.each_result()):
            gene = i['to']
            proteins_destab_out = proteins_destab_out + [gene]
    else:
        proteins_stab_out = proteins_stab
        proteins_destab_out = proteins_destab
    proteins_all_out = proteins_stab_out + proteins_destab_out
    return(proteins_all_out, proteins_stab_out, proteins_destab_out)

def createTargetTable_2DTPP(path_2dtpp_excel, cutoff=2, 
                              sheet_name = 1, header=1,
                              passed_filter_columns=['passed_filter'],
                              save_latex_table = False,
                              save_path=None,
                              file_name=None):
    """Outputs table with significant targets and curve parameters

    Can output table as either a pandas data frame, save as a .csv or
    create a LaTeX table for publication. 

    Parameters
    ---------
    path_2dtpp_excel: str
       Path and file name of results excel file for 2DTPP data.

    cutoff: int
       Minimum number of neighbouring temperature curves with
       good concentration curves. Default is 2

    sheet_name: int
       Specifies which excel sheet (in results file) that contain the
       data. Default is 1.

    header: int
       Specifies which excel row (in results file) that contain 
       the table header. Default is 1. 

    Output
    ------
    protein_all: list
       List of proteins which are either stabilized or destabilized 
       by compound.

    protein_stab: list
       List of proteins which are stabilized by compound.

    protein_destab: list
       List of proteins which are destabilized by compound.
    """
    #Importing 2D-TPP data (from excel results file)
    df_tpp = pd.read_excel(path_2dtpp_excel, 
                           sheet_name = sheet_name, 
                           header=header)
    
    #Filter cell extract file
    for passed_filter_column in passed_filter_columns:
        df_tpp = df_tpp[df_tpp[passed_filter_column] == 'Yes']
    
    #Stabilized proteins
    stab_col_name = 'protein_stabilized_neighb_temp_good_curves_count'
    destab_col_name = 'protein_destabilized_neighb_temp_good_curves_count'
    df_stab = df_tpp[df_tpp[stab_col_name] >= cutoff].copy()
    df_stab = df_stab[df_stab[destab_col_name] == 0].copy()
    
    #Destabilized proteins
    df_destab = df_tpp[df_tpp[destab_col_name] >= cutoff].copy()
    df_destab = df_destab[df_destab[stab_col_name] == 0].copy()

    #Generate output
    df_out = pd.DataFrame.from_dict({"Gene names": list(set(list(df_stab["protein name"])))+list(set(list(df_destab["protein name"]))),
                                    "Uniprot ID": ["NA"]*len(list(set(list(df_stab["protein name"]))))+["NA"]*len(list(set(list(df_destab["protein name"])))),
                                    "Stabilized/Destabilized": ["Stabilized"]*len(list(set(list(df_stab["protein name"]))))+["Destabilized"]*len(list(set(list(df_destab["protein name"])))),
                                    "$pEC_{50}$": [""]*len(list(set(list(df_stab["protein name"]))))+[0]*len(list(set(list(df_destab["protein name"])))),
                                    "No. neighb. temp. curves":  [cutoff]*len(list(set(list(df_stab["protein name"]))))+[cutoff]*len(list(set(list(df_destab["protein name"])))),
                                    })

    #Fill output data frame with pEC50
    for gene in list(df_out["Gene names"]):
        if len(df_stab.loc[df_stab["protein name"] == gene,stab_col_name]) > 0:
            #Fill output data with number of neighboring temperature curves
            df_out.loc[df_out["Gene names"] == gene,"No. neighb. temp. curves"] = list(df_stab.loc[df_stab["protein name"] == gene,stab_col_name])[0]
            #Fill output data with pEC50
            EC50_min = min(list(df_stab.loc[df_stab["protein name"] == gene,"pEC50"]))
            EC50_max = max(list(df_stab.loc[df_stab["protein name"] == gene,"pEC50"]))
            EC50 = str(EC50_min)+' - '+str(EC50_max)
            df_out.loc[df_out["Gene names"] == gene,"$pEC_{50}$"] = EC50
        if len(df_destab.loc[df_destab["protein name"] == gene,destab_col_name]) > 0:
            #Fill output data with number of neighboring temperature curves
            df_out.loc[df_out["Gene names"] == gene,"No. neighb. temp. curves"] = list(df_destab.loc[df_destab["protein name"] == gene,destab_col_name])[0]
            #Fill output data with pEC50
            EC50_min = min(list(df_destab.loc[df_destab["protein name"] == gene,"pEC50"]))
            EC50_max = max(list(df_destab.loc[df_destab["protein name"] == gene,"pEC50"]))
            EC50 = str(EC50_min)+' - '+str(EC50_max)
            df_out.loc[df_out["Gene names"] == gene,"$pEC_{50}$"] = EC50

    #Retrieve gene names from uniprot (OBS, might not always retrieve canonical sequence)
    request = IdMappingClient.submit(
        source="Gene_Name", dest="UniProtKB", ids=list(df_out["Gene names"]), taxon_id=9606
    )
    time.sleep(15)

    for i in list(request.each_result()):
        gene = i['from']
        prot = i['to']
        df_out.loc[df_out["Gene names"] == gene,"Uniprot ID"] = prot

    #Save as latex table
    if save_latex_table == True:
        if save_path is None or file_name is None:
            raise ValueError("save_path and file_name must be specified to save LaTeX table")
        save_file = save_path+file_name
        df_out = df_out.to_latex(buf=save_file, index=False)

    return(df_out)

def removeNoneGeneNames(protein_list):
    """Removes entries which are not real gene names.

    Parameters
    ----------
    protein_list: list
        List of protein names (gene names)

    Output
    ------
    protein_list: list
        List of protein names with only approves gene names. 
    """
    for i in protein_list:
        if '_HUMAN' in i:
            protein_list.remove(i)
        if 'KDA PROTEIN.' in i:
            protein_list.remove(i)
    return(protein_list)

def import_enrichment(protein_list):
    """Import GO molecular function data from string-db.

    Parameters
    ----------
    protein_list: list
        List of protein names (gene names)

    Output
    ------
    enrichment: pandas.DataFrame
        DataFrame with GO molecular function data and 
        other network enrichment data retrieved from string-db.
        Columns:
            category: str
            term: str
            number_of_genes: int
            number_of_genes_in_background: int
            ncbiTaxonId: int
            inputGenes: str, comma separated
            preferredNames: str, comma separated
            p_value: float
            fdr: float
            description: str
         
    
    """
    string_api_url = "https://version-11-9.string-db.org/api"
    output_format = "tsv"
    method = "enrichment"
    
    request_url = "/".join([string_api_url, output_format, method])
    params = {
    
        "identifiers" : "%0d".join(protein_list), # your protein list
        "species" : 9606, # species NCBI identifier 
    }
    
    r = requests.post(request_url, data=params)

    # pull the text from the response object and split based on new lines
    lines = r.text.split('\n')
    # split each line into its components based on tabs
    data = [l.split('\t') for l in lines] 
    # convert to dataframe using the first row as the column names; 
    # drop empty, final row
    df = pd.DataFrame(data[1:-1], columns = data[0]) 
    enrichment = df
    return(enrichment)


def get_tpp_protein(G, protein_list):
    """Takes TTP protein list and communities, returns node sizes

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        NetworkX network graph for a community.

    protein_list: list
        List of protein names (gene names) significantly changed
        in TPP experiment. 
    
    Output
    ------
    list of node sizes
    """
    ### CODE ###
    size_dict = dict.fromkeys(G.nodes(), 200)

    for i in list(size_dict.keys()):
        if i in protein_list:
            size_dict[i] = 1600
            
    return(list(size_dict.values()))

#Define function which takes any list of labels and translates to a set of colour labels for a network

def get_color_labels(G, feature_dict, feature_df):
    """Translate list of labels to a set of color lables for a network.

    Parameters
    ----------
    G: networkx.classes.graph.Graph
        NetworkX network graph for a community.

    feature_dict: dict

    feature_df: pandas.DataFrame
    
    Output
    ------
    """
    col_dict = dict.fromkeys(G.nodes(), '#def5ff')
    #cols = ['#a2fcc3', '#76e8a0', '#3cc26d', '#15b04e', '']
    for i in list(col_dict.keys()):
        #print(feature_df['Gene name'])
        if i in list(feature_df['Gene name']):
            #print(i)
            locations = list(feature_df[feature_df['Gene name'] == i]['Main location'])[0].split(';')
            #print(locations)
            for c in feature_dict.keys():
                if c in locations:
                    col_dict[i] = feature_dict[c]
    return(list(col_dict.values()))

def make_sl_histogram(protein_list, feature_df, file_name=None):
    
    #Create dataframe for histogram
    df_histogram = pd.DataFrame()
    df_histogram["Compartments"] = ["Nucleus", 
                                    "Cytosol", 
                                    "Mitochondria", 
                                    "Endoplasmic reticulum", 
                                    "Golgi", 
                                    "Actin filament and intermediate filament", 
                                    "Centrosome and microtubules", 
                                    "Vesicles", 
                                    "Plasma membrane"]
    
    df_histogram["Numb proteins"] = [0]*len(df_histogram["Compartments"])
    
    for i in protein_list:
        comp_list = str(list(feature_df[feature_df["Gene name"] == i]["Main location"]))[2:-2].split(";")
        for c in comp_list:
            if c != '':
                # Nucleus
                if c in ['Nuclear membrane',
                         'Nucleoli', 
                         'Nucleoli fibrillar center', 
                         'Nucleoli rim', 
                         'Nucleoplasm', 
                         'Nuclear speckles', 
                         'Nuclear bodies', 
                         'Mitotic chromosome', 
                         'Kinetochore']:
                    df_histogram.loc[0,"Numb proteins"] += 1
                
                # Cytosol
                if c in ['Aggresome', 
                         'Cytoplasmic bodies', 
                         'Cytosol', 
                         'Rods & rings']:
                    df_histogram.loc[1,"Numb proteins"] += 1
                
                # Mitochondria
                if c in ['Mitochondria']:
                    df_histogram.loc[2,"Numb proteins"] += 1
               
                # Endoplasmic reticulum
                if c in ['Endoplasmic reticulum']:
                    df_histogram.loc[3,"Numb proteins"] += 1
                
                # Golgi
                if c in ['Golgi apparatus']:
                    df_histogram.loc[4,"Numb proteins"] += 1
                
                # Actin filament and intermediate filament
                if c in ['Actin filaments', 
                         'Cleavage furrow', 
                         'Focal adhesion sites', 
                         'Intermediate filaments']:
                    df_histogram.loc[5,"Numb proteins"] += 1

                # Centrosome and microtubules
                if c in ['Centriolar satellite', 
                         'Centrosome', 
                         'Cytokinetic bridge', 
                         'Microtubule ends', 
                         'Microtubules', 
                         'Midbody', 
                         'Midbody ring', 
                         'Mitotic spindle']:
                    df_histogram.loc[6,"Numb proteins"] += 1

                # Vesicles
                if c in ['Endosomes', 
                         'Lipid droplets', 
                         'Lysosomes', 
                         'Peroxisomes', 
                         'Vesicles']:
                    df_histogram.loc[7,"Numb proteins"] += 1

                # Plasma membrane
                if c in ['Cell junctions', 
                         'Plasma membrane']:
                    df_histogram.loc[8,"Numb proteins"] += 1
                    
        comp_list = str(list(feature_df[feature_df["Gene name"] == i]["Additional location"]))[2:-2].split(";")
        for c in comp_list:
            if c != '':
                # Nucleus
                if c in ['Nuclear membrane',
                         'Nucleoli', 
                         'Nucleoli fibrillar center', 
                         'Nucleoli rim', 
                         'Nucleoplasm', 
                         'Nuclear speckles', 
                         'Nuclear bodies', 
                         'Mitotic chromosome', 
                         'Kinetochore']:
                    df_histogram.loc[0,"Numb proteins"] += 1
                
                # Cytosol
                if c in ['Aggresome', 
                         'Cytoplasmic bodies', 
                         'Cytosol', 
                         'Rods & rings']:
                    df_histogram.loc[1,"Numb proteins"] += 1
                
                # Mitochondria
                if c in ['Mitochondria']:
                    df_histogram.loc[2,"Numb proteins"] += 1
               
                # Endoplasmic reticulum
                if c in ['Endoplasmic reticulum']:
                    df_histogram.loc[3,"Numb proteins"] += 1
                
                # Golgi
                if c in ['Golgi apparatus']:
                    df_histogram.loc[4,"Numb proteins"] += 1
                
                # Actin filament and intermediate filament
                if c in ['Actin filaments', 
                         'Cleavage furrow', 
                         'Focal adhesion sites', 
                         'Intermediate filaments']:
                    df_histogram.loc[5,"Numb proteins"] += 1

                # Centrosome and microtubules
                if c in ['Centriolar satellite', 
                         'Centrosome', 
                         'Cytokinetic bridge', 
                         'Microtubule ends', 
                         'Microtubules', 
                         'Midbody', 
                         'Midbody ring', 
                         'Mitotic spindle']:
                    df_histogram.loc[6,"Numb proteins"] += 1

                # Vesicles
                if c in ['Endosomes', 
                         'Lipid droplets', 
                         'Lysosomes', 
                         'Peroxisomes', 
                         'Vesicles']:
                    df_histogram.loc[7,"Numb proteins"] += 1

                # Plasma membrane
                if c in ['Cell junctions', 
                         'Plasma membrane']:
                    df_histogram.loc[8,"Numb proteins"] += 1

    # Reorder it based on number of proteins:
    ordered_df_histogram = df_histogram.sort_values(by='Numb proteins')
    my_range=range(1,len(df_histogram.index)+1)
    
    # Horizontal version
    fig, ax = plt.subplots()
    ax.hlines(y=my_range, xmin=0, xmax=ordered_df_histogram['Numb proteins'], color='skyblue')
    ax.plot(ordered_df_histogram['Numb proteins'], my_range, "o")
    ax.set_yticks(my_range, ordered_df_histogram['Compartments'])
    ax.set_xlabel("Number of proteins")
    ax.set_xticks(np.arange(0, 30, 2))
    ax.set_title("TPP proteins per subcellular location")
    fig.show()

    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(file_name, bbox_inches='tight')

def retrievePreliminaryNetwork(protein_list, network_type="physical", add_nodes=0):
    """Retrieve and plot a preliminary PPI network.

    Based on plot from this function, manually decide how to
    partition the final network into subnetworks.

    Parameters
    ----------
    protein_list: list
        List of proteins to include in network. Should be
        gene names. 

    network_type: "physical" or "functional"
        Defines what type of PPI network should be retrieved from
        the STRING-DB. See STRING-DB documentation for an explanation: 
        https://string-db.org/cgi/help?sessionId=bRJlwy4ZVwJf
        Default is "physical".

    Output
    ------
    G: networkx.classes.graph.Graph object
        Object containing the NetworkX Graph with proteins.
    
    interactions: numpy.array
        Object containing the edge information of the network, 
        i.e. which nodes are connected to each other and the edge
        total score as retrieved from STRING db. 
    """
    
    ### CODE ###
    #Importing network
    proteins = '%0d'.join(protein_list)
    
    #Setting parameters for STRING API
    string_api_url = "https://version-11-9.string-db.org/api"
    output_format = "tsv"
    method = "network"

    if network_type == 'physical':
        required_score = 150
    else:
        required_score = 400
    
    request_url = "/".join([string_api_url, output_format, method])
    params = {
    
        "identifiers" : proteins, # your protein list
        "species" : 9606, # species NCBI identifier 
        "limit" : 1, # only one (best) identifier per input protein
        "add_nodes": add_nodes, # add x white nodes to my protein
        "network_type": network_type,
        "required_score": required_score
    }

    #Request network from STRING-DB
    r = requests.post(request_url, data=params)

    #Clean up retrieved data
    lines = r.text.split('\n') # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines] # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df = pd.DataFrame(data[1:-1], columns = data[0]) 
    # dataframe with the preferred names of the two proteins and the score of the interaction
    interactions = df[['preferredName_A', 'preferredName_B', 'score']]

    #Creating the network
    G=nx.Graph(name='Protein Interaction Graph')
    interactions = np.array(interactions)
    for i in range(len(interactions)):
        interaction = interactions[i]
        a = interaction[0] # protein a node
        b = interaction[1] # protein b node
        w = float(interaction[2]) # score as weighted edge where high scores = low weight
        G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph
    
    #Plot whole network
    pos = nx.spring_layout(G) # position the nodes using the spring layout
    plt.figure(figsize=(8,8),facecolor='white')
    nx.draw_networkx(G, pos)
    plt.axis('off')
    plt.show()

    return(G, interactions)

def plotPPINetwork(G_in, interactions, community, protein_list, external_data, 
                   label_method="subcellular", file_name=None, celline='U2OS', figure_size=(12,12), 
                   cluster_count_df=None, tpp_type='2DTPP'):
    """Retrieves and plots a PPI network.

    TODO: Add more description here

    Parameters
    ----------
    G_in: networkx.classes.graph.Graph object
       Object containing the NetworkX Graph generated using
       retrievePreliminaryNetwork().

    interactions: numpy.array
       Object containing all edges in network. Generated by
       retrievePreliminaryNetwork().

    community: int
       Number of separate (not connected) networks observed in 
       preliminary network graph. 

    protein_list: list
       List of proteins to include in network. Should be
       gene names.

    external_data: dict
       Dictionary containing all valid label_method values as keys
       and a list of length 1 or 2 as values. 
       First element of vector contain file paths as str. 
       Second element contains a dictionary of optional variables for reading file.

    label_method: str. "subcellular", "cell line", "cluster count"
                  "num temp curves", "direction", "other TPP data"
       Argument deciding which method will be used to
       color/label the network. Default is "subcellular".

    file_name: str or None
       If not None, saves the heatmap to file_name.

    celline: str
       Name of cell line to color nodes by when label_method="cell line".
       Default is "U2OS". 

    cluster_count_df: pandas.DataFrame or None
        Dataframe for number of times a target is present in CP
        cluster. Default is None. 

    tpp_type: str. "2DTPP" or "TPP-TR".
        Can be either one of two categories. Specifies what type of
        TPP experiment has been run. 
    
    Output
    ------
    between_g: Betweenness centrality for network. 
    """

    ### ERRORS ###
    #Check so that label_method is correct: 
    valid_methods = {"subcellular", "cell line", "num temp curves", "direction", "other TPP data", "cluster count"}
    if label_method not in valid_methods:
        raise ValueError("label_method must be one of %r." % valid_methods)
    if label_method == "cluster count":
        if cluster_count_df is None:
            raise ValueError("When label_method is set to 'cluster count', cluster_count_df cannot be missing")
    
    ### CODE ###
    # Clustering network into communitites using Edge Betweeness partition
    communities = nx.community.edge_betweenness_partition(G_in, community)

    if label_method == "subcellular":
        col_dict = {'Nuclear membrane': '#96f2b8', #Col 1
                    'Nucleoli' : '#3cc26d', 'Nucleoli fibrillar center' : '#3cc26d', 'Nucleoli rim' : '#3cc26d', #Col 2
                    'Nucleoplasm': '#016e29', 'Nuclear speckles' : '#016e29', 'Nuclear bodies' : '#016e29', 'Mitotic chromosome' : '#016e29', 'Kinetochore' : '#016e29', #Col 3
                    'Mitochondria': '#689cfc', #Col 4
                    'Endoplasmic reticulum': '#c168fc', 
                    'Aggresome': '#fc783f', 'Cytoplasmic bodies': '#fc783f', 'Cytosol': '#fc783f', 'Rods & rings': '#fc783f'} #Col 5
        
        col_dict_short = {'Nuclear membrane': '#96f2b8', #Col 1
                         'Nucleoli' : '#3cc26d', # Col 2
                         'Nucleoplasm': '#016e29', #Col 3
                         'Mitochondria': '#689cfc', #Col 4
                          'Endoplasmic reticulum': '#c168fc', 
                          'Cytosol': '#fc783f'} #Col 5
    
    c = 0
    
    between_g = dict()
    #Plot network with shape and color labels
    for com_sub in communities:
        #Add GO molecular function enrichment data to each network
        enrichment = import_enrichment(protein_list=list(com_sub))
        enrich_process = str(list(enrichment[enrichment["category"] == "Process"]["description"].head(1)))[2:-2]
        enrich_function = str(list(enrichment[enrichment["category"] == "Function"]["description"].head(1)))[2:-2]
        enrich_component = str(list(enrichment[enrichment["category"] == "Component"]["description"].head(1)))[2:-2]
        title_nw = "GO BP: " + enrich_process + "\nGO MF: " + enrich_function + "\nGO CC: " + enrich_component
    
    
        com_sub = list(com_sub)
        interactions_sub = []
        for i in com_sub:
            for j in interactions:
                if i == j[1]:
                    j_list = j.tolist()
                    col_group = '1' #group for changing colour of network. Should probably move this to before subsetting
                    j_list.append(col_group)
                    interactions_sub.append(j_list)
        
        interactions_sub = np.array(interactions_sub, dtype=object)
    
        #Create weight list
        w_list = []
        for i in range(len(interactions_sub)):
            w_list.append(float(interactions_sub[i][2])*3)
        
        #Generate network
        G=nx.Graph(name='Protein Interaction Graph')
        interactions_sub = np.array(interactions_sub)
        for i in range(len(interactions_sub)):
            interaction = interactions_sub[i]
            a = interaction[0] # protein a node
            b = interaction[1] # protein b node
            w = float(interaction[2]) # score as weighted edge where high scores = low weight
            G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph
    
        #Add tpp data as node size
        size_list = get_tpp_protein(G, protein_list)

        #Add node color based on label_method
        if not label_method == 'cluster count':
            external_df = readExternalDataFile(external_data, label_method)
        if label_method == "subcellular":
            #Add node color from coloring parametre
            col_list = get_color_labels(G, col_dict, external_df)
        
        if label_method == "cell line":
            #Add node color from coloring parametre
            external_df = external_df[external_df["Cell line"] == celline]
            pTPM_dict = dict.fromkeys(G.nodes(), np.nan)
            for i in list(pTPM_dict.keys()):
                if i in list(external_df["Gene name"]):
                    pTPM_dict[i] = external_df[external_df["Gene name"] == i]["pTPM"]
            col_list = list(pTPM_dict.values())

        if label_method == "num temp curves":
            #Check so that correct data type
            if tpp_type == 'TPP-TR':
                raise ValueError(
                    "Number of neighboring temperature curves cannot be calcualted for TPP-TR data."
                )
            #Calculate df_prot
            df_prot = calcNumbTempWindows(external_df, protein_list)
            #Add node color from coloring parametre 
            pTPM_dict = dict.fromkeys(G.nodes(), np.nan)
            for i in list(pTPM_dict.keys()):
                if i in list(df_prot["Gene_name"]):
                    pTPM_dict[i] = float(
                        df_prot[df_prot["Gene_name"] == i]["Numb_neighb_temp_windows"]
                    ) #This gives warning now. 
            col_list = list(pTPM_dict.values())

        if label_method == "direction":
            if tpp_type == '2DTPP':
                path_2dtpp_excel = external_data['direction'][0]
                protein_list2, protein_stab, protein_destab = extractSignifTargets2DTPP(
                    path_2dtpp_excel, cutoff=2, **external_data['direction'][1])
            if tpp_type == 'TPP-TR':
                path_tpptr_excel = external_data['direction'][0]
                protein_list2, protein_stab, protein_destab = extractSignifTargetsTPPTR(
                    path_tpptr_excel, **external_data['direction'][1], **external_data['direction'][2])
            #Add node color from coloring parametre
            col_dict = dict.fromkeys(G.nodes(), '#def5ff')
            col_dict_short = {'Stabilized': '#3cc26d', 'Destabilized': '#fc783f'}
            for i in list(col_dict.keys()):
                if i in protein_stab:
                    col_dict[i] = '#3cc26d'
                if i in protein_destab:
                    col_dict[i] = '#fc783f'
            col_list = list(col_dict.values())

        if label_method == "other TPP data":
            other_param = external_data['other TPP data'][2]
            if tpp_type == '2DTPP':
                path_2dtpp_excel = external_data['other TPP data'][0]
                protein_list2, protein_stab, protein_destab = extractSignifTargets2DTPP(
                    path_2dtpp_excel, **other_param, **external_data['other TPP data'][1])
            if tpp_type == 'TPP-TR':
                path_tpptr_excel = external_data['direction'][0]
                protein_list2, protein_stab, protein_destab = extractSignifTargetsTPPTR(
                    path_tpptr_excel, **external_data['other TPP data'][1], **other_param)
            col_dict = dict.fromkeys(G.nodes(), '#def5ff')
            col_dict_short = {'Stabilized': '#3cc26d', 'Destabilized': '#fc783f'}
            for i in list(col_dict.keys()):
                if i in protein_stab:
                    col_dict[i] = '#3cc26d'
                if i in protein_destab:
                    col_dict[i] = '#fc783f'
            col_list = list(col_dict.values()) 

        if label_method == 'cluster count':
            #Add node color from coloring parametre 
            pTPM_dict = dict.fromkeys(G.nodes(), np.nan)
            for i in list(pTPM_dict.keys()):
                if i in list(cluster_count_df["Target"]):
                    pTPM_dict[i] = float(cluster_count_df[cluster_count_df["Target"] == i]["Count"]) #This gives warning now. 
            col_list = list(pTPM_dict.values())
        
        #Plot network
        pos = nx.kamada_kawai_layout(G) # position the nodes using the kamada_kawai layout
        #pos = nx.spring_layout(G) # position the nodes using the spring layout
        plt.figure(figsize=figure_size,facecolor='white')
        nx.draw_networkx(G, pos, node_size=size_list, 
                         #node_color = col_list,
                         node_color = '#def5ff', 
                         font_size=10,
                        edge_color="grey",
                        width = w_list, 
                         bbox=dict(color="white",  
                                   alpha=0.7, 
                                   boxstyle="Round, pad=0.0"))
        nc = nx.draw_networkx_nodes(G, pos, node_size=size_list, node_color = col_list,
                                    cmap=plt.cm.YlOrRd, vmin = 0)
        plt.suptitle(title_nw)
        plt.axis('off')

        #Plot legend based on label_method
        if label_method in ["subcellular","direction", "other TPP data"]:    
            patch = [Patch(color=color, label=col_dict.keys()) for color in col_dict_short.values()]
            plt.legend(patch, col_dict_short.keys(), numpoints=1)

        if label_method == "cell line":
            plt.colorbar(nc,label="pTPM in "+celline+" cells")

        if label_method == "num temp curves":
            plt.colorbar(nc,label="Numb temperature windows where\nprotein is stabilized/destabilized")

        if label_method == "cluster count":
            plt.colorbar(nc,label="Number of times protein appear in CP cluster")

        #Save plot to file if file_name is defined. 
        if not file_name is None:
            file_name2 = file_name.split('.')
            if not len(file_name2) == 2:
                raise ValueError("file_name should not contain more than one dot ('.')")
            file_name2 = file_name2[0] + '_' + str(c) + '.' + file_name2[1]
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(file_name2)

        #Calculate betweeness centrality for network
        between_g[c] = nx.betweenness_centrality(G)
    
        c += 1
        plt.show()
    return(between_g)

def plotPPINetworkCommunitites(G_in, 
                               interactions, 
                               community,
                               protein_list,
                               method='edge_betweenness_partition',
                               file_name=None, 
                               figure_size=(12,12)):
    """Retrieves and plots a PPI network, colored by community.

    TODO: Add more description here

    Parameters
    ----------
    G_in: networkx.classes.graph.Graph object
       Object containing the NetworkX Graph generated using
       retrievePreliminaryNetwork().

    interactions: numpy.array
       Object containing all edges in network. Generated by
       retrievePreliminaryNetwork().

    community: int
       Number of separate (not connected) networks observed in 
       preliminary network graph.

    method: str
        Type of networkX method for community identification. 
        Approved methods: 
            "edge_betweenness_partition", 
            "greedy_modularity_communities",
            "k_clique_communities",
            "girvan_newman"

    protein_list: list
       List of proteins to include in network. Should be
       gene names.

    file_name: str or None
       If not None, saves the heatmap to file_name.

    figure_size: tuple of ints
       Size of output figure, default (12,12). 
    
    Output
    ------
    enrichment: Betweenness centrality for network. 
    """

    ### ERRORS ###
    #Check so that method is correct: 
    valid_methods = {"edge_betweenness_partition", 
                     "greedy_modularity_communities",
                     "k_clique_communities", #Not always working
                     "girvan_newman"
                    }
    if method not in valid_methods:
        raise ValueError("method must be one of %r." % valid_methods)
        
    ### CODE ###
    # Clustering network into communitites using method
    if method == "edge_betweenness_partition":
        communities = nx.community.edge_betweenness_partition(G_in, community)
    if method == "greedy_modularity_communities":
        communities = nx.community.greedy_modularity_communities(G_in)
    if method == "k_clique_communities":
        communities = nx.community.k_clique_communities(G_in, community)
    if method == "girvan_newman":
        communities = list(nx.community.girvan_newman(G_in))[community-2]
        
    c = 0
    
    between_g = dict()

    #Add node color based on label_method
    col_dict = dict.fromkeys(G_in.nodes(), '#def5ff')
    col_dict_short = {}
    colors = tab10(range(community))

    counter = 0
    #Plot network with shape and color labels

    enrichment = dict()
    
    for com_sub in communities:
        #Add GO molecular function enrichment data to each network
        enrichment['Community '+str(counter+1)] = import_enrichment(protein_list=list(com_sub))
        #enrich_process = str(list(enrichment[enrichment["category"] == "Process"]["description"].head(1)))[2:-2]
        #enrich_function = str(list(enrichment[enrichment["category"] == "Function"]["description"].head(1)))[2:-2]
        #enrich_component = str(list(enrichment[enrichment["category"] == "Component"]["description"].head(1)))[2:-2]
        #title_nw = "GO BP: " + enrich_process + "\nGO MF: " + enrich_function + "\nGO CC: " + enrich_component
    
    
        com_sub = list(com_sub)
    
        #Add tpp data as node size
        size_list = get_tpp_protein(G_in, protein_list)

        #Add node color 
        col_dict_short['Community '+str(counter+1)] = to_hex(colors[counter])
        for i in list(col_dict.keys()):
            if i in com_sub:
                col_dict[i] = to_hex(colors[counter])

        counter += 1
        
        
    #Plot network
    pos = nx.kamada_kawai_layout(G_in) # position the nodes using the kamada_kawai layout
    #pos = nx.spring_layout(G) # position the nodes using the spring layout
    plt.figure(figsize=figure_size,facecolor='white')
    nx.draw_networkx(G_in, pos, node_size=size_list, 
                     #node_color = col_list,
                     node_color = '#def5ff', 
                     font_size=10,
                    edge_color="grey",
                    #width = w_list, 
                     bbox=dict(color="white",  
                               alpha=0.7, 
                               boxstyle="Round, pad=0.0"))
    nc = nx.draw_networkx_nodes(G_in, pos, node_size=size_list, node_color = list(col_dict.values()),
                                cmap=plt.cm.YlOrRd, vmin = 0)
    #plt.suptitle(title_nw)
    plt.axis('off')
    patch = [Patch(color=color, label=col_dict.keys()) for color in col_dict_short.values()]
    plt.legend(patch, col_dict_short.keys(), numpoints=1)

    #Save plot to file if file_name is defined. 
    if not file_name is None:
        file_name2 = file_name.split('.')
        if not len(file_name2) == 2:
            raise ValueError("file_name should not contain more than one dot ('.')")
        file_name2 = file_name2[0] + '_' + str(c) + '.' + file_name2[1]
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(file_name2)

    #Calculate betweeness centrality for network
    #between_g[c] = nx.betweenness_centrality(G)
    
    c += 1
    plt.show()
    return(enrichment)

def readExternalDataFile(external_data, method):
    """Function that reads required external data file from path, based on defined method.

    Subcellular: path and file name to tsv file containing subcellular location data from
    the Human Protein Atlas database. 

    Cell line: path and file name to tsv file containing cell line expression data (RNAseq)
    from the Human Protein Atlas database. 

    Num temp curves: path and file name to excel results file for 2DTPP data. 

    Parameters
    ----------
    external_data: dict of vectors
       Dictionary with method as keys and vectors as values. 
       First element of vector contain file paths as str. 
       Second element contains a dictionary of optional variables for reading file.

    method: str
       Can be one of four methods: "subcellular", "cell line",
                  "num temp curves", "direction"

    Output
    ------
    df: pandas.DataFrame
       Returns external data (e.g. subcellular location) based on method.
    
    """

    ### CODE ###
    selected_path = external_data[method][0]
    file_type = selected_path.split('.')[-1]

    if file_type == "tsv":
        df = pd.read_csv(selected_path, sep="\t")
    if file_type == "csv":
        df = pd.read_csv(selected_path)
    if file_type == "xlsx":
        if len(external_data[method]) == 2:
            df = pd.read_excel(selected_path, **external_data[method][1])
        else:
            df = pd.read_excel(selected_path)

    return(df)

def calcNumbTempWindows(protein_df, protein_list):
    """Calculate number of temperature windows from a 2DTPP

    Parameters
    ----------
    protein_df: pandas.DataFrame
       2DTPP or TPP-TR data from results excel file

    protein_list: list
       List of proteins to calculate number of neighbouring
       temp curves for. 
       
    Output
    ------
    df_prot: pandas.DataFrame
       Contains two columns: "Gene name" and "Numb_neighb_temp_windows". 
       Column "Gene name" contains all proteins from protein_list. 
    """
    n_neighb_temp = []
    for p in protein_list:
        stab_col_name = 'protein_stabilized_neighb_temp_good_curves_count'
        destab_col_name = 'protein_destabilized_neighb_temp_good_curves_count'
        n_neighb_temp_stab = list(
            protein_df[protein_df["protein name"] == p][stab_col_name])[0]
        n_neighb_temp_destab = list(
            protein_df[protein_df["protein name"] == p][destab_col_name])[0]
        if n_neighb_temp_stab > n_neighb_temp_destab:
            n_neighb_temp.append(n_neighb_temp_stab)
        else: 
            n_neighb_temp.append(n_neighb_temp_destab)
    df_prot = pd.DataFrame({"Gene_name": protein_list, 
                            "Numb_neighb_temp_windows": n_neighb_temp}
                          )

    return(df_prot)

def findLargestBetweennessCentrality(between_c, protein_list, 
                                     removeList=False, output_size=5, cluster_index=0):
    """Find proteins with largest betweenness centrality from network.

    Takes a dictionary (between_c) of betweenness centrality scores
    as input and returnes the top N proteins with highest scores
    as defined by "output_size". 
    
    If removeList = True, the function first removes all proteins from
    protein_list. This is used in combination with consensus clustering
    when plotting a PPI network for all proteins identified in TPP as well
    as the most central proteins identified in CP. 

    Parameters
    ----------
    between_c: dict
       Betweenness centrality for network. Proteins as 
       keys and betweenness centrality scores as values.

    protein_list: list
       List of significant proteins from TPP data.

    removeList: bool
       If True, proteins in protein_list will be removed from
       returned pandas.DataFrame of proteins with largest 
       betweenness centrality. Default is False.

    output_size: int
       Number of proteins that function will return.
       Default is 5. 

    cluster_index: int
        Index in between_c for the cluster that BC should 
        be calculated on. 

    Output
    ------
    between_df: pandas.DataFrame
       Indexed by proteins (gene names)
       column "0": betweenness centrality score, sorted
       Only the top N proteins are returned, as 
       defined by"output_size". 
    
    """

    ### CODE ###
    between_df = pd.DataFrame.from_dict(between_c[cluster_index], orient='index')
    #Remove all rows where protein is in protein_list
    if removeList is True:
        between_df = between_df.loc[~between_df.index.isin(protein_list)]
    #Sort on highest betweenness centrality
    between_df = between_df.sort_values(by=[0], ascending=False)
    between_df = between_df.iloc[:output_size,:]
    return(between_df)

def findClusterProteinsToKeepInNetwork(between_c, protein_list, interactions, cluster_index=0):
    """Finds Cluster proteins to keep in final network.

    Cluster proteins to keep are based on betweenness centrality scores 
    (between_c). The algorithm first ranks the scores, then finds the cutoff
    value for the scores which preserves the total number of TPP protein nodes.

    Parameters
    ----------
    between_c: dict
        Betweenness centrality for network. Proteins as 
        keys and betweenness centrality scores as values.

    protein_list: list
        List of significant proteins from TPP data.

    interactions: numpy.array
        Object containing the edge information of the network, 
        i.e. which nodes are connected to each other and the edge
        total score as retrieved from STRING db. 

    cluster_index: int
        Index in between_c for the cluster that BC should 
        be calculated on. 

    Returns
    -------
    between_df: pandas.DataFrame
        Indexed by proteins (gene names)
        column "0": betweenness centrality score, sorted
        Only the top N proteins are returned, as 
        defined by"output_size". 
    """

    ### CODE ###
    between_df = pd.DataFrame.from_dict(between_c[cluster_index], orient='index')
    
    #Sort on highest betweenness centrality
    between_df = between_df.sort_values(by=[0], ascending=False)
    
    interactions_temp = interactions.copy()
    # Find number of TPP proteins in interaction list
    prot_tpp_list = []
    for n in range(len(interactions_temp)):
        if interactions_temp[n][0] in protein_list: 
            prot_tpp_list = prot_tpp_list + [interactions_temp[n][0]]
        if interactions_temp[n][1] in protein_list:
            prot_tpp_list = prot_tpp_list + [interactions_temp[n][1]]
    prot_tpp_list = list(set(prot_tpp_list))
    #Remove one node at the time, from bottom up, and check if len(proteins_tpp_list) changes.
    for i in list(reversed(range(len(between_df[0])))):
        #Find number of tpp proteins in interaction list
        prot = between_df.iloc[i,].name #Protein name at row i
        indx = [] #List of rows to remove from interactions_temp
        if prot not in protein_list:
            for n in range(len(interactions_temp)):
                if interactions_temp[n][0] == prot:
                    indx.append(n)
                if interactions_temp[n][1] == prot:
                    indx.append(n)
        #interactions_temp = [x for x in interactions_temp if not x in indx]
        interactions_temp = np.delete(interactions_temp, indx, axis=0)
        prot_tpp_list_i = []
        for n in range(len(interactions_temp)):
            if interactions_temp[n][0] in protein_list: 
                prot_tpp_list_i = prot_tpp_list_i + [interactions_temp[n][0]]
            if interactions_temp[n][1] in protein_list:
                prot_tpp_list_i = prot_tpp_list_i + [interactions_temp[n][1]]
        prot_tpp_list_i = list(set(prot_tpp_list_i))
        if len(prot_tpp_list_i) == len(prot_tpp_list):
            between_df = between_df.drop(index=prot)

    #Remove all proteins which are in protein_list from output
    between_df = between_df.loc[~between_df.index.isin(protein_list)]
        
    
    return(between_df)

def findClusterProteins(df, annotations, cluster_highlight, 
                        compound, CompoundNames):
    """Translate cluster indexes to protein names.

    This script is used to translate the output from a consensus 
    clustering to protein names (gene names), which is needed prior to
    plotting a PPI network. 

    The compound of interest is automatically removed from the cluster
    if present in the cell painting data.

    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing
       with process_cp_to_median().

    annotations: pandas.DataFrame
       Annotation data for compounds. 

    cluster_highlight: list
       List of indexes for compounds in CP data which belong to
       cluster.

    compound: list
       List of compounds to be removed from cluster. 
    
    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of compounds
       of interest as values. Should only contain compounds which 
       should be displayed in UMAP legend.
       
    """
    df = df.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')

    #Modify df index to be sorted by cbkid
    df_sorted = df.copy()

    #df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames)
    df_sorted.sort_values(by = ['cbkid'], inplace = True)
    df_sorted.reset_index(inplace=True, drop=True)
    df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames) #New
    
    #Extract list of proteins. Do not include compound of interest (since we assume its targets are unknown). 
    cluster_cbkid_sc3s = list(set(df_sorted.loc[cluster_highlight,:]["cbkid"]))
    cluster_cbkid_sc3s = [x for x in cluster_cbkid_sc3s if x not in compound]
    
    targets_cluster = list(set(df_sorted[df_sorted["cbkid"].isin(cluster_cbkid_sc3s)]["target"]))
    
    targets_cluster_unique = list()
    for targets in targets_cluster:
        if type(targets) is str:
            targets = targets.split('|')
            for target in targets:
                if target not in targets_cluster_unique:
                    targets_cluster_unique.append(target)
    
    return(targets_cluster_unique)

def countClusterProteins(df, annotations, cluster_highlight, 
                        compound, CompoundNames):
    """Translates cluster indexes to protein names. Counts proteins.

    This script is used to translate the output from a consensus 
    clustering to protein names (gene names), which is needed prior to
    plotting a PPI network. 

    The compound of interest is automatically removed from the cluster
    if present in the cell painting data.

    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing
       with process_cp_to_median().

    annotations: pandas.DataFrame
       Annotation data for compounds. 

    cluster_highlight: list
       List of indexes for compounds in CP data which belong to
       cluster.

    compound: list
       List of compounds to be removed from cluster. 
    
    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of compounds
       of interest as values. Should only contain compounds which 
       should be displayed in UMAP legend.
       
    """
    df = df.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')

    #Modify df index to be sorted by cbkid
    df_sorted = df.copy()

    #df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames)
    df_sorted.sort_values(by = ['cbkid'], inplace = True)
    df_sorted.reset_index(inplace=True, drop=True)
    df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames) #New
    
    #Extract list of proteins. Do not include compound of interest (since we assume its targets are unknown). 
    cluster_cbkid_sc3s = list(set(df_sorted.loc[cluster_highlight,:]["cbkid"]))
    cluster_cbkid_sc3s = [x for x in cluster_cbkid_sc3s if x not in compound]
    
    targets_cluster = list(df_sorted[df_sorted["cbkid"].isin(cluster_cbkid_sc3s)]["target"])
    
    targets_cluster_counted = pd.DataFrame(columns=["Target", "Count"])
    for targets in targets_cluster:
        if type(targets) is str:
            targets = targets.split('|')
            for target in targets:
                #Edit from below here
                if target in list(targets_cluster_counted['Target']):
                    targets_cluster_counted.loc[targets_cluster_counted['Target']==target,'Count'] += 1 
                else:
                    temp = pd.DataFrame.from_dict([{'Target': target, 'Count': 1}])
                    targets_cluster_counted = pd.concat([targets_cluster_counted, temp])
    
    return(targets_cluster_counted)

def plotClusterMOA(df, annotations, cluster_highlight, 
                        compound, CompoundNames, file_name=None):
    """Translates cluster indexes to mode of action (MoA) and plots.

    This script is used to translate the output from a consensus 
    clustering to mode/mechanism of action (MoA), which is then
    plotted as a histogram. 

    The compound of interest is automatically removed from the cluster
    if present in the cell painting data.

    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing
       with process_cp_to_median().

    annotations: pandas.DataFrame
       Annotation data for compounds. 

    cluster_highlight: list
       List of indexes for compounds in CP data which belong to
       cluster.

    compound: list
       List of compounds to be removed from cluster. 
    
    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of compounds
       of interest as values. Should only contain compounds which 
       should be displayed in UMAP legend.

    file_name: str or None
       If not None, saves the heatmap to file_name.
       
    """
    df = df.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')

    #Modify df index to be sorted by cbkid
    df_sorted = df.copy()

    #df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames)
    df_sorted.sort_values(by = ['cbkid'], inplace = True)
    df_sorted.reset_index(inplace=True, drop=True)
    df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames) #New
    
    #Extract list of proteins. Do not include compound of interest (since we assume its targets are unknown). 
    cluster_cbkid_sc3s = list(set(df_sorted.loc[cluster_highlight,:]["cbkid"]))
    cluster_cbkid_sc3s = [x for x in cluster_cbkid_sc3s if x not in compound]
    
    moa_cluster = list(df_sorted[df_sorted["cbkid"].isin(cluster_cbkid_sc3s)]["moa"])
    
    moa_cluster_counted = pd.DataFrame(columns=["MoA", "Count"])
    for moas in moa_cluster:
        if type(moas) is str:
            moas = moas.split('|')
            for moa in moas:
                #Edit from below here
                if moa in list(moa_cluster_counted['MoA']):
                    moa_cluster_counted.loc[moa_cluster_counted['MoA']==moa,'Count'] += 1 
                else:
                    temp = pd.DataFrame.from_dict([{'MoA': moa, 'Count': 1}])
                    moa_cluster_counted = pd.concat([moa_cluster_counted, temp])
    
    moa_cluster_counted = moa_cluster_counted.sort_values(by=['Count'], ascending=True)
    
    #Plot moa
    fig, ax = plt.subplots(figsize=(5,round((len(moa_cluster_counted)+1)/2, 0)))
    ax.barh(moa_cluster_counted['MoA'], moa_cluster_counted['Count'], color='grey')
    #addlabels(list(dfPlot['label']), list(dfPlot['fdr']), list(dfPlot['number_of_genes']))
    ax.set_title(' and '.join(compound), loc='left', fontweight='bold')
    ax.set_xlabel("Count in cluster")
    fig.show()

    moa_cluster_counted = moa_cluster_counted.sort_values(by=['Count'], ascending=False)
    
    #Save plot to file if file_name is defined. 
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(file_name)

    return(cluster_cbkid_sc3s)

def plotGOenrichmentCommunities(enrichment, file_name = None):
    """Plot barcharts for GO enrichment based on communitites.

    Takes output from function plotPPINetworkCommunities() as input and plots
    a horizontal barchart for each identified community. Colors are kept the same
    as in network graph. 

    Parameters
    ----------
    enrichment: pandas.DataFrame
        DataFrame with GO molecular function data and 
        other network enrichment data retrieved from string-db.
        Columns:
            category: str
            term: str
            number_of_genes: int
            number_of_genes_in_background: int
            ncbiTaxonId: int
            inputGenes: str, comma separated
            preferredNames: str, comma separated
            p_value: float
            fdr: float
            description: str

    file_name: str or None
       If not None, saves the heatmap to file_name.

    """

    ### CODE ###
    def addlabels(x,y,z):
        for i in range(len(x)):
            plt.text(y[i]-0.1, i, z[i], ha = 'center', va='center', color='white', fontweight='bold')
    
    colors = tab10(range(len(enrichment)))
    c = 0
    for com_sub in enrichment.keys():
        dfPlot = enrichment[com_sub].loc[enrichment[com_sub]['category'].isin(['Function','Process','RCTM']),:].sort_values('fdr')
        dfPlot['fdr'] = np.log10(dfPlot['fdr'].astype(float))*-1
        dfPlot = dfPlot.sort_values('fdr', ascending=True)

        #Create renames categories
        newCat = {'Function': 'GO MF', 'Process': 'GO BP', 'RCTM': 'Reactome'}
        dfPlot['category'] = dfPlot['category'].replace(newCat)
        dfPlot['label'] = dfPlot['description'] + ' (' + dfPlot['category'] + ': '+ dfPlot['term'] +')' 
        
        if len(dfPlot) > 0:    
            my_range=range(1,len(dfPlot.index)+1)
            
            fig, ax = plt.subplots(figsize=(5,round((len(dfPlot)+1)/2, 0)))
            ax.barh(dfPlot['label'], dfPlot['fdr'], color=colors[c])
            addlabels(list(dfPlot['label']), list(dfPlot['fdr']), list(dfPlot['number_of_genes']))
            ax.set_title(com_sub, loc='left', fontweight='bold')
            ax.set_xlabel("-log10(FDR)")
            fig.show()

            #Save plot to file if file_name is defined. 
            if not file_name is None:
                file_name2 = file_name.split('.')
                if not len(file_name2) == 2:
                    raise ValueError("file_name should not contain more than one dot ('.')")
                file_name2 = file_name2[0] + '_cluster' + str(c) + '.' + file_name2[1]
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(file_name2)
    
        c += 1


def findAdditionalHubProteins(protein_list_cluster,
                              protein_list_tpp,
                              network_type="physical"):
    """For final protein list, identify additional hub proteins.

    Downloads the whole set of interaction partners for proteins in
    protein_list_cluster from STRING DB, then counts the number of
    nodes for each protein connecting it to the orignial list of
    proteins in query. HUb nodes are sorted and returned as a list.

    Parameters
    ----------
    protein_list_cluster: list
        List of proteins from final joint TPP-CP network

    protein_list_tpp: list
        List of proteins from TPP

    network_type: str, default="physical"
        Defined type of network to query from string-db.
        Allowed values: "functional", "physical".

    Returns
    -------
    df_hubs: pandas.DataFrame
        DataFrame with final protein nodes and their ranks.
    """

    ### ERRORS ###
    valid_types = {"functional", "physical"}
    if network_type not in valid_types:
        raise ValueError("network_type must be one of %r." % valid_types)
        
    ### CODE ###
    #Importing network
    proteins = '%0d'.join(protein_list_cluster)
    
    #Setting parameters for STRING API
    string_api_url = "https://version-11-9.string-db.org/api"
    output_format = "tsv"
    method = "interaction_partners"

    if network_type == 'physical':
        required_score = 150
    else:
        required_score = 400
    
    request_url = "/".join([string_api_url, output_format, method])
    params = {
    
        "identifiers" : proteins, # your protein list
        "species" : 9606, # species NCBI identifier 
        #"limit" : 1, # only one (best) identifier per input protein
        "network_type": network_type,
        #"required_score": required_score
        "required_score": 400
    }

    #Request network from STRING-DB
    r = requests.post(request_url, data=params)

    #Clean up retrieved data
    lines = r.text.split('\n') # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines] # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df = pd.DataFrame(data[1:-1], columns = data[0]) 

    #Count prevalence of all interactors
    interactors_unique = list(set(list(df['preferredName_B'])))
    df_hubs = dict.fromkeys(interactors_unique, 0)
    df_hubs = pd.DataFrame.from_dict(df_hubs, orient='index', columns=['count'])
    df_hubs.reset_index(inplace=True, names='interactor')
    for i in interactors_unique:
        df_i = df.loc[df['preferredName_B'] == i,:] #New
        df_i = df_i.loc[df_i['preferredName_A'].isin(protein_list_tpp),:] #New
        df_hubs.loc[df_hubs['interactor'] == i,'count'] = len(
            #list(df.loc[df['preferredName_B'] == i,'preferredName_B'])
            list(df_i.loc[:,'preferredName_B']) #New
        )
    df_hubs = df_hubs.sort_values(by='count', ascending=False)
    
    
    return(df_hubs)

#Functions for comparing cell lines between TPP and CP

def compareTargetRNAexpression_cellLines(expression_df, 
                                         targets, 
                                         cell_cp, 
                                         cell_tpp,
                                         file_name = None,
                                         figsize = (6,6)
                                        ):
    """Compares RNA expression between cell lines for specified targets.

    RNA expression is extracted from the human protein atlas. 

    Parameters
    ----------
    expression_df: pandas.DataFrame
        cell line RNA expression data from Human Protein Atlas.

    targets: list
        proteins (gene names) which should be compared between cell lines.

    cell_cp: str
        name of cell line in cell painting (CP) data.

    cell_tpp: str
        name of cell line in thermal protein profiling (TPP) data.
    
    Returns
    -------
    figure (barchart). 
    
    """

    ##ERRORS##    

    ##CODE##
    #Subset data with only targets and cell lines of interest
    cells = [cell_cp, cell_tpp]

    df_cells = expression_df[expression_df['Cell line'].isin(cells)]
    df_cells = df_cells[df_cells['Gene name'].isin(targets)]

    #Plot barcharts
    plt.figure(figsize=figsize)
    sns.set_theme(style="white", palette="Set2")
    sns.barplot(x='Gene name', y='pTPM', hue='Cell line', data=df_cells, order=targets)

    #Save plot to file if file_name is defined. 
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(file_name)