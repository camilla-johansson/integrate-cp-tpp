"""Model for Cell Painting - Thermal Proteome Profiling project (cp-tpp)

Version: 1
Author: Camilla Johansson
"""
import os
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set_style("white")
import umap.umap_ as umap
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
import matplotlib.offsetbox as osb
import sc3s
from sklearn.cluster import AgglomerativeClustering
import scanpy as sc
from math import pi
from matplotlib.patches import Patch
import requests
import networkx as nx
from sklearn.manifold import TSNE

##### functions for processing cell painting data #####
def process_cp_to_median(cp_df, file_name=None):
    """Function for calculating per well median z-scores for cell painting data.

    Parameters
    ----------
    cp_df: pandas DataFrame 
       Contains the full cell painting data. If data is located on several
       plates, all plates should be merged before using the function.

    file_name: str
       If not None, saves the processed cell painting data to a .parquet file 
       with path and file name as in file_name. 

    Output
    ------
    dfGroupedByPlate1: pandas DataFrame
       TODO: Add more description
    """
    
    #### Code starts here ####

    #Remove images with few cells or extreme values
    #TODO: Change so that these are optional input parameters. 
    minCells = 20
    clipValue = 10
    
    dfMinNrCells  = cp_df[cp_df['Count_nuclei'] >= minCells].copy() # remove images with few cells
    
    # clip extreme values
    numCols = list(dfMinNrCells.select_dtypes(np.number).columns)
    numCols.remove('Count_nuclei')
    dfNumeric = dfMinNrCells[numCols].copy()
    dfNumeric.clip(lower=-clipValue, upper=clipValue, inplace=True)
    
    #Summarize data
    dfNotNumeric = dfMinNrCells[dfMinNrCells.columns[~dfMinNrCells.columns.isin(numCols)]].copy()
    dfProcessed = pd.concat([dfNotNumeric, dfNumeric], axis = 1)
    
    dfProcessed['CompOnPlate']= dfProcessed['batch_id'] + '_' + dfProcessed['cbkid'] + '_' + dfProcessed['Metadata_Barcode'] +  '_' + dfProcessed['project']
    
    dfGroupedByPlate1 = dfProcessed.groupby(['CompOnPlate']).median(numeric_only=True)
    dfGroupedByPlate1['batch_id'] = dfGroupedByPlate1.index.str.split('_').str[0]
    dfGroupedByPlate1['cbkid'] = dfGroupedByPlate1.index.str.split('_').str[1]
    dfGroupedByPlate1['Metadata_Barcode'] = dfGroupedByPlate1.index.str.split('_').str[2]
    dfGroupedByPlate1['project'] = dfGroupedByPlate1.index.str.split('_').str[3]
    dfGroupedByPlate1.sort_values(by = ['cbkid'], inplace = True)
    dfGroupedByPlate1.reset_index(inplace=True, drop=True)

    #If file_name is not None, saves the dfGroupedByPlate1 data as a .parquet file. 
    if not file_name is None: 
        dfGroupedByPlate1.to_parquet(file_name)
    
    return(dfGroupedByPlate1)

def create_subtitles(legend):
    """Used to create sublabels in legends when either alpha==0 or sizes==0.

    Parameters
    ----------
    legend: plt or sns legend

    Output
    ---------
    legend: reformatted plt or sns legend
    """
    
    vpackers = legend.findobj(osb.VPacker)
    for vpack in vpackers[:-1]: #Last vpack will be the title box
        vpack.align = 'left'
        for hpack in vpack.get_children():
            draw_area, text_area = hpack.get_children()
            for collection in draw_area.get_children():
                alpha = collection.get_alpha()
                sizes = collection.get_sizes()
                if alpha == 0 or all(sizes == 0):
                    draw_area.set_visible(False)
    return legend



### Function for plotting UMAP
def umap_compounds(df, annotations, CompoundNames, ControlNames=None, Cluster=None, random_state=111, n_neighbors=30, file_name=None):
    """Plots UMAP for CP data, coloured by CompoundNames.

    Plots a UMAP which is based on z-scores (no prior PCA).
    If ControlNames is not None, controls will be shown as 
    colored small circles in the plot. 
    If Cluster is not None, clusters around the compounds 
    will be plotted in blue. 
    
    Parameters
    ----------

    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing with 
       process_cp_to_median().

    annotations: pandas.DataFrame
       Annotation data for compounds. 

    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of 
       compounds of interest as values. Should only contain compounds
       which should be displayed in UMAP legend.

    ControlNames: dict or None
       Dictionary with cbkid as keys and displayed names of 
       control compounds as values. Should only contain control compounds
       which should be displayed in UMAP legend.
       If None, controls will not be shown.

    Cluster: dict or None
       Dictionary with displayed compound names as keys and row index
       of clustered compounds as values. 
       If None, cluster will not be shown. 

    random_state: int
       Sets random_state for umap.UMAP. Default is 111. 

    n_neighbors: int
       Sets hyperparametre n_neighbors for umap.UMAP. Default is 30. 

    file_name: str
       If not None, saves the umap graph to a file with path and 
       file name as in file_name.

    Output
    -------
    """
    ### CODE ###
    #Prepare compound color palette 
    Compounds = list(CompoundNames.keys())
    
    Compounds2 = [CompoundNames.get(n, n) for n in Compounds]
    
    color_maps = sns.color_palette('hls', 14)
    Compounds_colors = dict(zip(Compounds2, color_maps))

    if not ControlNames is None:
        Controls = list(ControlNames.keys())
        
        Controls2 = [ControlNames.get(n, n) for n in Controls]
        Controls_colors = dict(zip(Controls2, color_maps))
    
    #Prepare data for UMAP graph
    # Setting random state
    random_state = random_state
    
    # UMAP configuration with a random state for reproducibility
    n_neighbors = n_neighbors
    reducer = umap.UMAP(n_neighbors=n_neighbors, init='random', random_state=random_state, n_components=2)
    
    # Sample the data with a random state for reproducibility
    X = df.iloc[:, 1:-12]
    X = X.assign(batch_id=df["batch_id"], cbkid=df["cbkid"])
    X.reset_index(inplace=True, drop=True)
    
    # Perform UMAP embedding
    embedding = reducer.fit_transform(X.iloc[:, :-2])
    
    # Creating a DataFrame for UMAP results
    umap_result = pd.DataFrame(data=embedding, columns=['umap1', 'umap2'])
    umap_resultDf = pd.concat([umap_result, X.iloc[:, -2:]], axis=1)
    umap_resultDf.sort_values(by=['batch_id'], inplace=True)
    umap_resultDf = umap_resultDf.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')
    umap_resultDf.reset_index(inplace=True, drop=True)

    #Plot UMAP
    #Rename cbkid to full names
    #umap_resultDf['cbkid'] = umap_resultDf['cbkid'].replace(CompoundNames)
    umap_resultDf.sort_values(by = ['cbkid'], inplace = True)
    umap_resultDf.reset_index(inplace=True, drop=True)
    umap_resultDf['cbkid'] = umap_resultDf['cbkid'].replace(CompoundNames) #New
    if not ControlNames is None:
        umap_resultDf['cbkid'] = umap_resultDf['cbkid'].replace(ControlNames)
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    
    # Plot all other compounds in gray
    other_compounds_df = umap_resultDf
    sns.scatterplot(
        x='umap1',
        y='umap2',
        data=other_compounds_df,
        color='gray',
        s=15,
        alpha=0.75,
        ax=ax
    )


    if not Cluster is None:
        #Dummy plot for subtitle "Cluster"
        compounds_df = umap_resultDf[umap_resultDf['cbkid'].isin(Compounds2)]
        sns.scatterplot(
            x='umap1',
            y='umap2',
            data=compounds_df,
            s=0,
            ax=ax, 
            label="Cluster"
        )

        #Plot each Cluster in blue
        for i in Cluster.keys():
            cluster_umap_df = umap_resultDf.loc[Cluster[i],:]
            sns.scatterplot(
                x='umap1',
                y='umap2',
                data=cluster_umap_df,
                color='blue',
                s=45,
                alpha=0.75,
                ax=ax,
                label='For ' + i
            )

        #Dummy plot for blank row after sublegend "Cluster"
        compounds_df = umap_resultDf[umap_resultDf['cbkid'].isin(Compounds2)]
        sns.scatterplot(
            x='umap1',
            y='umap2',
            data=compounds_df,
            s=0,
            ax=ax, 
            label=" "
        )
            
    #Dummy plot for subtitle "Compounds"
    compounds_df = umap_resultDf[umap_resultDf['cbkid'].isin(Compounds2)]
    sns.scatterplot(
        x='umap1',
        y='umap2',
        data=compounds_df,
        s=0,
        ax=ax, 
        label="Compounds"
    )
    
    # Plot compounds with specified colors
    for compound, color in Compounds_colors.items():
        compound_df = umap_resultDf[umap_resultDf['cbkid'] == compound]
        sns.scatterplot(
            x='umap1',
            y='umap2',
            data=compound_df,
            color=color,
            s=45,
            alpha=1,
            ax=ax,
            label=compound, 
            marker='s'
        )

    
    #Dummy plot for subtitle "Controls"
    compounds_df = umap_resultDf[umap_resultDf['cbkid'].isin(Compounds2)]
    sns.scatterplot(
        x='umap1',
        y='umap2',
        data=compounds_df,
        s=0,
        ax=ax, 
        label="\nControls"
    )

    if not ControlNames is None:
        # Plot controls with specified colors
        for control, color in Controls_colors.items():
            control_df = umap_resultDf[umap_resultDf['cbkid'] == control]
            sns.scatterplot(
                x='umap1',
                y='umap2',
                data=control_df,
                color=color,
                #color_palette("husl", 8),
                s=15,
                alpha=1,
                ax=ax,
                label=control
            )
    
    # Plot 'PHB000001' in black
    phb000001_df = umap_resultDf[umap_resultDf['batch_id'] == 'PHB000001']
    sns.scatterplot(
        x='umap1',
        y='umap2',
        data=phb000001_df,
        color='black',
        s=15,
        alpha=0.75,
        ax=ax,
        label='DMSO'
    )
       
    # Add the custom legend to the plot
    ax_legend = ax.legend(loc='upper right', fontsize=10)

    #Center legend labels w/ alpha==0
    create_subtitles(ax_legend)
    
    # Rest of your code for setting face color, adjusting subplots and saving figure
    ax.set_facecolor('w')
    
    # If file_name is defined, saves plot. 
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()

def tsne_compounds(df, annotations, CompoundNames, ControlNames=None, Cluster=None, random_state=111, file_name=None):
    """Plots t-SNE for CP data, coloured by CompoundNames.

    Plots a t-SNE which is based on prior PCA.
    If ControlNames is not None, controls will be shown as 
    colored small circles in the plot. 
    If Cluster is not None, clusters around the compounds 
    will be plotted in blue. 
    
    Parameters
    ----------

    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing with 
       process_cp_to_median().

    annotations: pandas.DataFrame
       Annotation data for compounds. 

    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of 
       compounds of interest as values. Should only contain compounds
       which should be displayed in t-SNE legend.

    ControlNames: dict or None
       Dictionary with cbkid as keys and displayed names of 
       control compounds as values. Should only contain control compounds
       which should be displayed in t-SNE legend.
       If None, controls will not be shown.

    Cluster: dict or None
       Dictionary with displayed compound names as keys and row index
       of clustered compounds as values. 
       If None, cluster will not be shown. 

    random_state: int
       Sets random_state for umap.UMAP. Default is 111.  

    file_name: str
       If not None, saves the umap graph to a file with path and 
       file name as in file_name.

    Output
    -------
    """
    ### CODE ###
    #Prepare compound color palette 
    Compounds = list(CompoundNames.keys())
    
    Compounds2 = [CompoundNames.get(n, n) for n in Compounds]
    
    color_maps = sns.color_palette('hls', 14)
    Compounds_colors = dict(zip(Compounds2, color_maps))

    if not ControlNames is None:
        Controls = list(ControlNames.keys())
        
        Controls2 = [ControlNames.get(n, n) for n in Controls]
        Controls_colors = dict(zip(Controls2, color_maps))
    
    #Generating PCA
    x = df.iloc[:, 1:-12].values
    pca = PCA(n_components=50)
    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents)
    principalDf.rename(columns={0: 'principal component 1', 1: 'principal component 2'}, 
                       inplace=True)
    principalDf = principalDf.merge(df[['batch_id', 
                                        'Metadata_Barcode', 
                                        'cbkid', 'Count_nuclei', 
                                        'project']], 
                                    left_index = True, 
                                    right_index = True)
    principalDf.sort_values(by = ['batch_id'], inplace = True)

    #Generate t-SNE
    perpl =30
    
    X= principalDf.sample(frac = 1)
    X.reset_index(inplace=True, drop=True)
    
    X_embedded = TSNE(n_components=2, 
                      perplexity=perpl, 
                      init='pca',
                      learning_rate='auto').fit_transform(X.iloc[:,:50].values)
    tsne_result = pd.DataFrame(data = X_embedded, columns=["tsne1", "tsne2"])
    tsne_resultDf = pd.concat([tsne_result, X.iloc[:,-5:]], axis = 1)
    tsne_resultDf.sort_values(by = [ 'batch_id'], inplace = True)
    tsne_resultDf = tsne_resultDf.merge(annotations, 
                                        left_on='batch_id', 
                                        right_on = 'Batch nr', 
                                        how='left'
                                       )
    tsne_resultDf.reset_index(inplace=True, drop=True)

    #Plot t-SNE
    #Rename cbkid to full names
    tsne_resultDf.sort_values(by = ['cbkid'], inplace = True)
    tsne_resultDf.reset_index(inplace=True, drop=True)
    tsne_resultDf['cbkid'] = tsne_resultDf['cbkid'].replace(CompoundNames) #New
    if not ControlNames is None:
        tsne_resultDf['cbkid'] = tsne_resultDf['cbkid'].replace(ControlNames)
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    
    # Plot all other compounds in gray
    other_compounds_df = tsne_resultDf
    sns.scatterplot(
        x='tsne1',
        y='tsne2',
        data=other_compounds_df,
        color='gray',
        s=15,
        alpha=0.75,
        ax=ax
    )


    if not Cluster is None:
        #Dummy plot for subtitle "Cluster"
        compounds_df = tsne_resultDf[tsne_resultDf['cbkid'].isin(Compounds2)]
        sns.scatterplot(
            x='tsne1',
            y='tsne2',
            data=compounds_df,
            s=0,
            ax=ax, 
            label="Cluster"
        )

        #Plot each Cluster in blue
        for i in Cluster.keys():
            cluster_tsne_df = tsne_resultDf.loc[Cluster[i],:]
            sns.scatterplot(
                x='tsne1',
                y='tsne2',
                data=cluster_tsne_df,
                color='blue',
                s=45,
                alpha=0.75,
                ax=ax,
                label='For ' + i
            )

        #Dummy plot for blank row after sublegend "Cluster"
        compounds_df = tsne_resultDf[tsne_resultDf['cbkid'].isin(Compounds2)]
        sns.scatterplot(
            x='tsne1',
            y='tsne2',
            data=compounds_df,
            s=0,
            ax=ax, 
            label=" "
        )
            
    #Dummy plot for subtitle "Compounds"
    compounds_df = tsne_resultDf[tsne_resultDf['cbkid'].isin(Compounds2)]
    sns.scatterplot(
        x='tsne1',
        y='tsne2',
        data=compounds_df,
        s=0,
        ax=ax, 
        label="Compounds"
    )
    
    # Plot compounds with specified colors
    for compound, color in Compounds_colors.items():
        compound_df = tsne_resultDf[tsne_resultDf['cbkid'] == compound]
        sns.scatterplot(
            x='tsne1',
            y='tsne2',
            data=compound_df,
            color=color,
            s=45,
            alpha=1,
            ax=ax,
            label=compound, 
            marker='s'
        )

    
    #Dummy plot for subtitle "Controls"
    compounds_df = tsne_resultDf[tsne_resultDf['cbkid'].isin(Compounds2)]
    sns.scatterplot(
        x='tsne1',
        y='tsne2',
        data=compounds_df,
        s=0,
        ax=ax, 
        label="\nControls"
    )

    if not ControlNames is None:
        # Plot controls with specified colors
        for control, color in Controls_colors.items():
            control_df = tsne_resultDf[tsne_resultDf['cbkid'] == control]
            sns.scatterplot(
                x='tsne1',
                y='tsne2',
                data=control_df,
                color=color,
                #color_palette("husl", 8),
                s=15,
                alpha=1,
                ax=ax,
                label=control
            )
    
    # Plot 'PHB000001' in black
    phb000001_df = tsne_resultDf[tsne_resultDf['batch_id'] == 'PHB000001']
    sns.scatterplot(
        x='tsne1',
        y='tsne2',
        data=phb000001_df,
        color='black',
        s=15,
        alpha=0.75,
        ax=ax,
        label='DMSO'
    )
       
    # Add the custom legend to the plot
    ax_legend = ax.legend(loc='upper right', fontsize=10)

    #Center legend labels w/ alpha==0
    create_subtitles(ax_legend)
    
    # Rest of your code for setting face color, adjusting subplots and saving figure
    ax.set_facecolor('w')
    ax.set_xlim(-70, 120)
    ax.set_ylim(-80, 100)
    
    # If file_name is defined, saves plot. 
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()

def cluster_compound_sc3s(df,
                          CompoundNames, 
                          n_comps=100, 
                          n_clusters=[100, 110, 120, 150, 180], 
                          d_range=[13], 
                          n_runs = 200, 
                          batch_size = 2000,
                          random_state = 111,
                          file_name=None):
    """Cluster cell painting data using SC3s.

    This function uses SC3s, Single Cell Consensus Clustering with speed,
    combined with some downstream filtering steps to create tight clusters 
    of Cell Painting data around the compounds of interest. 

    SC3s was originally published in: 
    Quah, F.X., Hemberg, M. SC3s: efficient scaling of single cell consensus
    clustering to millions of cells. BMC Bioinformatics 23, 536 (2022).
    https://doi.org/10.1186/s12859-022-05085-z

    Code and documentation for SC3s is available from:
    https://github.com/hemberg-lab/sc3s/tree/master
    
    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing with 
       process_cp_to_median().

    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of 
       compounds of interest as values. Should only contain compounds
       which should be clustered.

    n_comps: int
       Number of components for PCA dimensionality reduction.

    n_clusters: list of ints
       Number of clusters for KNN in sc3s. Sc3s can loop through
       several cluster sizes. 

    d_range: list of ints
       number of PCs to include in sc3s analysis.

    batch_size: int
       Number of iterations of KNN for each size of n_clusters.

    file_name: str
       If not None, saves the cell painting PCA data with 
       cluster information to a .parquet file

    Output
    ------
    """

    ### CODE ###
    # Convert original data to an AnnData format, which is the required format for sc3s
    adata = sc.AnnData(df.iloc[:, 1:-12].values, 
        df.iloc[:, 1:-12].index.to_frame(), 
        df.iloc[:, 1:-12].columns.to_frame())
    
    # dimensionality reduction with PCA
    sc.tl.pca(adata, svd_solver='arpack', zero_center=False, n_comps=n_comps)

    #Cluster SC3s with 13 PCs (80% cumulative expected variance)
    print("Clustering with SC3s...")
    sc3s.tl.consensus(adata, 
                      n_clusters = n_clusters, 
                      d_range = d_range,
                      n_runs = n_runs, 
                      batch_size = batch_size,
                      random_state = random_state
                     )
    
    #Returning clustered data to original pandas dataframe
    principalDf_conc = pd.DataFrame(adata.obsm['X_pca'])
    principalDf_conc = principalDf_conc.merge(pd.DataFrame(adata.obs), left_index=True, right_on=0)
    principalDf_conc.index = principalDf_conc.index.astype(int)
    principalDf_conc = principalDf_conc.iloc[:,1:]
    principalDf_conc.rename(columns={'0_x': 'principal component 1', 1: 'principal component 2'}, inplace=True)
    principalDf_conc = principalDf_conc.merge(df[['batch_id', 'Metadata_Barcode', 
                                                                 'cbkid', 'Count_nuclei', 'project']], 
                                              left_index = True, right_index = True)
    
    #Rename cbkid to full names
    #principalDf_conc['cbkid'] = principalDf_conc['cbkid'].replace(CompoundNames)
    principalDf_conc.sort_values(by = ['cbkid'], inplace = True)
    principalDf_conc.reset_index(inplace=True, drop=True)
    principalDf_conc['cbkid'] = principalDf_conc['cbkid'].replace(CompoundNames) #New

    #Find cluster for each compound
    cluster_highlight_i = {i:[] for i in CompoundNames.values()}
    for i in CompoundNames.values():
        cluster_highlight_i[i] = find_cluster(principalDf_conc, n_clusters, i)

    #If file_name is not None, saves the principalDf_conc data as a .parquet file. 
    if not file_name is None: 
        principalDf_conc.to_parquet(file_name)

    return(cluster_highlight_i)

def cluster_compound_agglomerative(df, CompoundNames, n_clusters=300, distance_threshold=None, file_name=None): 
    """Cluster cell painting data using agglomerative clustering.

    This function uses agglomerative hierarchical clustering from 
    skikit-learn combined with some downstream filtering steps to 
    create tight clusters of Cell Painting data around the compounds 
    of interest. 

    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame with processed cell painting data, after processing with 
       process_cp_to_median().

    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of 
       compounds of interest as values. Should only contain compounds
       which should be clustered.

    file_name: str
       If not None, saves the cell painting PCA data with 
       cluster information to a .parquet file

    Returns
    ------
    """

    ### CODE ###
    #Create PCA data set
    #Generating PCA
    print('Generating PCA...')
    x = df.iloc[:, 1:-12].values
    pca = PCA(n_components=100)
    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents)
    principalDf.rename(columns={0: 'principal component 1', 1: 'principal component 2'}, inplace=True)
    principalDf.columns = principalDf.columns.astype(str)
    principalDf = principalDf.merge(df[['batch_id', 
                                        'Metadata_Barcode', 
                                        'cbkid', 'Count_nuclei', 
                                        'project']], 
                                    left_index = True, right_index = True)
    principalDf.sort_values(by = ['cbkid'], inplace = True)
    principalDf.reset_index(inplace=True, drop=True)

    #Perform agglomerative clustering
    print('Perform agglomerative clustering...')
    aggl_clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                              #compute_full_tree=True,
                                              distance_threshold=distance_threshold
                                             ).fit(principalDf.iloc[:,:50])

    #Add cluster information to PCA df
    principalDf2 = principalDf.assign(cluster = list(aggl_clustering.labels_))
    
    #Check if compounds are clustered. 
    cluster_highlight_i = {compound:[] for compound in CompoundNames.values()}
    for compound in CompoundNames.keys():
        cluster_no_pca_aggl = list(
            principalDf2[principalDf2["cbkid"] == compound]["cluster"]
        )[0]
    
        if cluster_no_pca_aggl < 0:
            print(str(CompoundNames[compound])+" is not clustered")
        # Names for cluster to colour
        cluster_highlight_i[CompoundNames[compound]] = list(
            set(principalDf2[principalDf2["cluster"] == cluster_no_pca_aggl].index)
        )
        print("Number of compounds in "+str(CompoundNames[compound]) +
              " cluster: " + str(len(cluster_highlight_i[CompoundNames[compound]])))
    return(cluster_highlight_i)

def rename_keys(dict_, new_keys):
    """
     new_keys: type List(), must match length of dict_
    """

    # dict_ = {oldK: value}
    # d1={oldK:newK,} maps old keys to the new ones:  
    d1 = dict( zip( list(dict_.keys()), new_keys) )

          # d1{oldK} == new_key 
    return {d1[oldK]: value for oldK, value in dict_.items()}

def find_cluster(principalDf_conc, n_clusters, compound, cutoff=20, rep=0):
    """Find smallest cluster for each compound.

    Function is used inside cluster_compound_sc3s() to find 
    the smallest cluster among all the consensus clusters for
    n_clusters which contains the compound of interest. 

    The selected cluster needs to contain at least 20 compounds 
    by default, which can be changed using the "cutoff" parameter. 

    Parameters
    -------
    principalDf_conc: pandas.DataFrame
       DataFrame containing the calculated PCs for the cell painting
       data together with the consensus clusters for each value of
       n_clusters. 

    n_clusters: list of ints
       Number of clusters for KNN in sc3s. Sc3s can loop through
       several cluster sizes.

    compound: str
       Name of compound to find smallest cluster for. 

    cutoff: int
       Minimum size of selected "smallest" cluster for compound.
       Default is 20. TODO.

    rep: int
        Which replicate of compound in CP data that cluster should
        include. Default is 0. Change only if there is a large
        difference between replicate morphology in CP. 

    Output
    ------
    cluster_highlight_compound: list
       List of row indices for the selected cluster.

    """

    ### CODE ###
    #Find cluster for compound
    cluster_names = n_clusters
    cluster_names = ['sc3s_'+ str(element) for element in cluster_names]
    
    cluster_no_temp = list(principalDf_conc[principalDf_conc["cbkid"] == compound][cluster_names[0]])[rep]
    cluster_highlight = list(set(principalDf_conc[principalDf_conc[cluster_names[0]] == cluster_no_temp].index))
    min_cluster = len(cluster_highlight)
    cluster_name_smallest = cluster_names[1]
    cluster_no = list(principalDf_conc[principalDf_conc["cbkid"] == compound][cluster_name_smallest])[rep]
    
    print("For compound "+ compound)
    for cluster_name in cluster_names:
        #Find cluster for JQ1
        cluster_no_temp = list(principalDf_conc[principalDf_conc["cbkid"] == compound][cluster_name])[rep]
        cluster_highlight = list(set(principalDf_conc[principalDf_conc[cluster_name] == cluster_no_temp].index))
        print("Cluster " + cluster_name + " is " + str(len(cluster_highlight)) + " long.")
        if len(cluster_highlight) < min_cluster:
            min_cluster = len(cluster_highlight)
            cluster_no = list(principalDf_conc[principalDf_conc["cbkid"] == compound][cluster_name])[rep]
            cluster_name_smallest = cluster_name
             

    #TODO: Incorporate cutoff. 
    
    # Names for cluster to colour
    cluster_highlight_compound = list(set(principalDf_conc[principalDf_conc[cluster_name_smallest] == cluster_no].index))
    print("Smallest cluster " + cluster_name_smallest + " with length " + str(len(cluster_highlight_compound)))

    return(cluster_highlight_compound)


def processDataForRadar(df, CompoundNames, Cluster=None):
    controls = list(CompoundNames.values())

    df_sorted = df.copy()

    #df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames)
    df_sorted.sort_values(by = ['cbkid'], inplace = True)
    df_sorted.reset_index(inplace=True, drop=True)

    if not Cluster is None:
        #Find names of compounds in Cluster
        cluster_compounds = controls
        for i in Cluster.keys():
            cluster_compounds = cluster_compounds + list(df_sorted.loc[list(Cluster[i]),'cbkid'])
        controls = list(set(cluster_compounds)) #Get only unique values
    

    df_sorted['cbkid'] = df_sorted['cbkid'].replace(CompoundNames) #New
    df_sorted['CompOnPlate']= df_sorted['batch_id'] + '_' + df_sorted['cbkid'] 

    dfGroupedByCompound = df_sorted.groupby(['CompOnPlate']).median(numeric_only=True)
    dfGroupedByCompound['batch_id'] = dfGroupedByCompound.index.str.split('_').str[0]
    dfGroupedByCompound['cbkid'] = dfGroupedByCompound.index.str.split('_').str[1]
    dfGroupedByCompound.tail()
    
    dfRadar = dfGroupedByCompound[dfGroupedByCompound['cbkid'].isin(controls)]
    dfRadar.set_index('cbkid', inplace=True, drop=True)
    
    dfRadar = dfRadar.rename_axis("cbkid").reset_index()

    return(dfRadar)





def plotRadar(data, cbkid, file_name=None):
    """Plots Radar Charts for cell painting data

    TODO: Add more description. 
    TODO: Check if someone should be credited for this code.

    Paramters
    ---------
    data: pandas.DataFrame
       DataFrame processed using function processDataForRadar().

    cbkid: str
       CBKID of compound to plot

    file_name: str
       If not None, saves the radar chart to file_name.

    Output
    ------

    """
    # Define the main groups and their corresponding subgroups with required words
    main_groups = {
        "Nucleus": {
            "I": ["Intensity", "HOECHST", "_nuclei"],
            "G": ["Granularity", "HOECHST", "_nuclei"],
            "L": ["Location", "HOECHST", "_nuclei"],
            "RD": ["RadialDistribution", "HOECHST", "_nuclei"]
        },
        "ER": {
            "I": ["Intensity", "CONC", "_cells"],
            "G": ["Granularity", "CONC", "_cells"],
            "L": ["Location", "CONC", "_cells"],
            "RD": ["RadialDistribution", "CONC", "_cells"]
        },
        "Ncli.+c.RNA": {
            "I": ["Intensity", "SYTO", "_cells"],
            "G": ["Granularity", "SYTO", "_cells"],
            "L": ["Location", "SYTO", "_cells"],
            "RD": ["RadialDistribution", "SYTO", "_cells"]
        },
        "Golgi+cytk": {
            "I": ["Intensity", "PHAandWGA", "_cells"],
            "G": ["Granularity", "PHAandWGA", "_cells"],
            "L": ["Location", "PHAandWGA", "_cells"],
            "RD": ["RadialDistribution", "PHAandWGA", "_cells"]
        },
        "Mitochondria": {
            "I": ["Intensity", "MITO", "_cells"],
            "G": ["Granularity", "MITO", "_cells"],
            "L": ["Location", "MITO", "_cells"],
            "RD": ["RadialDistribution", "MITO", "_cells"]
        },
        "AreaShape": {
            "C": ["AreaShape", "_cells"],
            "N": ["AreaShape", "_nuclei"],
            "Cy": ["AreaShape", "_cytoplasm"]
        }
    }
    
    # Function to categorize features into main groups and subgroups
    def categorize_features(feature_names, main_groups):
        categorized_features = {main_group: {subgroup: [] for subgroup in subgroups} 
                                for main_group, subgroups in main_groups.items()}
    
        for feature in feature_names:
            for main_group, subgroups in main_groups.items():
                for subgroup, keywords in subgroups.items():
                    if all(keyword in feature for keyword in keywords) and keywords[0] == feature.split("_")[0]:
                        categorized_features[main_group][subgroup].append(feature)
                        break  # Found the right subgroup, no need to check others
    
        return categorized_features
    
    # Load the CSV file
    #df = pd.read_csv('ssmd_rh30.csv', delimiter=';')
    
    # Check if 'cbkid' column exists
    if 'cbkid' not in data.columns:
        raise ValueError("'cbkid' column not found in the DataFrame")
    
    # Specify the cbkid values you are interested in
    #cbkid = 'Roscovitine'
    
    # Assuming feature names start from the 3rd column
    feature_names = data.columns[2:]
    categorized_features = categorize_features(feature_names, main_groups)
    
    # Prepare data for radar chart
    all_stats = []
    all_labels = None
    
    
    #data = dfTPP[dfTPP['cbkid'] == cbkid]
    data = data[data['cbkid'] == cbkid]
    for d in ['pos','neg']:
        if data.empty:
            print(f"No data found for cbkid: {cbkid}")
            continue
    
        averages = {}
        for main_group, subgroups in categorized_features.items():
            averages[main_group] = {}
            for subgroup, features in subgroups.items():
                if not features:
                    avg_value = 0
                else:
                    #print(data[features].median(axis=1))
                    #print(abs(data[features].median(axis=1).median()))
                    if d == 'neg':
                        if data[features].median(axis=1).median() > 0:
                            avg_value = 0
                        else: 
                            avg_value = abs(data[features].median(axis=1).median())
                    else:
                        if data[features].median(axis=1).median() > 0:
                            avg_value = data[features].median(axis=1).median()
                        else: 
                            avg_value = 0
                            
                averages[main_group][subgroup] = avg_value
    
        labels = [f"{subgroup}" for main_group, subgroups in averages.items() for subgroup in subgroups]
        labels2 = [f"{main_group}" for main_group, subgroups in averages.items() for subgroup in subgroups]
        stats = [averages[main_group][subgroup] for main_group, subgroups in averages.items() for subgroup in subgroups]
    
        stats += stats[:1]
        all_stats.append(stats)
        all_labels = labels
    
    # Create radar chart
    angles = np.linspace(0, 2 * pi, len(all_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    # Define colors for each main group
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
    if len(main_groups) > len(colors):
        raise ValueError("Not enough colors defined for the number of main groups")
    
    # Calculate the total number of subgroups
    total_subgroups = sum(len(subgroups) for subgroups in main_groups.values())
    
    # Find the maximum value for the outer edge of the filled segments
    max_value = max(max(cbkid_stats) for cbkid_stats in all_stats)
    outer_edge_value = max_value * 1.03  # Adjust as necessary to cover the full segment
    inner_ring_value = max_value * 1.3      # Set to the max value of the stats
    
    # Calculate starting angles for each main group
    group_start_angles = {}
    current_angle = 0
    for group in main_groups.keys():
        group_start_angles[group] = current_angle
        current_angle += len(main_groups[group]) * (2 * pi / total_subgroups)
    
    # Draw the colored segments for the last part of the radar plot
    # Angle increment to extend the start and end of each colored segment
    angle_extension = 2 * pi / total_subgroups * -0.5  # Adjust this value as needed
    
    # Draw the colored segments with extended start and end angles
    for group, color in zip(main_groups.keys(), colors):
        # Original start and end angles for the segment
        start_angle = group_start_angles[group]
        end_angle = start_angle + len(main_groups[group]) * (2 * pi / total_subgroups)
    
        # Extend the start and end angles
        extended_start_angle = (start_angle + angle_extension) % (2 * pi)
        extended_end_angle = (end_angle + angle_extension) % (2 * pi)
    
        # Generate the extended segment angles
        if extended_start_angle < extended_end_angle:
            segment_angles = np.linspace(extended_start_angle, extended_end_angle, 100)
        else:
            segment_angles = np.concatenate((np.linspace(extended_start_angle, 2 * pi, 50), np.linspace(0, extended_end_angle, 50)))
    
        # Fill the segment with extended angles
        ax.fill_between(segment_angles, inner_ring_value, outer_edge_value, color=color, alpha=0.5, zorder=3)
    
    
    #plt.xticks(angles[:-1], all_labels, color='grey', size=8)
    # Place subgroup names around the radar plot
    for angle, label in zip(angles[:-1], all_labels):
        ax.text(angle, outer_edge_value * 1.12, label,
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, color='black', zorder=4)
        
    
    # Function to calculate text properties for angled placement
    def calculate_text_properties(angle):
        # Alignment is centered for all labels
        alignment = 'center'
        
        # Rotate text based on its position around the chart
        if 0 <= angle < pi:
            # Left half of the chart
            rotation = np.degrees(angle) - 90
        else:
            # Right half of the chart
            rotation = np.degrees(angle) - 90
    
        return alignment, rotation
    
    # Set the distance from the center for the main group names
    label_distance = outer_edge_value * 1.45  # Adjust as necessary
    
    def get_group_midpoint_angle(group_start_angle, num_subgroups, total_subgroups):
        group_angle_span = (num_subgroups * 2 * pi / total_subgroups)
        midpoint_angle = -0.13+group_start_angle + group_angle_span / 2
        return midpoint_angle
    
    # Place main group names angled towards the center of their segments
    for main_group, start_angle in group_start_angles.items():
        num_subgroups = len(main_groups[main_group])
        midpoint_angle = get_group_midpoint_angle(start_angle, num_subgroups, total_subgroups)
        alignment, rotation = calculate_text_properties(midpoint_angle)
    
        ax.text(midpoint_angle, label_distance, main_group,
                horizontalalignment=alignment, verticalalignment='center',
                rotation=rotation, fontsize=10, color='black', rotation_mode='anchor', zorder=5)
    
    plt.xticks(angles[:-1], [])  # Remove the original labels
    
    # After preparing all_stats
    all_values = [stat for stats in all_stats for stat in stats]
    min_value = min(all_values)
    max_value = max(all_values)
    
    # Now generate radial_tick_values and set them using set_rgrids
    radial_tick_values = np.linspace(start=min_value, stop=max_value, num=5, endpoint=False)
    ax.set_rgrids(radii=radial_tick_values, labels=[f'{r:.1f}' for r in radial_tick_values], angle=0, fontsize=8)
    
    # Draw the radar plots for each dataset
    for d, d_stats in zip(['pos','neg'], all_stats):
        ax.plot(angles, d_stats, label=d)
        ax.fill(angles, d_stats, alpha=0.25, zorder=1)
    
    # Remove the outer boundary line of the radar plot
    ax.spines['polar'].set_visible(False)
    
    # Set the minimum value from which the radial lines should start
    min_value = min_value
    
    # Disable the default polar grid
    ax.xaxis.grid(False)
    
    # Customize the appearance of the radial grid lines
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=1, zorder=1)
    
    
    # Manually draw each radial line from the min_value to the outer_edge_value
    num_vars = len(all_labels)  
    for i in range(num_vars):
        angle = angles[i]
        ax.plot([angle, angle], [min_value, outer_edge_value], color='gray', linestyle='-', linewidth=0.5, zorder=1)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))  # Add a legend

    ax.set_facecolor('w')
    
    # Save the plot
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(file_name, bbox_inches='tight')
        print('Radar Chart saved as:\n'+file_name)
    
    
    # Display the plot
    plt.show()

def plotFeatureHeatmap(df, CompoundNames, ControlNames, file_name=None, Cluster=None, height=3.5):
    """Function for plotting feature heatmaps.

    TODO: Add more description.

    Parameters
    ----------
    df: pandas.DataFrame
       DataFrame processed using function process_cp_to_median()

    CompoundNames: dict
       Dictionary with cbkid as keys and displayed names of 
       compounds of interest as values.

    ControlNames: dict
       Dictionary with cbkid as keys and displayed names of 
       control compounds as values.

    file_name: str
       If not None, saves the heatmap to file_name.

    Cluster: dict or None
       If not None, adds cluster compounds identified through
       cluster_compound_sc3s() to heatmap. TODO.
    
    """
    
    #if not Cluster is None:
        #Prepare Cluster names for inclusion in AllNames
        #df = df.
    
    #Prepare df for plot
    AllNames = CompoundNames | ControlNames
    dfHeat = processDataForRadar(df, AllNames, Cluster=Cluster)
    dfHeat.set_index('cbkid', inplace=True, drop=True)

    #Create labels for heaetmap
    d = {'names': dfHeat.iloc[:,1:-10].columns}
    feature_label = pd.DataFrame(d)
    feature_label['group'] = feature_label['names'].str.split('_').str[-1]
    feature_label.set_index('names', inplace=True, drop=True)

    #Create colors for labels
    lut = dict(zip(feature_label['group'].unique(), "rbg"))
    col_colors = feature_label['group'].map(lut)
    
    #Plot heatmap
    sns.set(font_scale=1)
    sns_cluster = sns.clustermap(dfHeat.iloc[:,1:-1],
                                 metric='euclidean', 
                                 z_score=None,
                                 standard_scale=None,  cbar_kws=None, figsize=(10, height), 
                                 row_cluster=True, col_cluster=False,
                                 cbar_pos= (0.01, 0.8, 0.01, 0.3),
                                 dendrogram_ratio=(.2, 0),
                                 vmin=-5, vmax=5,
                                 row_linkage=None, col_linkage=None, row_colors=None,
                                 col_colors=col_colors, cmap="RdBu_r", xticklabels=False, yticklabels= True)

    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='Group',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper left')

    #If file_name is not None, save heatmap as file_name
    if not file_name is None:
        plt.rcParams['svg.fonttype'] = 'none'
        sns_cluster.savefig(file_name,  dpi=300, bbox_inches='tight')

def findTargetsForCompound(df, annotations, compound, merge=True):
    """Create list of targets for compounds.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with processed cell painting data, after processing
        with process_cp_to_median().

    annotations: pandas.DataFrame
        Annotation data for compounds.

    compound: str
        cbkid for compound of interest. 
        
    Output
    ------
    target_list: list
        list of targets (gene names) for compound. 
    """

    ### CODE ###
    if merge == True:
        df_merged = df.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')
    else:
        df_merged = df.copy()
    
    targets_str_all = df_merged.loc[df_merged['cbkid'] == compound,'target']
    
    target_list = []
    for l in range(len(targets_str_all)): 
        target_list_l = str(targets_str_all.iloc[l]).split('|')
        target_list = target_list + target_list_l

    target_list = list(set(target_list))

    return(target_list)

def findClusterForCompoundTargets(df, annotations, target_list):
    """Find all indexes for compounds where target is in target_list.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with processed cell painting data, after processing
        with process_cp_to_median().

    annotations: pandas.DataFrame
        Annotation data for compounds.

    target_list: list
        list of targets (gene names) for compound.
        
    Output
    ------
    """

    ### CODE ###
    #Sort and index dataframe on cbkid
    df_merged = df.merge(annotations, left_on='batch_id', right_on='Batch nr', how='left')
    df_merged.sort_values(by = ['cbkid'], inplace = True)
    df_merged.reset_index(inplace=True, drop=True)

    #For each compound in CP data, check if targets are in target_list
    compound_indices_out = []
    for i in df_merged.index:
        compound_i = df_merged.loc[i,'cbkid']
        targets_i = findTargetsForCompound(df_merged, annotations, compound_i, merge=False)
        for target in targets_i:
            if target in target_list:
                compound_indices_out = compound_indices_out + [i]
            
    return(compound_indices_out)