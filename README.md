# Integrating Cell Painting and Thermal Proteome Profiling for Improved Inference of Mechanism of Action
 
This repository contains analysis scripts and code for the paper "Integrating Cell Painting and Thermal Proteome Profiling for Improved Inference of Mechanism of Action" (update with bioRxiv preprint doi)

The workflow for data integration proposed in the manuscript is described in the image below.

[insert image here]

## Summary

In this article, we propose a strategy for integrating data from cell painting (CP), a high-throughput cell imaging method for classifying drugs based on morphological perturbation, with thermal proteome profiling (TPP). Our proposed strategy involves constructing protein-protein interaction networks based on potential targets identified using both assays. We used five public TPP data set for the small molecule compounds (+)-JQ1, I-BET151, Vemurafenib, Crizotinib and Panobinostat, as well as CP screening data for 5300 compounds on U2OS cells. 

## Repository structure
- **`cp/`**
    - **`cp.py`**
    Script for analyzing cell painting data

- **`tpp/`**
    - **`tpp.py`**
    Script for analyzing thermal proteome profiling data

- **`Crizotinib.ipynb`**
Analysis workflow for the compound Crizotinib to generate figures and results in manuscript. 

- **`JQ1_IBET151.ipynb`**
Analysis workflow for the compounds (+)-JQ1 and I-BET151 to generate figures and results in manuscript.

- **`Panobinostat.ipynb`**
Analysis workflow for the compound Panobinostat to generate figures and results in manuscript.

- **`Vemurafenib.ipynb`**
Analysis workflow for the compound Vemurafenib to generate figures and results in manuscript.

- **`SC3s_10repeats.ipynb`**
Analysis workflow for assessing reproducibility of the clustering algorithm SC3s on CP data in manuscript.

- **`template_run_model.ipynb`**
A template for the analysis workflow described in manuscript. Includes descriptions of the different steps to allow anyone to adapt the workflow to their own TPP and CP data. 

## Using the analysis workflow template
The template `template_run_model.ipynb` can be used to test and adapt the workflow described in the manuscript to other TPP and CP data. The template uses functions from the scripts `cp/cp.py` and `tpp/tpp.py`. More detailed descriptions of the workflow and functions can be found in the template. 

## Data and code availability 
The SPECS repurposing cell painting data will be made available in an appropriate repository.

All thermal proteome profiling data used in these analysis scripts have been retrieved from the supplementary material of the following manuscripts: 

* **(+)-JQ1**: Savitski MM, Zinn N, Faelth-Savitski M, Poeckel D, Gade S, Becher I, et al. Multiplexed Proteome Dynamics Profiling Reveals Mechanisms Controlling Protein Homeostasis. Cell. 2018 Mar 22;173(1):260-274.e25. (doi: 10.1016/j.cell.2018.02.030)

* **I-BET151**: Savitski MM, Zinn N, Faelth-Savitski M, Poeckel D, Gade S, Becher I, et al. Multiplexed Proteome Dynamics Profiling Reveals Mechanisms Controlling Protein Homeostasis. Cell. 2018 Mar 22;173(1):260-274.e25. (doi: 10.1016/j.cell.2018.02.030)

* **Vemurafenib**: Savitski MM, Reinhard FBM, Franken H, Werner T, Savitski MF, Eberhard D, et al. Tracking cancer drugs in living cells by thermal profiling of the proteome. Science. 2014 Oct 3;346(6205):1255784. (doi: 10.1126/science.1255784)

* **Crizotinib**: Savitski MM, Reinhard FBM, Franken H, Werner T, Savitski MF, Eberhard D, et al. Tracking cancer drugs in living cells by thermal profiling of the proteome. Science. 2014 Oct 3;346(6205):1255784. (doi: 10.1126/science.1255784)

* **Panobinostat**: Franken H, Mathieson T, Childs D, Sweetman GMA, Werner T, Tögel I, et al. Thermal proteome profiling for unbiased identification of direct and indirect drug targets using multiplexed quantitative mass spectrometry. Nature Protocols. 2015 Oct 1;10(10):1567–93. (doi: 10.1038/nprot.2015.101)

Part of the workflow uses the following python package: 

* [**SC3s**](https://github.com/hemberg-lab/sc3s): hemberg-lab/sc3s

## Citation

If you use this code, please cite: Integrating Cell Painting and Thermal Proteome Profiling for Improved Inference of Mechanism of Action ([update with bioRxiv preprint doi])