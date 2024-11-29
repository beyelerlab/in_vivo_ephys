# Population dynamics analysis

# Quick overview

Script used for the single-unit analysis of anxiety- and valence-related activity in the anterior insular cortex. cf. paper [Dopamine transmission in the anterior insula shapes the neural coding of anxiety](https://www.biorxiv.org/content/10.1101/2024.10.25.620186v2).

Original script available here: https://gitlab.com/TanmaiR/ephys_beyelerlabb

It performed linear (principle component analysis, PCA) and non-linear (deep learning using [cebra](https://cebra.ai/docs/index.html)) dimensionality reduction analyses, from two assays: anxiety-related in the elevated plus maze (Fig. 4 and Fig. 6) and valence-related assay during sucrose/quinine consumptio (Fig. 5 and Fig. 6). 

`EPM_PCA.ipynb`: Implement PCA 
`EPM_CEBRA.ipynb`: Implement CEBRA 
`SQ_PCA.ipynb`: Implement PCA 
`SQ_CEBRA.ipynb`: Implement CEBRA

Rest of the files are the helper functions used in the analysis.

# Getting started 

## Installation 

Clone the repository your local machine. 

Install the requirements.

Python version 3.8.18

## Hardware requirements 

RAM: 64.0 GB
Processor: 12th Gen Intel(R) Core(TM) i7-12700   2.10 GHz

# Demo 

### Input file structure 

These scripts assume that they are dealing with the preprocessed data. Usage instructions are already present in the begining of each jupyter notebook.

Raw data can be find here: LINK
