# graph-dhcp-population
Testing various graph learning algorithms for dHCP

## Scripts

### 3Dautoencoder.py
Main script for training and evaluating a 3D Vision Transformer Autoencoder on fetal brain MRI data.

### adaptative_graph_learning.py
Main script for testing the adaptive graph learning algorithm on the dHCP dataset.

### combat_harmonization.R
R script for applying ComBat harmonization to merge and normalize data from dHCP and MULTIFACT datasets.

### data_analysis_samescript.ipynb
Jupyter notebook for general data analysis, including merging dHCP and MULTIFACT data, applying QC, and visualizing key variables.

### data_preparation.py
Script for preparing data for the graph learning model, combining data from dHCP and MULTIFACT datasets.

### VitAE_analysis.py
Script for analyzing a specific ViT Autoencoder model, including attention maps and latent space representation.

### vit_graph.py
Script combining the Vision Transformer model with the graph model for fetal brain analysis.

## Local Scripts

### apply_mask_dhcp.py
Quick script that applies the mask to the nifti files in the dHCP dataset.

### apply_mask_svrtk.py
Script that applies the mask to the input image using SVRTK and saves it to the dHCP data directory.

## Configurations
## Graph Learning

The `graph_learning/` directory referos to the implementation of the Pytorch Lightning implementation for the paper 
[Multimodal brain age estimation using interpretable adaptive population-graph learning](https://arxiv.org/abs/2307.04639)
(MICCAI 2023) by Kyriaki-Margarita Bintsi, Vasileios Baltatzis, Rolandos Alexandros Potamias, Alexander Hammers, and Daniel Rueckert. The code can be found in the [adaptive-graph-learning](https://github.com/bintsi/adaptive-graph-learning) repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
