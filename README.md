# graph-dhcp-population
Collection of scripts for testing various population graph learning algorithms for fetal MRI derived data.

## Scripts

### 3Dautoencoder.py
Main script for training and evaluating a 3D Vision Transformer Autoencoder on fetal brain MRI data.

### adaptative_graph_learning.py
Main script for testing the adaptive graph learning algorithm on the dHCP dataset.

### data_preparation.py
Script for preparing data for the graph learning model, combining data from dHCP and MULTIFACT datasets.

### VitAE_analysis.py
Script for analyzing a specific ViT Autoencoder model, including attention maps and latent space representation.

### MKL.ipynb
Jupyter notebook for testing the unsupervised MKL algorithm on the dHCP dataset.

### MKL_experiment.py
Script for testing the unsupervised MKL algorithm on the dHCP dataset and doing hyperparameter search.

## Local Scripts

### apply_mask_dhcp.py
Quick script that applies the mask to the nifti files in the dHCP dataset.

### apply_mask_svrtk.py
Script that applies the mask to the input image using SVRTK and saves it to the dHCP data directory.

## Configurations
## Graph Learning

The `graph_adapt_learning/` directory referos to the implementation of the Pytorch Lightning implementation for the paper 
[Multimodal brain age estimation using interpretable adaptive population-graph learning](https://arxiv.org/abs/2307.04639)
(MICCAI 2023) by Kyriaki-Margarita Bintsi, Vasileios Baltatzis, Rolandos Alexandros Potamias, Alexander Hammers, and Daniel Rueckert. The code can be found in the [adaptive-graph-learning](https://github.com/bintsi/adaptive-graph-learning) repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
