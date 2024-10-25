"""
This script is used to prepare the data for the graph learning model
Combining data from dHCP and from the MULTIFACT dataset

First, what it does is to compare the distribution of the various parameters.

Afterwards, we will try to normalize them and see if the distributions are more similar using an existing method (COMBAT)?
"""
import pandas as pd
import numpy as np

data = pd.read_csv("data/dhcp_quantitative_values.csv", index_col=0)

# need to create columns "WM", "Deep GM", "Cortical GM"
# each of those columns are the sum of the corresponding columns in the data

# Cerebellum 

correspondences = {
    "WM": [
        "Anterior temporal lobe, medial part left WM",
        "Anterior temporal lobe, medial part right WM",
        "Anterior temporal lobe, lateral part left WM",
        "Anterior temporal lobe, lateral part right WM",
        "Gyri parahippocampalis et ambiens anterior part left WM",
        "Gyri parahippocampalis et ambiens anterior part right WM",
        "Superior temporal gyrus, middle part left WM",
        "Superior temporal gyrus, middle part right WM",
        "Medial and inferior temporal gyri anterior part left WM",
        "Medial and inferior temporal gyri anterior part right WM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left WM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right WM",
        "Insula right WM",
        "Insula left WM",
        "Occipital lobe right WM",
        "Occipital lobe left WM",
        "Gyri parahippocampalis et ambiens posterior part right WM",
        "Gyri parahippocampalis et ambiens posterior part left WM",
        "Medial and inferior temporal gyri posterior part right WM",
        "Medial and inferior temporal gyri posterior part left WM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left WM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right WM",
        "Medial and inferior temporal gyri posterior part right WM",
        "Medial and inferior temporal gyri posterior part left WM",
        "Superior temporal gyrus, posterior part right WM",
        "Superior temporal gyrus, posterior part left WM",
        "Cingulate gyrus, anterior part right WM",
        "Cingulate gyrus, anterior part left WM",
        "Cingulate gyrus, posterior part right WM",
        "Cingulate gyrus, posterior part left WM",
        "Frontal lobe right WM",
        "Frontal lobe left WM",
        "Parietal lobe right WM",
        "Parietal lobe left WM",
    ],
    "Deep GM": [
        "Hippocampus left",
        "Hippocampus right",
        "Amygdala left",
        "Amygdala right",
        "Thalamus left, high intensity part in T2",
        "Thalamus right, high intensity part in T2",
        "Subthalamic nucleus right",
        "Subthalamic nucleus left ",
        "Caudate nucleus right",
        "Caudate nucleus left",
        "Lentiform Nucleus left",
        "Lentiform Nucleus right",
        "Corpus Callosum",
        ],
    "Cortical GM": [
        "Anterior temporal lobe, medial part left GM",
        "Anterior temporal lobe, medial part right GM",
        "Anterior temporal lobe, lateral part left GM",
        "Anterior temporal lobe, lateral part right GM",
        "Gyri parahippocampalis et ambiens anterior part left GM",
        "Gyri parahippocampalis et ambiens anterior part right GM",
        "Superior temporal gyrus, middle part left GM",
        "Superior temporal gyrus, middle part right GM",
        "Medial and inferior temporal gyri anterior part left GM",
        "Medial and inferior temporal gyri anterior part right GM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left GM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right GM",
        "Insula right GM",
        "Insula left GM",
        "Occipital lobe right GM",
        "Occipital lobe left GM",
        "Gyri parahippocampalis et ambiens posterior part right GM",
        "Gyri parahippocampalis et ambiens posterior part left GM",
        "Medial and inferior temporal gyri posterior part right GM",
        "Medial and inferior temporal gyri posterior part left GM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left GM",
        "Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right GM",
        "Superior temporal gyrus, posterior part right GM",
        "Superior temporal gyrus, posterior part left GM",
        "Cingulate gyrus, anterior part right GM",
        "Cingulate gyrus, anterior part left GM",
        "Cingulate gyrus, posterior part right GM",
        "Cingulate gyrus, posterior part left GM",
        "Frontal lobe right GM",
        "Frontal lobe left GM",
        "Parietal lobe right GM",
        "Parietal lobe left GM",
    ],
}

# add the columns to data
for key in correspondences.keys():
    data[key] = 0

# sum the values
for key, values in correspondences.items():
    data[key] = data[values].sum(axis=1)

# save the data
data.to_csv("data/dhcp_quantitative_values_summed.csv")