"""
Small script that loads a specific model from wandb and test its attention maps and 
its representation of the latent space on the database.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from monai.data import DataLoader, Dataset
from graph_learning.vit_vae import ViTAutoEncModule
import yaml
import argparse
from utils.data import load_bids_dataset, prepare_dataloader
import pandas as pd

# specify a run
run_id = "entfz9s7"
api = wandb.Api()
run = api.run(f"upf-fetal/vit-3d-ae/{run_id}")

# load configuration file from wandb
config = run.config

# remove key "config" from the dictionary
config = {key: config[key] for key in config if key != "config"}

if not isinstance(config, argparse.Namespace):
    config = argparse.Namespace(**config)

# Load the model from the logs folder
model = ViTAutoEncModule.load_from_checkpoint("logs/vit-vae-epoch=299-val_loss=98.43.ckpt")

# Now, from this model, extract both 1) attention maps for a specific image and 2) the latent space representation for all images

# Load the dataset
load_bids_dataset(config.bids_root, config.modality)

# load the data 
df_main = pd.read_csv("data/dhcp_multifact_ga_full.csv", index_col=0)

# Prepare val transforms
# all is testing, so update ratios
test_loader = prepare_dataloader(
    config,
    amp=False,
    separate=False
)

# # Create lists to store the images and hidden states
# list_of_images = []
# list_of_hidden_states = []
# list_of_subjects = []

# # Get the first image from the test_loader
# for i, batch in enumerate(test_loader):
#     print("Batch number: ", i)
#     subjects = batch["subject"]
#     batch = batch["image"].to(model.device)
#     print(subjects)

#     output, hidden_states = model.forward(batch)

#     # COnvert outputs to numpy
#     outputs = output.detach().cpu().numpy()

#     # Convert hidden states to numpy
#     hidden_states = hidden_states[-1].detach().cpu().numpy()

#     list_of_images.append(outputs)
#     # append only the last hidden state
#     list_of_hidden_states.append(hidden_states)

#     # append the subjects
#     list_of_subjects.append(subjects)

#     # clear memory
#     del batch
#     del output
#     del hidden_states

# # merge the first two dimenions of each array
# list_of_images = np.concatenate(list_of_images, axis=0)
# list_of_hidden_states = np.concatenate(list_of_hidden_states, axis=0)
# list_of_subjects = np.concatenate(list_of_subjects, axis=0)

# # print size
# print(list_of_images.shape)
# print(list_of_hidden_states.shape)
# print(list_of_subjects.shape)

# # save list of hidden states to disk
# np.save("hidden_states.npy", list_of_hidden_states)

# # save the list of subjects
# np.save("subjects.npy", list_of_subjects)

hidden_states = np.load("hidden_states.npy")
subjects = np.load("subjects.npy")

# combine two last dimensions of hidden states concatenating them
def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])


hidden_states = combine_dims(hidden_states, 1) # combines dimension 1 and 2

# Get the gestational age and the diagnosis from the dataframe
# in the order of the subjects (column participant_id)
gestational_age = df_main.loc[subjects, "gestational_weeks"]
diagnosis = df_main.loc[subjects, "GRUPO"]
# convert diagnosis to indic

dict_dx = {
    "CONTROL": 0,
    "IUGR": 1,
    "CC<5": 2,
    "DISGENESIA": 3,
    "ACC": 4
}

# Apply PCA to the hidden states
from sklearn.decomposition import PCA
import seaborn as sns 

pca = PCA(n_components=2)
pca.fit(hidden_states)

# Transform the hidden states
hidden_states_pca = pca.transform(hidden_states)

# Create a DataFrame with the necessary data
df_pca = pd.DataFrame(hidden_states_pca, columns=['PCA1', 'PCA2'])
df_pca['Gestational_Age'] = gestational_age.values  # Assuming gestational_age is a Series
df_pca['Diagnosis'] = diagnosis.values

# Remove all subjects with gestational age < 10
df_pca = df_pca[df_pca['Gestational_Age'] >= 10]

# Plot with Seaborn for gestational age
# Plot
plt.figure(figsize=(10, 6))
points = plt.scatter(data=df_pca, x='PCA1', y='PCA2', c='Gestational_Age', cmap="viridis")

# Adding colorbar
plt.colorbar(points, label='Gestational Age (weeks)')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Hidden States by Gestational Age")
plt.savefig("pca_hidden_states_age_seaborn.png")

plt.close()

# Plot with Seaborn for diagnosis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Diagnosis', legend="full")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Hidden States by Diagnosis")
# For a legend that maps diagnosis codes to dia gnoses, consider adding custom legend handling here
plt.savefig("pca_hidden_states_dx_seaborn.png")
