"""
Script combining the VIT model with the graph model.

Capture the hidden states from the VIT model and use them as input to the graph model.

Combine the loss of the graph model with the loss of the VIT model.

See what happens when you train the model with this new loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import wandb



NOMÉS HE COPIAT ELS CONTROLS I ELS IUGR PER ENTRENAMENT