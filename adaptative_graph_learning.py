"""Main script for testing the adaptative graph learning algorithm From Bintsi et al. 
https://github.com/bintsi/adaptive-graph-learning on the dhcp dataset.
"""

import wandb
import yaml
import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from argparse import Namespace
from utils.data import preprocess_data
from graph_adapt_learning.graph_construction import PopulationGraphUKBB, UKBBageDataset
from graph_adapt_learning.model import GraphLearningModel
from utils.plot import plot_graph

os.environ["WANDB_SILENT"]="true"

def main(config):

    # Initialize wandb
    wandb.init(project="adaptive-graph-learning-dhcp", config=config)

    # Load the data
    dhcp_data = pd.read_csv(config.data_path, index_col=0)
    wandb.log({"data": wandb.Table(dataframe=dhcp_data)})

    # Print tthe shape of the data
    print(f"Data shape: {dhcp_data.shape}")

    # Preprocess the data
    dhcp_data_preprocessed = preprocess_data(dhcp_data, normalize=True)
    wandb.log({"data_preprocessed": wandb.Table(dataframe=dhcp_data_preprocessed)})

    # Divide into train, validation, and test sets
    train_data = dhcp_data_preprocessed.sample(frac=0.8, random_state=42)
    test_data = dhcp_data_preprocessed.drop(train_data.index)
    val_data = train_data.sample(frac=0.1, random_state=42)
    train_data = train_data.drop(val_data.index)
    wandb.log({"train_data": wandb.Table(dataframe=train_data)})
    wandb.log({"val_data": wandb.Table(dataframe=val_data)})
    wandb.log({"test_data": wandb.Table(dataframe=test_data)})

    # Print size of test, train, and val
    print(f"Train size: {train_data.shape[0]}")
    print(f"Val size: {val_data.shape[0]}")
    print(f"Test size: {test_data.shape[0]}")

    # Define the parameters of the model
    # config.phenotype_columns = train_data.columns[1:]
    # config.node_columns = train_data.columns[2:]  # same but without sex

    print(f"Phenotype columns: {config.phenotype_columns}")
    print(f"Node columns: {config.node_columns}")

    config.num_node_features = len(config.node_columns)

    # TODO: This needs to be inputted in the config file, or removed!
    config.conv_layers[0][0] = config.num_node_features
    config.dgm_layers[0] = [
        len(config.phenotype_columns),
        len(config.phenotype_columns)
    ]

    task = config.task
    num_classes = config.num_classes
    k = config.k
    edges = config.edges

    ## Create the population graph
    population_graph = PopulationGraphUKBB(
        train_data, val_data, test_data,
        config.phenotype_columns, config.node_columns, config.num_node_features,
        task, num_classes, k, edges
        )

    population_graph = population_graph.get_population_graph()

    plot = plot_graph(population_graph, population_graph.y)
    wandb.log({"initial_graph": plot})

    # Create the datasets and DataLoaders
    train_data_g = UKBBageDataset(
        graph=population_graph, split='train',
        device='cuda', num_classes=num_classes
        )
    val_data_g = UKBBageDataset(
        graph=population_graph, split='val',
        samples_per_epoch=1, num_classes=num_classes
        )
    test_data_g = UKBBageDataset(
        graph=population_graph, split='test',
        samples_per_epoch=1, num_classes=num_classes
        )

    train_loader = DataLoader(train_data_g, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_data_g, batch_size=1, num_workers=1)
    test_loader = DataLoader(test_data_g, batch_size=1)

    # Class for lightning data module
    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    # Define the model
    # configure input feature sizes
    if config.pre_fc is None or len(config.pre_fc) == 0:
        if len(config.dgm_layers[0]) > 0:
            config.dgm_layers[0][0] = train_data_g.phenotypes.shape[1]
        config.conv_layers[0][0] = train_data_g.n_features
    else:
        config.pre_fc = train_data_g.n_features

    if config.fc_layers is not None:
        config.fc_layers[-1] = train_data_g.num_classes

    # TRAIN THE MODEL
    model = GraphLearningModel(config)
    print(model)

    # Evaluate and save the model
    if config.task == 'regression':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min'
        )
    elif config.task == 'classification':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_acc',
            mode='max')
    else:
        raise ValueError('Task should be either regression or classification.')

    callbacks = [checkpoint_callback]
    if val_data_g == test_data_g:
        callbacks = None

    # TODO change the logger to wandb
    logger = WandbLogger(name="adaptive-graph-learning-dhcp")

    # This could be done with CLI and much easier but I'm not sure how to do it
    trainer = pl.Trainer(
        log_every_n_steps=config.log_every_n_steps,
        max_epochs=config.max_epochs,
        enable_progress_bar=config.enable_progress_bar,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        logger=logger, callbacks=callbacks
        )

    trainer.fit(model, datamodule=MyDataModule())

    # Evaluate results on validation and test set
    # Results will be logged automatically
    val_results = trainer.validate(
        ckpt_path=checkpoint_callback.best_model_path,
        dataloaders=val_loader
    )
    test_results = trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        dataloaders=test_loader
    )

    # Predict the validation and test results
    # DOES NOT WORK, FIX THIS
    pred_val = trainer.predict(model, val_loader)
    pred_test = trainer.predict(model, test_loader)
    pred_train = trainer.predict(model, train_loader)

    # Manually extract, improve when I have time
    x_val = pred_val[0][0]
    x_test = pred_test[0][0]
    x_train = pred_train[0][0]

    y_val = pred_val[0][1]
    y_test = pred_test[0][1]
    y_train = pred_train[0][1]

    w_val = pred_val[0][2].squeeze()
    w_test = pred_test[0][2].squeeze()
    w_train = pred_train[0][2].squeeze()

    # Save to 3 dataframe with columns x and y
    val_ga = pd.DataFrame({"x": x_val.numpy().squeeze(), "y": y_val.numpy().squeeze()})
    test_ga = pd.DataFrame({"x": x_test.numpy().squeeze(), "y": y_test.numpy().squeeze()})
    train_ga = pd.DataFrame({"x": x_train.numpy().squeeze(), "y": y_train.numpy().squeeze()})

    logger.log_table(key="pred_val_ga", data=val_ga)
    logger.log_table(key="pred_test_ga", data=test_ga)
    logger.log_table(key="pred_train_ga", data=train_ga)

    # Apply the learned weights to the phenotypes
    train_data[config.phenotype_columns] = train_data[config.phenotype_columns]*w_train
    val_data[config.phenotype_columns] = val_data[config.phenotype_columns]*w_val
    test_data[config.phenotype_columns] = test_data[config.phenotype_columns]*w_test

    ## Plot the final graph with the learned weights
    population_graph = PopulationGraphUKBB(
        train_data, val_data, test_data,
        config.phenotype_columns, config.node_columns, 
        config.num_node_features,
        task, num_classes, k, edges
        )

    population_graph = population_graph.get_population_graph()

    # Plot the graph
    plot = plot_graph(population_graph, population_graph.y)

    # the return is a mtplotlib figure, so we can log it
    wandb.log({"final_graph": plot})

    # Close wandb
    wandb.finish()


if __name__ == "__main__":

    # argparse the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_hyper.yaml")
    args = parser.parse_args()

    config_file = args.config

    # Load the configuration file
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: The configuration file {config_file} does not exist.")
        sys.exit(1)

    if type(config) is not Namespace:
        config = Namespace(**config)
    
    main(config)
