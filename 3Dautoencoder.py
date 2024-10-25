"""
Code to train a 3D autoencoder finetuning a pretrained model
"""

import argparse
import os
import sys
import torch

from monai.data import DataLoader, Dataset
from monai.data.utils import partition_dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    Resized,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from graph_learning.vit_vae import ViTAutoEncModule
import pytorch_lightning as pl
import nibabel as nib
import numpy as np

import yaml
import wandb

os.environ["WANDB_SILENT"] = "true"


def load_bids_dataset(bids_root, modality):
    """
    Load a BIDS dataset.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        modality (str): Modality to load (e.g., "T1w", "T2w").

    Returns:
        list[dict]: A list of dictionaries, each containing the image and metadata.
    """
    dataset = []
    # Example BIDS path: sub-01/ses-00/anat/sub-01_T1w.nii.gz
    for subdir, dirs, files in os.walk(bids_root):
        for f in files:
            if modality in f and f.endswith("_T2w_masked.nii.gz"):
                data = {"image": os.path.join(subdir, f)}
                dataset.append(data)
    return dataset


def prepare_dataloader(
    args,
    amp=False,
):
    """
    Prepare the data loader for training and validation.

    Args:
        config (object): The arguments object containing the configuration parameters.
        amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            CropForegroundd(
                keys="image", margin=0, source_key="image"
            ),  # does this work?
            Resized(keys="image", spatial_size=args.img_size),
            # Data Augmentation
            RandAffined(  # Random affine transformations
                keys=["image"],
                prob=0.5,  # Probability to apply the transform
                rotate_range=(0, 0, np.pi / 15),  # Rotation range in radians
                scale_range=(0.1, 0.1, 0.1),  # Scaling range
                translate_range=(15, 15, 15),  # Translation range in pixels
            ),
            RandFlipd(  # Random flipping
                keys=["image"],
                prob=0.5,  # Probability to apply the transform
                spatial_axis=[0, 1],  # Flip axis, 0 for horizontal, 1 for
            ),
            RandGaussianNoised(  # Random Gaussian noise
                keys=["image"], prob=0.5, mean=0.0, std=0.1
            ),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=255
            ),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            CropForegroundd(
                keys="image", margin=0, source_key="image"
            ),  # does this work?
            Resized(keys="image", spatial_size=args.img_size),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=255
            ),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )

    dataset_items = load_bids_dataset(args.bids_root, args.modality)

    ratios = args.ratios
    # Use MONAI's partition_dataset
    # It automatically shuffles the dataset before splitting.
    train_set, val_set, test_set = partition_dataset(
        dataset_items, ratios, shuffle=True, seed=42
    )

    # Create datasets
    train_dataset = Dataset(data=train_set, transform=train_transforms)
    val_dataset = Dataset(data=val_set, transform=val_transforms)
    test_dataset = Dataset(data=test_set, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0.0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main(args):

    # DataLoader
    train_loader, val_loader, test_loader = prepare_dataloader(
        args,
        amp=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="logs/",
        filename="vit-vae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    wandb_logger = WandbLogger()

    args.T_max = len(train_loader) * args.max_epochs

    # Initialize the ViTAutoEncModule
    model = ViTAutoEncModule(args)

    # Initialize the Trainer
    trainer = pl.Trainer(
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=args.enable_progress_bar,
        max_epochs=args.n_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,  # Add the WandbLogger here
    )

    # Train the model
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # LOG the best model in WANDB
    wandb.save(checkpoint_callback.best_model_path)

    # Evaluate the model
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()


if __name__ == "__main__":

    # argparse the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/vitae_base.yaml"
    )
    args = parser.parse_args()

    config_file = args.config

    # Load the configuration file
    try:
        with open(config_file, encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: The configuration file {config_file} does not exist.")
        sys.exit(1)

    if not isinstance(config, argparse.Namespace):
        config = argparse.Namespace(**config)

    # Initialize a new wandb run
    wandb.init(project="vit-3d-ae", config=config)

    main(config)
