"""Utils for data manipulation.
"""

import pandas as pd
import numpy as np
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
import argparse
import numpy as np
import torch
import os

def preprocess_data(data, normalize=False):
    """Preprocess the data. Remove outliers, remove NA, etc.
    Also normalize the data if needed.

    Args:
        data (pd.DataFrame): The data to preprocess.
        normalize (bool, optional): Whether to normalize the data. Defaults to False.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    # Convert column "sex" to numerical
    data['sex_at_birth'] = data['sex_at_birth'].replace({'M': 0, 'F': 1})

    # Make sure that all coluns but the first 3 are numerical
    data = data.apply(pd.to_numeric, errors='coerce')

    # Remove NA
    data = data.dropna()

    # Preprocess the data
    # First three columns (subject_id, age, sex) are not preprocessed
    if normalize:
        data.iloc[:, 2:] = (data.iloc[:, 2:] - data.iloc[:, 2:].min()) / (data.iloc[:, 2:].max() - data.iloc[:, 2:].min())
    
    # data = data[(data > data.quantile(0.05)) & (data < data.quantile(0.95))]

    return data




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
                
                # extract subject ID from the path
                # subdir is something like "data/sub-01/ses-00/anat"
                # we want to extract "sub-01"

                # split the path by "/"
                parts = subdir.split("/")
                # get the last part
                subject_id = parts[-3]

                # remove the sub
                subject_id = subject_id.replace("sub-", "")
                
                # if numeric, remove trailing zeros
                if subject_id.isdigit():
                    subject_id = str(int(subject_id))

                data = {"image": os.path.join(subdir, f),
                        "subject": subject_id
                       }
                dataset.append(data)
    return dataset


def prepare_dataloader(
    args,
    amp=False,
    separate=True
):
    """
    Prepare the data loader for training and validation.

    Args:
        config (object): The arguments object containing the configuration parameters.
        amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        separate (bool): Whether to separate the dataset into training, validation, and test sets.


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

    if separate:
        # Split the dataset into training, validation, and test sets
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
    else:
        dataset = Dataset(data=dataset_items, transform=val_transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        return data_loader

