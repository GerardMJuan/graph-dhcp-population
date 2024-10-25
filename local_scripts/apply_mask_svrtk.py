"""
Script that applies the mask to the input image using SVRTK
and saves it to the dhcp data directory with the appropiate format
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd

# Set the path to the directory containing the nifti files
output_dir = "/home/gmarti/DATA/DHCP_FETAL/dhcp_anat_pipeline/"
data_dir = "/home/gmarti/DATA/ERANEU_MULTIFACT/derivatives/svrtk_full_seg"
csv_qc_dir = os.path.join(data_dir, "qc_svrtk.xlsx")

df_qc = pd.read_excel(csv_qc_dir)

dx_file = "/home/gmarti/DATA/ERANEU_merged.csv"
df_dx = pd.read_csv(dx_file)


# For each folder inside data_dir
for folder in os.listdir(data_dir):

    # if ses_folder is not a directory, skip it
    if not os.path.isdir(os.path.join(data_dir, folder)):
        continue

    for ses_folder in os.listdir(os.path.join(data_dir, folder)):

        # if ses_folder is not a directory, skip it
        if not os.path.isdir(os.path.join(data_dir, folder, ses_folder)):
            continue

        sub = folder.split("-")[1]
        ses = ses_folder.split("-")[1]

        # Check if the subject has been QCed
        # subject is determined by column participant_id in the qc csv
        # if column "svrtk" is 0, skip the subject
        if (
            df_qc.loc[df_qc["participant_id"] == f"sub-{sub}"]["svrtk"].values[
                0
            ]
            == 0
        ):
            print(f"Skipping sub-{sub} ses-{ses} because it has QC=0")
            continue

        # check if participant_id in file df_dx has GRUPO either 4, 5 or 6
        # if not, skip the subject
        if df_dx.loc[df_dx["participant_id"] == f"sub-{sub}"]["GRUPO"].values[
            0
        ] not in [1, 2, 3]:
            print(
                f"Skipping sub-{sub} ses-{ses} because it is not in GRUPO 1, 2 or 3"
            )
            continue

        # Set the path to the mask file
        nifti_file = os.path.join(
            data_dir,
            folder,
            ses_folder,
            "anat",
            "input",
            "reo-SVR-output-brain.nii.gz",
        )
        mask_file = os.path.join(
            data_dir,
            folder,
            ses_folder,
            "anat",
            "reo-SVR-output-brain-mask-bet-1.nii.gz",
        )

        # try if they both exists, if not, continue
        if not os.path.exists(nifti_file) or not os.path.exists(mask_file):
            print(f"File {nifti_file} or {mask_file} does not exist, skipping")
            continue

        # Load the mask file
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata()

        # Load the nifti file
        nifti = nib.load(nifti_file)
        nifti_data = nifti.get_fdata()

        # Apply the mask to the nifti file
        nifti_data = nifti_data * mask_data

        # Save the masked nifti file
        masked_nifti = nib.Nifti1Image(nifti_data, nifti.affine)
        masked_nifti_file = os.path.join(
            output_dir,
            folder,
            ses_folder,
            "anat",
        )

        # create the directory if it does not exist
        if not os.path.exists(masked_nifti_file):
            os.makedirs(masked_nifti_file)

        masked_nifti_file = os.path.join(
            masked_nifti_file,
            f"sub-{sub}_ses-{ses}_desc-restore_T2w_masked.nii.gz",
        )

        nib.save(masked_nifti, masked_nifti_file)
