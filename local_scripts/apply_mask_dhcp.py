"""
Quick script that applies the mask to the nifti files in the dhcp dataset.
"""

import os
import nibabel as nib
import numpy as np

# Set the path to the directory containing the nifti files
data_dir = "/home/gmarti/DATA/DHCP_FETAL/dhcp_anat_pipeline/"

# For each folder inside data_dir
for folder in os.listdir(data_dir):

    for ses_folder in os.listdir(os.path.join(data_dir, folder)):

        # if ses_folder is not a directory, skip it
        if not os.path.isdir(os.path.join(data_dir, folder, ses_folder)):
            continue
        
        sub = folder.split("-")[1]
        ses = ses_folder.split("-")[1]

        # Set the path to the mask file
        nifti_file = os.path.join(data_dir, folder, ses_folder, "anat", f"sub-{sub}_ses-{ses}_desc-restore_T2w.nii.gz")
        mask_file = os.path.join(data_dir, folder, ses_folder, "anat", f"sub-{sub}_ses-{ses}_desc-brainmask.nii.gz")

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
        masked_nifti_file = os.path.join(data_dir, folder, ses_folder, "anat", f"sub-{sub}_ses-{ses}_desc-restore_T2w_masked.nii.gz")
        nib.save(masked_nifti, masked_nifti_file)