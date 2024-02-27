"""Utils for data manipulation.
"""

import pandas as pd
import numpy as np

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


