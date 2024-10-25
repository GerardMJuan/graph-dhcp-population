import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import wandb
import math
from wandb import Image as WandbImage

from PyMKL import MKL
from PyMKL.kernels import kernel_stack, get_W_and_D
from pygam import LinearGAM, s

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_and_preprocess_data(file_path, features, detrend=True, norm=False):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.
    features (list): List of features to use.
    detrend (bool): Whether to detrend the features based on gestational weeks.
    norm (bool): Whether to normalize the features.

    Returns:
    tuple: DataFrame, feature matrix, and gestational weeks array.
    """
    df = pd.read_csv(file_path)
    # Map group labels
    GRUPO_labels = {
        1: 'ACC',
        2: 'DISGENESIA_CC',
        3: 'CC<5',
        4: 'IUGR',
        5: 'CONTROLS VM',
        6: 'CONTROLS  MULTIFACT',
        7: 'CONTROLS DHCP',
        8: "VM"
    }
    df['DX'] = df['GRUPO'].map(GRUPO_labels)
    gest_weeks = df['gestational_weeks'].values
    df_subset = df[features]
    
    if detrend:
        df_subset = detrend_features(df_subset, gest_weeks)
    
    if norm:
        df_subset[features] = (df_subset[features] - df_subset[features].mean()) / df_subset[features].std()
    
    X_full = df_subset.values
    return df, X_full, gest_weeks


def detrend_features(df_subset, gest_weeks):
    """
    Detrend features based on gestational weeks using a Generalized Additive Model (GAM).

    Parameters:
    df_subset (DataFrame): DataFrame containing the features to detrend.
    gest_weeks (array): Array of gestational weeks.

    Returns:
    DataFrame: Detrended features.
    """
    residuals = []
    for feature in df_subset.columns:
        biomarker_values = df_subset[feature]
        gestational_ages = gest_weeks.reshape(-1, 1)
        gam = LinearGAM(s(0))
        gam.gridsearch(gestational_ages, biomarker_values)
        predicted = gam.predict(gestational_ages)
        residuals.append(biomarker_values - predicted)
    X_residuals = np.array(residuals).T
    df_detrended = pd.DataFrame(X_residuals, columns=df_subset.columns)
    return df_detrended

def create_kernels(X_full, kernel_type, alpha, knn):
    """
    Create kernel matrices using the provided feature matrix.

    Parameters:
    X_full (ndarray): Feature matrix.
    kernel_type (str): Type of kernel to use (e.g., 'euclidean').
    alpha (float): Hyperparameter controlling kernel combination.
    knn (int): Number of nearest neighbors.

    Returns:
    tuple: Combined kernel matrix, weight matrix, and degree matrix.
    """
    X = [X_full[:, i] for i in range(X_full.shape[1])]
    kernel = [kernel_type for _ in range(len(X))]
    K, var = kernel_stack(X, kernel, alpha=alpha, knn=knn, return_sigmas=False)
    W, D = get_W_and_D(K, var)
    return K, W, D

def train_mkl(K, W, D):
    """
    Train a Multiple Kernel Learning (MKL) model.

    Parameters:
    K (ndarray): Combined kernel matrix.
    W (ndarray): Weight matrix.
    D (ndarray): Degree matrix.

    Returns:
    tuple: Trained MKL model and projected data.
    """
    mkl = MKL(K=K, W=W, D=D, solver='smcp', maxiter=50)
    mkl.fit()
    Proj = mkl.transform()
    return mkl, Proj

def perform_clustering(Proj, n_clusters=4):
    """
    Perform agglomerative clustering on the projected data.

    Parameters:
    Proj (ndarray): Projected data from MKL.
    n_clusters (int): Number of clusters to form.

    Returns:
    ndarray: Cluster labels for each sample.
    """
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = cluster.fit_predict(Proj)
    return cluster_labels

def evaluate_clusters(df, cluster_labels, Proj, gest_weeks):
    """
    Evaluate the quality of the clusters using various metrics.

    Parameters:
    df (DataFrame): Original data DataFrame.
    cluster_labels (ndarray): Cluster labels for each sample.
    Proj (ndarray): Projected data from MKL.
    gest_weeks (array): Gestational weeks array.

    Returns:
    dict: Dictionary containing cluster evaluation metrics.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    from scipy.stats import spearmanr
    
    # Silhouette Score
    silhouette_avg = silhouette_score(Proj, cluster_labels)
    
    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(Proj, cluster_labels)
    
    # Davies-Bouldin Index
    db_score = davies_bouldin_score(Proj, cluster_labels)
    
    # Correlation with Gestational Age
    corr_gest_age, p_value = spearmanr(cluster_labels, gest_weeks)
    
    # Adjusted Mutual Information with 'DX'
    ami_score = adjusted_mutual_info_score(df['DX'], cluster_labels)
    
    metrics = {
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_score': db_score,
        'gest_age_correlation': corr_gest_age,
        'gest_age_p_value': p_value,
        'adjusted_mutual_info': ami_score
    }
    # Cluster Stability
    stability = assess_cluster_stability(Proj, n_clusters=4, n_runs=5)
    metrics['cluster_stability'] = stability
    
    return metrics


def assess_cluster_stability(Proj, n_clusters=4, n_runs=5):
    """
    Assess the stability of clusters by running clustering multiple times and calculating the Adjusted Rand Index (ARI).

    Parameters:
    Proj (ndarray): Projected data from MKL.
    n_clusters (int): Number of clusters to form.
    n_runs (int): Number of times to run the clustering algorithm.

    Returns:
    float: Average ARI score across runs.
    """
    cluster_labels_runs = []
    for _ in range(n_runs):
        cluster_labels = perform_clustering(Proj, n_clusters)
        cluster_labels_runs.append(cluster_labels)
    
    # Compute pairwise ARI
    ari_scores = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            ari = adjusted_rand_score(cluster_labels_runs[i], cluster_labels_runs[j])
            ari_scores.append(ari)
    stability = np.mean(ari_scores)
    return stability

def plot_embedding(Proj, df, save_dir, method='t-SNE'):
    """
    Plot the 2D embedding of the projected data using t-SNE or PCA.

    Parameters:
    Proj (ndarray): Projected data from MKL.
    df (DataFrame): Original data DataFrame.
    save_dir (str): Directory to save the plot.
    method (str): Method to use for dimensionality reduction ('t-SNE' or 'PCA').
    """
    # Choose the dimensionality reduction method
    if method == 't-SNE':
        reducer = TSNE(n_components=3, random_state=42)
    elif method == 'PCA':
        reducer = PCA(n_components=3)
    else:
        raise ValueError("Method should be 't-SNE' or 'PCA'")

    embedding = reducer.fit_transform(Proj)

    # Set seaborn style for talk
    sns.set_context(context="talk")

    dimension_pairs = [(0, 1), (0, 2), (1, 2)]
    
    # Lump together all controls
    df['DX'] = df['DX'].apply(lambda x: 'IUGR' if x == 'IUGR' else 'VM' if x == 'VM' else 'CONTROLS')

    for (x, y) in dimension_pairs:
        # Create a figure with subplots
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        cmap_gestational_weeks = sns.cubehelix_palette(as_cmap=True)
        cmap_volume = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

        # Plot for Gestational Weeks with a colorbar
        sns.scatterplot(x=embedding[:, x], y=embedding[:, y], hue=df['gestational_weeks'],
                        palette=cmap_gestational_weeks, ax=ax[0], s=100, edgecolor='w', legend=False)
        ax[0].set_title('Gestational Weeks')
        ax[0].set_xlabel(f'Dimension {x}')
        ax[0].set_ylabel(f'Dimension {y}')
        norm = plt.Normalize(df['gestational_weeks'].min(), df['gestational_weeks'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap_gestational_weeks, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[0], orientation='vertical', label='Gestational Weeks')

        # Plot for Group with discrete legend
        sns.scatterplot(x=embedding[:, x], y=embedding[:, y], hue=df['DX'],
                        palette="deep", ax=ax[1], s=100, edgecolor='w')
        ax[1].set_title('Group')
        ax[1].set_xlabel(f'Dimension {x}')
        ax[1].set_ylabel(f'Dimension {y}')
        ax[1].legend(title='Group', loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)

        # Plot for Volume with a colorbar
        sns.scatterplot(x=embedding[:, x], y=embedding[:, y], hue=df['volume'],
                        palette=cmap_volume, ax=ax[2], s=100, edgecolor='w', legend=False)
        ax[2].set_title('Volume')
        ax[2].set_xlabel(f'Dimension {x}')
        ax[2].set_ylabel(f'Dimension {y}')
        norm = plt.Normalize(df['volume'].min(), df['volume'].max())
        sm = plt.cm.ScalarMappable(cmap=cmap_volume, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[2], orientation='vertical', label='Volume')

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(save_dir, f'{method}_embedding_dim{x}{y}.png')
        plt.savefig(plot_path)
        plt.close()

        # Log to wandb
        wandb.log({f"{method}_embedding_dim{x}{y}": WandbImage(plot_path)})


def plot_mkl_weights(mkl, feature_names, save_dir):
    """
    Plot the weights of each feature as determined by the MKL model.

    Parameters:
    mkl (MKL): Trained MKL model.
    feature_names (list): List of feature names.
    save_dir (str): Directory to save the plot.
    """
    weights = mkl.betas

    weights_dict = {features[i]: weights[i] for i in range(len(features))}

    # Plot the weights
    plt.figure(figsize=(20, 10))
    sns.barplot(weights_dict)
    # put the x labels in 45 angle
    plt.xticks(rotation=90)
    plt.show()
    # Save the plot
    plot_path = os.path.join(save_dir, 'mkl_weights.png')
    plt.savefig(plot_path)
    plt.close()

    # Log to wandb
    wandb.log({"mkl_weights": WandbImage(plot_path)})


def plot_kernel_heatmap(K, save_dir):
    """
    Plot a heatmap of the combined kernel matrix.

    Parameters:
    K (ndarray): Combined kernel matrix.
    save_dir (str): Directory to save the plot.
    """

    # Calculate the total plots needed (21 kernels in K plus 1 for W)
    total_plots = K.shape[0] + 1

    # Determine grid dimensions to be as square as possible
    cols = 4
    rows = math.ceil(total_plots/cols)
    print(rows)
    plt.figure(figsize=(30, 50))  # Adjust the size as needed
    # Plot each kernel from K
    for i in range(len(features)):

        plt.subplot(rows, cols, i + 1)
        sns.heatmap(K[i], cmap='viridis', square=True, cbar=False)
        # disable ticks for better visibility
        plt.xticks([])
        plt.yticks([])

        plt.title(f'Kernel {features[i]}')

    # Plot W separately
    plt.title('Kernel combined')

    # Save the plot
    plot_path = os.path.join(save_dir, 'kernel_heatmap.png')
    plt.savefig(plot_path)
    plt.close()

    # Log to wandb
    wandb.log({"kernel_heatmap": WandbImage(plot_path)})


def plot_dendrogram(Proj, save_dir):
    """
    Plot a dendrogram for hierarchical clustering of the projected data.

    Parameters:
    Proj (ndarray): Projected data from MKL.
    save_dir (str): Directory to save the plot.
    """
    linked = linkage(Proj, 'ward')

    labelList = range(1, Proj.shape[0]+1)

    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        labels=labelList,
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    # Save the plot
    plot_path = os.path.join(save_dir, 'dendrogram.png')
    plt.savefig(plot_path)
    plt.close()

    # Log to wandb
    wandb.log({"dendrogram": WandbImage(plot_path)})


def create_save_dir(base_dir, knn, alpha):
    """
    Create a directory to save results and plots for a given combination of hyperparameters.

    Parameters:
    base_dir (str): Base directory to create the save directory in.
    knn (int): Number of nearest neighbors.
    alpha (float): Alpha parameter for kernel combination.

    Returns:
    str: Path to the created directory.
    """
    save_dir = os.path.join(base_dir, f'knn_{knn}_alpha_{alpha}')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


if __name__ == "__main__":
    
    # Initialize Weights and Biases (wandb) logging
    wandb.init(project="MKL_fetal_analysis")

    file_path = '/home/gmarti/DATA/FETAL_GRAPHANALYSIS_DATA/CLEANED_DATA/cleaned_data_surf_combat.csv'
    base_dir = '/home/gmarti/CODE/fetal-graph-cortical-network/MKL_experiments'

    base_feature_old = ["thickness", "sulc"]
    base_features_new = ["LGI_mean", "dpf_mean", "mean_curvature_mean", "spangy_dom_band_mean"]
    brain_regions = ["Insula", "Occipital_lobe", "Frontal_lobe", "Parietal_lobe", "Temporal_lobe"]

    base_feature_old = [f"{feature}_{region}_mean" for region in brain_regions for feature in base_feature_old]

    #base_features_new = [f"{feature}_{region}_mean" for region in brain_regions for feature in base_features_new]

    features = [
        "volume",
        "volume_CSF",
        "volume_Cortical_gray_matter",
        "volume_White_matter",
        "volume_Ventricles",
        "volume_Cerebellum",
        "volume_Deep_Gray_Matter",
        "volume_Brainstem",
        "volume_Hippocampi_and_Amygdala",
        "mean_thickness",
        "gaussian_curvature_mean",
        "curvedness_index_mean",
        "folding_index_mean",
        "shape_index_mean",
        "B1", "B2", "B3", "B4", "B5", "B6",
        "cerebellar_supratentorial_ratio",
    ]

    features = base_features_new + base_feature_old +features
    
    # wandb log the features
    wandb.log({"features": features})
    print("Features logged to wandb")

    # Define hyperparameter ranges
    knn_values = [10, 20, 30, 40, 50, 100, 150]  # Example values
    alpha_values = [-1.0, -0.7, -0.5, -0.2]  # Example values

    

    # Initialize a dictionary to store results
    results = {}

    for knn in knn_values:
        for alpha in alpha_values:
            print(f"Starting iteration with knn={knn}, alpha={alpha}")
            
            try:
                # Create a directory to save plots and results
                save_dir = create_save_dir(base_dir, knn, alpha) 
                print(f"Save directory created: {save_dir}")
                
                # Data loading and preprocessing
                df, X_full, gest_weeks = load_and_preprocess_data(file_path, features, detrend=False, norm=False)
                print("Data loaded and preprocessed")
                
                # Kernel creation
                K, W, D = create_kernels(X_full, kernel_type='euclidean', alpha=alpha, knn=knn)
                print("Kernels created")
                
                plot_kernel_heatmap(K, save_dir)
                print("Kernel heatmap plotted")

                # MKL training
                mkl, Proj = train_mkl(K, W, D)
                print("MKL trained")
                
                # Clustering
                cluster_labels = perform_clustering(Proj, n_clusters=4)
                print("Clustering performed")
                
                # Evaluation metrics
                metrics = evaluate_clusters(df, cluster_labels, Proj, gest_weeks)
                print("Clusters evaluated")
                
                # Log metrics to wandb
                wandb.log({
                    'knn': knn,
                    'alpha': alpha,
                    **metrics
                })
                print("Metrics logged to wandb")
                
                # Store the results
                results[(knn, alpha)] = metrics
                
                # Plotting
                plot_embedding(Proj, df, save_dir, method='t-SNE')
                plot_embedding(Proj, df, save_dir, method='PCA')
                plot_mkl_weights(mkl, features, save_dir)
                plot_dendrogram(Proj, save_dir)
                print("All plots created")
            
                # Log additional information to wandb
                wandb.log({
                    'knn': knn,
                    'alpha': alpha,
                    **metrics,
                    'cluster_labels': wandb.Table(data=[[label] for label in cluster_labels], columns=["Cluster"])
                })
                print("Additional information logged to wandb")

                # Remove all variables to save memory
                del mkl, Proj, K, W, D, df, X_full, gest_weeks, cluster_labels, metrics
                print(f"Completed iteration with knn={knn}, alpha={alpha}")

            except Exception as e:
                print(f"Error during iteration with knn={knn}, alpha={alpha}: {str(e)}")
                continue

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.names = ['knn', 'alpha']
    results_df.reset_index(inplace=True)

    # Example: Find the configuration with the highest silhouette score
    best_config = results_df.loc[results_df['silhouette_score'].idxmax()]
    print("Best configuration based on silhouette score:")
    print(best_config)

    # Finish wandb run
    wandb.finish()
    print("Wandb run finished")
