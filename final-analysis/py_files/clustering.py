# Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score


# Function to perform PCA on the data
def pca(data):
    data = data.drop('track_id', axis=1)
    scaler = StandardScaler()
    scale = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    df = pca.fit_transform(scale)
    return df


# Function that performs K-means, Ward, DBScan, GMM Clustering on the data
def clusters(df):
    warnings.simplefilter('ignore')
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    print()
    # K-means Clustering
    print('K-means Clustering')
    # Initialize the class object
    kmeans = KMeans(n_clusters=3)
    # Predict the labels of clusters.
    label = kmeans.fit_predict(df)
    # Plotting the results:
    plt.figure(figsize=(10, 6))
    for i in np.unique(label):
        plt.scatter(df[label == i, 0], df[label == i, 1], label=f'Cluster {i}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering')
    plt.legend()
    print()
    # Calculate Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(df, label)
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")
    # Calculate Silhouette Score
    silhouette = silhouette_score(df, label)
    print(f"Silhouette Score: {silhouette}")
    print()

    # Ward Clustering
    print('Ward Clustering')
    linkage_matrix = linkage(df, method='ward')
    # Dendrogram Visualization
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    dendrogram(linkage_matrix, p=5, truncate_mode='level', color_threshold=20, above_threshold_color='grey')
    plt.title('Ward Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()
    print()
    # Calculate Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(df, fcluster(linkage_matrix, 5, criterion='maxclust'))
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")
    # Calculate Silhouette Score
    silhouette = silhouette_score(df, fcluster(linkage_matrix, 5, criterion='maxclust'))
    print(f"Silhouette Score: {silhouette}")
    print()

    # DBSCAN Clustering
    print('DBScan Clustering')
    dbscan = DBSCAN(eps=0.5, min_samples=6)
    labels = dbscan.fit_predict(df)
    # Visualize the Clusters
    plt.figure(figsize=(10, 6))
    # Plot points that are not in any cluster (label = -1)
    plt.scatter(df[labels == -1, 0], df[labels == -1, 1], color='gray', label='Outliers')
    # Plot points in each cluster
    for cluster_label in np.unique(labels[labels != -1]):
        plt.scatter(df[labels == cluster_label, 0], df[labels == cluster_label, 1],
                    label=f'Cluster {cluster_label}')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    print()
    # Calculate Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(df, labels)
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")
    # Calculate Silhouette Score
    silhouette = silhouette_score(df, labels)
    print(f"Silhouette Score: {silhouette}")
    print()

    # GMM Clustering
    print('GMM Clustering')
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(df)
    # Visualize the Clusters
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')
    plt.title('GMM Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    print()
    # Calculate Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(df, labels)
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")
    # Calculate Silhouette Score
    silhouette = silhouette_score(df, labels)
    print(f"Silhouette Score: {silhouette}")
    print()
    return


# Main function
def main():
    data = pd.read_csv('../csv_files/spotify_api_mood_cleaned.csv')
    print("Data Imported")
    print("PCA")
    # PCA on the data
    df = pca(data)
    print("Clustering")
    # Clustering the data
    clusters(df)
    print("Completed")
    return


if __name__ == '__main__':
    main()