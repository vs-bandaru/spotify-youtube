# Imported Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score

# Merge datasets
def data():
  print('Merge datasets')
  df1 = pd.read_csv('../csv_files/spotify_api_mood.csv')
  df2 = pd.read_csv('../csv_files/mood.csv')
  df2 = df2[:10000]
  df = pd.merge(df1, df2, on='uri')
  print('Datasets merged')
  print()
  return df

# Print data
def printdata(df):
  del df['Unnamed: 0']
  del df['uri']
  del df['analysis_url']
  del df['track_href']
  df.drop_duplicates(inplace=True, keep='first')
  print(df.head())
  print()
  print(df.tail())
  print()
  print(df.describe())
  print()
  print(df.info())
  print('Data printed')
  print()
  return

# Variable Identification

# Categorical Variables
# # 1. key
# # 2. mode
# # 3. time_signature
# # 4. labels ( 0: Sad, 1: Happy, 2: Energetic, 3: Calm)
# Continuous Variables:
# # 1. danceability
# # 2. energy
# # 3. loudness
# # 4. speechiness
# # 5. acousticness
# # 6. intrumentalness
# # 7. liveness
# # 8. valence
# # 9. tempo
# # 10. duration_ms

# Univariate Analysis
def univariateAnalysis(df):
  print('Univariate Analysis')
  plt.figure(figsize=(8, 6))
  sns.countplot(data=df, x=df['key'])
  plt.xticks(rotation=45)
  plt.title("Key of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.countplot(data=df, x=df['mode'])
  plt.title("Mode of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.countplot(data=df, x=df['time_signature'])
  plt.title("Time Signature of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.countplot(data=df, x=df['labels'])
  plt.title("Mood Labels of the songs - 0: Sad, 1: Happy, 2: Energetic, 3: Calm")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['danceability'], bins=10, kde=True)
  plt.title("Danceability of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['energy'], bins=10, kde=True)
  plt.title("Energy of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['loudness'], bins=20, kde=True)
  plt.title("Loudness of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['speechiness'], bins=10, kde=True)
  plt.title("Speechiness of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['acousticness'], bins=10, kde=True)
  plt.title("Acousticness of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['intrumentalness'], bins=10, kde=True)
  plt.title("Intrumentalness of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['liveness'], bins=20, kde=True)
  plt.title("Liveness of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['valence'], bins=20, kde=True)
  plt.title("Valence of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['tempo'], bins=20, kde=True)
  plt.title("Tempo of the songs")
  plt.show()
  print()

  plt.figure(figsize=(8, 6))
  sns.histplot(df['duration_ms'], bins=50, kde=True)
  plt.title("Duration of the songs")
  plt.show()
  print('Univariate Analysis Completed')
  print()
  return

# Bivariate Analysis
def bivariateAnalysis(df):
  print('Bivariate Analysis')
  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='key', hue='mode')
  plt.xlabel('Key')
  plt.ylabel('Count')
  plt.title('Key and Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='mode', hue='time_signature')
  plt.xlabel('Mode')
  plt.ylabel('Count')
  plt.title('Mode and Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='time_signature', hue='key')
  plt.xlabel('Time Signature')
  plt.ylabel('Count')
  plt.title('Time Signature and Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='key', hue='labels')
  plt.xlabel('Key')
  plt.ylabel('Count')
  plt.title('Key and Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='mode', hue='labels')
  plt.xlabel('Mode')
  plt.ylabel('Count')
  plt.title('Mode and Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.countplot(data=df, x='time_signature', hue='labels')
  plt.xlabel('Time Signature')
  plt.ylabel('Count')
  plt.title('Time Signature and Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='acousticness', y='intrumentalness', hue='key')
  plt.xlabel('Acousticness')
  plt.ylabel('Intrumentalness')
  plt.title('Acousticness and Intrumentalness based on Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='acousticness', y='intrumentalness', hue='mode')
  plt.xlabel('Acousticness')
  plt.ylabel('Intrumentalness')
  plt.title('Acousticness and Intrumentalness based on Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='acousticness', y='intrumentalness', hue='time_signature')
  plt.xlabel('Acousticness')
  plt.ylabel('Intrumentalness')
  plt.title('Acousticness and Intrumentalness based on Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='acousticness', y='intrumentalness', hue='labels')
  plt.xlabel('Acousticness')
  plt.ylabel('Intrumentalness')
  plt.title('Acousticness and Intrumentalness based on Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='energy', y='danceability', hue='key')
  plt.xlabel('Energy')
  plt.ylabel('Danceability')
  plt.title('Energy and Danceability based on Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='energy', y='danceability', hue='mode')
  plt.xlabel('Energy')
  plt.ylabel('Danceability')
  plt.title('Energy and Danceability based on Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='energy', y='danceability', hue='time_signature')
  plt.xlabel('Energy')
  plt.ylabel('Danceability')
  plt.title('Energy and Danceability based on Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='energy', y='danceability', hue='labels')
  plt.xlabel('Energy')
  plt.ylabel('Danceability')
  plt.title('Energy and Danceability based on Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='tempo', y='danceability', hue='key')
  plt.xlabel('Tempo')
  plt.ylabel('Danceability')
  plt.title('Tempo and Danceability based on Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='tempo', y='danceability', hue='mode')
  plt.xlabel('Tempo')
  plt.ylabel('Danceability')
  plt.title('Tempo and Danceability based on Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='tempo', y='danceability', hue='time_signature')
  plt.xlabel('Tempo')
  plt.ylabel('Danceability')
  plt.title('Tempo and Danceability based on Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='tempo', y='danceability', hue='labels')
  plt.xlabel('Tempo')
  plt.ylabel('Danceability')
  plt.title('Tempo and Danceability based on Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='speechiness', y='danceability', hue='key')
  plt.xlabel('Speechiness')
  plt.ylabel('Danceability')
  plt.title('Speechiness and Danceability based on Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='speechiness', y='danceability', hue='mode')
  plt.xlabel('Speechiness')
  plt.ylabel('Danceability')
  plt.title('Speechiness and Danceability based on Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='speechiness', y='danceability', hue='time_signature')
  plt.xlabel('Speechiness')
  plt.ylabel('Danceability')
  plt.title('Speechiness and Danceability based on Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='speechiness', y='danceability', hue='labels')
  plt.xlabel('Speechiness')
  plt.ylabel('Danceability')
  plt.title('Speechiness and Danceability based on Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='liveness', y='valence', hue='key')
  plt.xlabel('Liveness')
  plt.ylabel('Valence')
  plt.title('Liveness and Valence based on Key')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='liveness', y='valence', hue='mode')
  plt.xlabel('Liveness')
  plt.ylabel('Valence')
  plt.title('Liveness and Valence based on Mode')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='liveness', y='valence', hue='time_signature')
  plt.xlabel('Liveness')
  plt.ylabel('Valence')
  plt.title('Liveness and Valence based on Time Signature')
  plt.show()
  print()

  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='liveness', y='valence', hue='labels')
  plt.xlabel('Liveness')
  plt.ylabel('Valence')
  plt.title('Liveness and Valence based on Mood Labels - 0: Sad, 1: Happy, 2: Energetic, 3:Calm')
  plt.show()
  print('Bivariate Analysis Completed')
  print()
  return

# Missing values treatment
def missingValuesTreatment(df):
  print('Missing values')
  df.isnull()
  df.isnull().sum()
  df.info()
  print('Missing values identified')
  print()
  return

# Data field information for outliers
def describeData(df):
  print('Data description')
  print(df.describe())
  print('Data description completed')
  print()
  return

# Identifying Outliers
def statisticalOutlier(df):
  print('Outliers')
  print('danceability')
  IQR=df['danceability'].quantile(0.75)-df['danceability'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['danceability'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['danceability'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['danceability']>=Upper_OutlierLimit) | (df['danceability']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('energy')
  IQR=df['energy'].quantile(0.75)-df['energy'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['energy'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['energy'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['energy']>=Upper_OutlierLimit) | (df['energy']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('loudness')
  IQR=df['loudness'].quantile(0.75)-df['loudness'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['loudness'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['loudness'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['loudness']>=Upper_OutlierLimit) | (df['loudness']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('speechiness')
  IQR=df['speechiness'].quantile(0.75)-df['speechiness'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['speechiness'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['speechiness'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['speechiness']>=Upper_OutlierLimit) | (df['speechiness']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('acousticness')
  IQR=df['acousticness'].quantile(0.75)-df['acousticness'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['acousticness'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['acousticness'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['acousticness']>=Upper_OutlierLimit) | (df['acousticness']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('intrumentalness')
  IQR=df['intrumentalness'].quantile(0.75)-df['intrumentalness'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['intrumentalness'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['intrumentalness'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['intrumentalness']>=Upper_OutlierLimit) | (df['intrumentalness']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('liveness')
  IQR=df['liveness'].quantile(0.75)-df['liveness'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['liveness'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['liveness'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['liveness']>=Upper_OutlierLimit) | (df['liveness']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('valence')
  IQR=df['valence'].quantile(0.75)-df['valence'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['valence'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['valence'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['valence']>=Upper_OutlierLimit) | (df['valence']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('tempo')
  IQR=df['tempo'].quantile(0.75)-df['tempo'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['tempo'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['tempo'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['tempo']>=Upper_OutlierLimit) | (df['tempo']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print()

  print('duration_ms')
  IQR=df['duration_ms'].quantile(0.75)-df['duration_ms'].quantile(0.25)
  print(IQR)
  Upper_OutlierLimit = df['duration_ms'].quantile(0.75) + 1.5*IQR
  Lower_OutlierLimit = df['duration_ms'].quantile(0.25) - 1.5*IQR
  print('Upper bound')
  print(Upper_OutlierLimit)
  print('Lower bound')
  print(Lower_OutlierLimit)
  OutlierValues=pd.DataFrame()
  OutlierValues = df[(df['duration_ms']>=Upper_OutlierLimit) | (df['duration_ms']<=Lower_OutlierLimit)]
  print('Outliers')
  print(OutlierValues)
  print("Outliers identified")
  print()
  return

# Plots to detect outliers
def plots(df):
  print('Boxplots')
  sns.boxplot(y='danceability',data=df)
  plt.show()
  print()
  sns.boxplot(y='energy',data=df)
  plt.show()
  print()
  sns.boxplot(y='loudness',data=df)
  plt.show()
  print()
  sns.boxplot(y='speechiness',data=df)
  plt.show()
  print()
  sns.boxplot(y='acousticness',data=df)
  plt.show()
  print()
  sns.boxplot(y='intrumentalness',data=df)
  plt.show()
  print()
  sns.boxplot(y='liveness',data=df)
  plt.show()
  print()
  sns.boxplot(y='valence',data=df)
  plt.show()
  print()
  sns.boxplot(y='tempo',data=df)
  plt.show()
  print()
  sns.boxplot(y='duration_ms',data=df)
  plt.show()
  print()
  print('Boxplots plotted')
  print()
  return

# Correlation Analysis
def correlation(df):
  print('Correlation')
  corr = df.describe().loc[['mean', '50%', 'std']]
  print(corr)
  print('Correlation found')
  print()
  return

# PCA Analysis
def pca(df):
  print('PCA')
  cols = ["danceability", "energy", "loudness", "speechiness", "liveness", "valence", "tempo", "duration_ms"]
  data = df[cols].dropna()
  scaler = StandardScaler()
  scale = scaler.fit_transform(data)
  pca = PCA(n_components=2)
  pcares = pca.fit_transform(scale)
  pcadf = pd.DataFrame(data=pcares, columns=['PC1', 'PC2'])
  pcadf['labels'] = df.loc[data.index, 'labels'].values
  plt.figure(figsize=(10, 8))
  sns.scatterplot(x='PC1', y='PC2', data=pcadf, hue='labels', palette='bright', s=50)
  plt.title('PCA of Spotify Song Attributes')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.show()
  print('PCA completed')
  return df

def clusters(df):
  print('Clusters')
  data = df
  data = data[['danceability', 'energy', 'intrumentalness', 'liveness', 'tempo']]
  n_clusters = 5
  # Ward hierarchical clustering
  ward_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
  ward_labels = ward_model.fit_predict(data)
  ward_calinski_harabaz = calinski_harabasz_score(data, ward_labels)
  plt.scatter(data['danceability'], data['energy'], c=ward_labels)
  plt.title('Ward Clustering')
  plt.xlabel('Danceability')
  plt.ylabel('Energy')
  plt.show()
  print()
  # K-means clustering
  kmeans_model = KMeans(n_clusters=n_clusters)
  kmeans_labels = kmeans_model.fit_predict(data)
  kmeans_calinski_harabaz = calinski_harabasz_score(data, kmeans_labels)
  plt.scatter(data['danceability'], data['energy'], c=kmeans_labels)
  plt.title('K-means Clustering')
  plt.xlabel('Danceability')
  plt.ylabel('Energy')
  plt.show()
  print()
  # DBSCAN clustering
  dbscan_model = DBSCAN(eps=0.5, min_samples=5)
  dbscan_labels = dbscan_model.fit_predict(data)
  dbscan_calinski_harabaz = calinski_harabasz_score(data, dbscan_labels)
  plt.scatter(data['danceability'], data['energy'], c=dbscan_labels)
  plt.title('DBSCAN Clustering')
  plt.xlabel('Danceability')
  plt.ylabel('Energy')
  plt.show()
  print()
  # GMM clustering
  gmm_model = GaussianMixture(n_components=n_clusters)
  gmm_labels = gmm_model.fit(data).predict(data)
  gmm_calinski_harabaz = calinski_harabasz_score(data, gmm_labels)
  plt.scatter(data['danceability'], data['energy'], c=gmm_labels)
  plt.title('GMM Clustering')
  plt.xlabel('Danceability')
  plt.ylabel('Energy')
  plt.show()
  print()
  # Calinski-Harabasz scores for each method
  print(f'Ward Score = {ward_calinski_harabaz}')
  print(f'K-means Score = {kmeans_calinski_harabaz}')
  print(f'DBSCAN Score = {dbscan_calinski_harabaz}')
  print(f'GMM Score = {gmm_calinski_harabaz}')
  print('Clusters completed')
  print()
  return

def main():
  print('Main')
  df = data()
  printdata(df)
  univariateAnalysis(df)
  bivariateAnalysis(df)
  missingValuesTreatment(df)
  describeData(df)
  statisticalOutlier(df)
  plots(df)
  correlation(df)
  df1 = pca(df)
  clusters(df1)
  print('Process complete')
  print()
  return

if __name__ == '__main__':
  print('Calling main')
  main()