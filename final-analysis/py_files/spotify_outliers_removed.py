import pandas as pd

# Data cleaning
from scipy import stats
import numpy as np

# Data visulization
import matplotlib.pyplot as plt
import seaborn as sns

def remove_outliers(df_mood,df_song):

    # Calculate the Z-score to detect the outliers
    columns_to_drop_1 = ['track_id', 'labels']
    columns_to_drop_2 = ['track_id', 'track_popularity', 'artist_popularity']
    df_mood_cleaned = df_mood.drop(columns=columns_to_drop_1)
    df_song_cleaned = df_song.drop(columns=columns_to_drop_2)

    # Check the boxplot for each attribute of the spotify_api_mood and spotify_api_song datasets
    # Sets the size of the graph
    plt.figure(figsize=(16, 10))

    for i in range(len(df_mood_cleaned.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_mood_cleaned[df_mood_cleaned.columns[i]])

    plt.tight_layout()
    plt.show()

    print('The box plot of spotify_api_song dataset')
    plt.figure(figsize=(16, 10))
    for i in range(len(df_song_cleaned.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_song_cleaned[df_song_cleaned.columns[i]])

    plt.tight_layout()
    plt.show()

    z_mood = np.abs(stats.zscore(df_mood_cleaned))
    z_song = np.abs(stats.zscore(df_song_cleaned))

    # See the z-score of each cell in spotify_api_mood and spotify_api_song datasets
    print("See the z-score of each sell in spotify_api_mood dataset:")
    print(z_mood)
    print()
    print("See the z-score of each sell in spotify_api_song dataset")
    print(z_song)

    # Get features in spotify_api_mood and spotify_api_song datasets
    mood_attributes = df_mood_cleaned.columns.to_list()
    song_attributes = df_song_cleaned.columns.to_list()

    # Specify a threshold, which means that the outlier will receive a Z-score of threshold value
    threshold1 = 3

    outliers_indices_mood = np.where(z_mood > threshold1)
    outliers_indices_song = np.where(z_song > threshold1)

    df_mood_cleaned_1 = df_mood.drop(df_mood.index[outliers_indices_mood[0]])
    df_song_cleaned_1 = df_song.drop(df_song.index[outliers_indices_song[0]])

    print(f'Shape of mood data frame with outliers removed when threshold = 3: {df_mood_cleaned_1.shape}')
    print(f'Shape of song data frame with outliers removed when threshold = 3: {df_song_cleaned_1.shape}')

    # Draw boxplots for features in the data frame with outliers removed
    columns_to_drop_1 = ['track_id', 'labels']
    columns_to_drop_2 = ['track_id', 'track_popularity', 'artist_popularity']
    df_mood_cleaned_2 = df_mood_cleaned_1.drop(columns=columns_to_drop_1)
    df_song_cleaned_2 = df_song_cleaned_1.drop(columns=columns_to_drop_2)

    print()
    print('The box plot of df_mood after removing outliers using Z-score when threshold = 3')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_mood_cleaned_2.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_mood_cleaned_2[df_mood_cleaned_2.columns[i]])

    plt.tight_layout()
    plt.show()

    print()
    print('The box plot of df_song after removing outliers using Z-score when threshold = 3')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_song_cleaned_2.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_song_cleaned_2[df_song_cleaned_2.columns[i]])

    plt.tight_layout()
    plt.show()

    # threshold = 2.5
    threshold2 = 2.5

    outliers_indices_mood_2 = np.where(z_mood > threshold2)
    outliers_indices_song_2 = np.where(z_song > threshold2)

    df_mood_cleaned_3 = df_mood.drop(df_mood.index[outliers_indices_mood_2[0]])
    df_song_cleaned_3 = df_song.drop(df_song.index[outliers_indices_song_2[0]])

    print(f'Shape of mood data frame with outliers removed when threshold = 2.5: {df_mood_cleaned_3.shape}')
    print(f'Shape of song data frame with outliers removed when threshold = 2.5: {df_song_cleaned_3.shape}')

    # Draw boxplots for features in the data frame with outliers removed
    columns_to_drop_1 = ['track_id', 'labels']
    columns_to_drop_2 = ['track_id', 'track_popularity', 'artist_popularity']
    df_mood_cleaned_4 = df_mood_cleaned_3.drop(columns=columns_to_drop_1)
    df_song_cleaned_4 = df_song_cleaned_3.drop(columns=columns_to_drop_2)

    print()
    print('The box plot of df_mood after removing outliers using Z-score when threshold = 2.5')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_mood_cleaned_4.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_mood_cleaned_4[df_mood_cleaned_4.columns[i]])

    plt.tight_layout()
    plt.show()

    print()
    print('The box plot of df_song after removing outliers using Z-score when threshold = 2.5')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_song_cleaned_4.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_song_cleaned_4[df_song_cleaned_4.columns[i]])

    plt.tight_layout()
    plt.show()

    # threshold = 2
    threshold3 = 2

    outliers_indices_mood_3 = np.where(z_mood > threshold3)
    outliers_indices_song_3 = np.where(z_song > threshold3)

    df_mood_cleaned_5 = df_mood.drop(df_mood.index[outliers_indices_mood_3[0]])
    df_song_cleaned_5 = df_song.drop(df_song.index[outliers_indices_song_3[0]])

    print(f'Shape of mood data frame with outliers removed when threshold = 2: {df_mood_cleaned_5.shape}')
    print(f'Shape of song data frame with outliers removed when threshold = 2: {df_song_cleaned_5.shape}')

    # Draw boxplots for features in the data frame with outliers removed
    columns_to_drop_1 = ['track_id', 'labels']
    columns_to_drop_2 = ['track_id', 'track_popularity', 'artist_popularity']
    df_mood_cleaned_6 = df_mood_cleaned_5.drop(columns=columns_to_drop_1)
    df_song_cleaned_6 = df_song_cleaned_5.drop(columns=columns_to_drop_2)

    print()
    print('The box plot of df_mood after removing outliers using Z-score when threshold = 2')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_mood_cleaned_6.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_mood_cleaned_6[df_mood_cleaned_6.columns[i]])

    plt.tight_layout()
    plt.show()

    print()
    print('The box plot of df_song after removing outliers using Z-score when threshold = 2')
    plt.figure(figsize=(16, 10))

    for i in range(len(df_song_cleaned_6.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_song_cleaned_6[df_song_cleaned_6.columns[i]])

    plt.tight_layout()
    plt.show()

    # Save the df_mood and df_songs datasets after removing outliers
    df_mood_cleaned_5.to_csv('../csv_files/spotify_api_mood_cleaned.csv', index=False)
    df_song_cleaned_5.to_csv('../csv_files/spotify_api_song_cleaned.csv', index=False)


def main():
    # import spotify_api_song and spotify_api_mood datasets
    df_song = pd.read_csv('../csv_files/spotify_api_song.csv')
    df_mood = pd.read_csv('../csv_files/spotify_api_mood.csv')

    # remove outliers for spotify_api_song and spotify_api_mood datasets
    remove_outliers(df_mood,df_song)


if '__main__' == __name__:
    main()

