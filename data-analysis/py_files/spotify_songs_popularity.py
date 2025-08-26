
# Import relevants libraries

# Data manipulation
import pandas as pd
import numpy as np
# Data cleaning
from scipy import stats
# Data visulization
import matplotlib.pyplot as plt
import seaborn as sns


def data_analysis():
    ###########################
    ##### 1. Data preprocessing
    ###########################

    # Read in data spotify_api_songs and spotify_api_popularity directly into pandas
    df_songs = pd.read_csv('../csv_files/spotify_api_songs.csv', sep=',', encoding='latin1')
    df_pop = pd.read_csv('../csv_files/spotify_api_popularity.csv', sep=',', encoding='latin1')

    #  Sporify_api_songs Data Attributes Catalog
    # The data attributes catalog can be found here (https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features).
    # |    Attributes Name     | Descripcion                                                                                                                                                                                                         |
    # |    acousticness        | A confidence measure from **0.0 to 1.0** of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
    # |    danceability        | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. **A value of 0.0 is least danceable and 1.0 is most danceable.**
    # |    duration_ms         | The duration of the track in milliseconds.
    # |    energy              | Energy is a measure from **0.0 to 1.0** and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
    # |    instrumentalness    | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
    # |    Key                 | The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. **Range: -1 - 11**.
    # |    liveness            | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above **0.8** provides strong likelihood that the track is live.
    # |    loudness            | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between **-60 and 0 db**.
    # |    mode                | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. **Major is represented by 1 and minor is 0**.
    # |    speechiness         | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above **0.66** describe tracks that are probably made entirely of spoken words. Values between **0.33 and 0.66** describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values **below 0.33** most likely represent music and other non-speech-like tracks.
    # |    tempo               | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
    # |    valence             | A measure from **0.0 to 1.0** describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
    # |    time_signature      | An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from **3 to 7** indicating time signatures of "3/4", to "7/4".
    # |    uri                 |  Uniform Resource Identifier (URI). Each Spotify resource, such as a song, album, artist, etc., has a unique URI that can be used to locate that resource on the Spotify platform.
    # |    analysis_url        | A URL to access the full audio analysis of this track. An access token is required to access this data.
    # |    track_href          | A link to the Web API endpoint providing full details of the track.

    # Sporify_api_pop Data Attributes Catalog
    # | Attributes Name        | Descripcion                                                                                                                                                                                                         |
    # | track_popularity       | The popularity of the track. **Range: 0-100**.
    # | artist_popularity      | The popularity of the artist of the track. **Range: 0-100**.

    # 1.1 Data Frame spotify_api_songs and spotify_api_popularity format
    print("\n\nSpotify_api_songs and Spotify_api_popularity Data Frame format")
    print(df_songs)
    print(df_pop)

    # 1.2 Data Frame spotify_api_songs and spotify_api_popularity first 5 rows
    # print the first 5 rows of spotify_api_songs data frame
    print("\n\nSpotify_api_songs and Spotify_api_popularityData Frame first 5 rows")
    print(df_songs[:5])
    # print the first 5 rows of spotify_api_pop data frame
    print(df_pop[:5])

    # 1.3 Data Frame spotify_api_songs 'danceability' column
    # print the 'danceability' column of spotify_api_songs data frame
    print("\n\nSpotify_api_songs Attirbutes - danceability")
    df_songs['danceability']

    # 1.4 Data Frame spotify_api_popularity first 5 rows of 'track_popularity' column
    # print the first 5 rows of the 'track_popuplarity' column of spotify_api_pop data frame
    print("\n\nSpotify_api_popularity Attributes - track_populairty- first 5 rows")
    print(df_pop['track_popularity'][:5])

    # 1.5 See the count of each value for track_popularity and artist_popularity in Data Frame sporify_api_popularity

    # See the count of each value for track_popularity and artist_popularity in Data Frame sporify_api_popularity
    print("\n\nThe count of each value for track_popularity and artist_popularity in Data Frame sporify_api_popularity")
    track_pop_counts = df_pop['track_popularity'].value_counts()
    artist_pop_counts = df_pop['artist_popularity'].value_counts()

    print(track_pop_counts)
    print(artist_pop_counts)

    # 1.6 Count all the rows with the most (>90) and the least (<10) track and artist popularity from the list

    # Count all the rows with the most (>90) and the least (<10) track and artist popularity from the list
    most = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    least = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    track_row_with_most = df_pop['track_popularity'].isin(most)
    artist_row_with_most = df_pop['artist_popularity'].isin(most)
    track_row_with_least = df_pop['track_popularity'].isin(least)
    artist_row_with_least = df_pop['artist_popularity'].isin(least)

    count_track_most = track_row_with_most.sum()
    count_track_least = track_row_with_least.sum()
    count_artist_most = artist_row_with_most.sum()
    count_artist_least = artist_row_with_least.sum()

    print(f'Count of the most popular tracks: {count_track_most}')
    print(f'Count of the most popular artists: {count_artist_most}')
    print(f'Count of the least popular tracks: {count_track_least}')
    print(f'Count of the least popular artists: {count_artist_least}')

    # 1.7 Unique time signature list

    # Unique time_signature list
    time_sig_List = pd.unique(df_songs["time_signature"])
    print(f'Unique time signature List: {time_sig_List}')

    # 1.8 See the max, min values and their differences of the columns 'tempo' and 'duration_ms' from spotify_api_songs data frame

    # See the max, min values and their differences of the columns 'tempo' and 'duration_ms'
    # from spotify_api_songs data frame
    tempo_min = df_songs['tempo'].min()
    tempo_max = df_songs['tempo'].max()
    tempo_dif = tempo_max - tempo_min

    duration_min = df_songs['duration_ms'].min()
    duration_max = df_songs['duration_ms'].max()
    duration_dif = duration_max - duration_min

    print(f'the min value of tempo: {tempo_min}')
    print(f'the max value of tempo: {tempo_max}')
    print(f'the difference of the max and min value of tempo: {tempo_dif}')
    print()
    print(f'the min value of duration_ms: {duration_min}')
    print(f'the max value of duration_ms: {duration_max}')
    print(f'the difference of the max and min value of duration_ms: {duration_dif}')

    # 1.9 Quick view of data frame spotify_api_songs and spotify_api_popularity

    # Quick view of data frame spotify_api_songs and spotify_api_popularity
    print(f'Number of rows in spotify_api_songs data frame: {df_songs.shape[0]}')  # number of records
    print(f'Number of rows in spotify_api_popularity data frame: {df_pop.shape[0]}')
    print(f'Number of columns in spotify_api_songs data frame: {df_songs.shape[1]}')  # number of attributes
    print(f'Number of columns in spotify_api_popularity data frame: {df_pop.shape[1]}')
    print(f'Number of values in spotify_api_songs data frame: {df_songs.count().sum()}')  # number of values
    print(f'Number of values in spotify_api_popularity data frame: {df_songs.count().sum()}')

    songs_columns = ', '.join(df_songs.columns.tolist())
    pop_columns = ', '.join(df_pop.columns.tolist())
    print()
    print(f'Columns in spotify_api_songs data frame: {songs_columns}')  # name of all attributes
    print()
    print(f'Column in spotify_api_popularity data frame: {pop_columns}')

    ###########################
    ##### 2. Clean & Organize the data
    ###########################

    # 2.1 Drop the columns from spotify_api_songs
    # * Unnamed: 0
    # * uri
    # * analysis_url
    # * track_href

    # Drop 4 columns from the spotify_api_songs data frame
    del df_songs['Unnamed: 0']
    del df_songs['uri']
    del df_songs['analysis_url']
    del df_songs['track_href']

    print("\n\nspotify_api_songs data frame after dropping 4 columns")
    print(df_songs)

    # 2.2 Handle duplicate

    # Drop duplicates in spotify_api_songs and spotify_api_popularity data frame by track id
    df_songs = df_songs.drop_duplicates(subset="tracks_id")
    df_pop = df_pop.drop_duplicates(subset="track_id")

    print("\n\nspotify_api_songs and spotify_api_popularity data frames after handling duplicate")
    print(df_songs)
    print(df_pop)

    # 2.3 Integrating df_songs and df_pop based on track_id

    # Change the name of 'tracks_id' into 'track_id' for spotify_api_songs data frame
    df_songs.rename(columns={'tracks_id': 'track_id'}, inplace=True)

    # Merge df_songs and df_pop into data frame df, which is called spotify_api_songs_pop data frame
    df = pd.merge(df_songs, df_pop, on='track_id', how='inner')

    # Print data frame spotify_api_songs_pop format
    print(
        "\n\nspotify_api_songs_popularity data frame after merging spotify_api_songs and spotify_api_popularity dataset")
    print(df)

    # 2.3 Handle the missing values

    # Check the spotify_api_songs_pop data frame for missing values
    print(f'Number of missing values in spotify_api_songs_pop data frame: {sum(df.isna().sum())}')

    # Column Wise missing values
    print("\n\nColumn Wise missing values:")
    print(df.isna().sum().sort_values(ascending=False))

    # 2.4 Convert the duration from ms to min

    # Convert the duration from ms to min
    def convert_miliseconds(miliseconds, conversion='minutes'):
        if conversion == 'minutes':
            return miliseconds / 60000
        elif conversion == 'seconds':
            return miliseconds / 1000

    df['duration_min'] = df['duration_ms'].apply(convert_miliseconds, args=('minutes',))
    df = df.drop(columns='duration_ms')

    print("\n\nConverting duration_ms to duration min")
    print(df)

    ###########################
    ##### 3. Basic Statistical Analysis and data cleaning
    ###########################

    # 3.1 View the mean, median and standard deviation

    # View the mean, median and standard deviation of continuous variables in spotify_api_songs_pop data frame
    columns_to_drop = ['key', 'mode', 'time_signature', 'track_id']
    df_filtered = df.drop(columns=columns_to_drop)
    print(
        "\n\nView the mean, median and standard deviation of continuous variables in spotify_api_songs_pop data frame:")
    print(df_filtered.describe().loc[['mean', '50%', 'std']])

    # 3.2 Handle noise outliers

    # Draw boxplots for features in spotify_api_songs_pop data frame
    columns_to_drop = ['track_id']
    df_filtered1 = df.drop(columns=columns_to_drop)

    plt.figure(figsize=(16, 10))

    for i in range(len(df_filtered1.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_filtered1[df_filtered1.columns[i]])
        plt.title(df_filtered1.columns[i])

    plt.tight_layout()
    plt.show()

    # Remove outliers with Z-scores

    # Calculate the Z-score to detect the outliers
    columns_to_drop = ['track_id']
    df_filtered1 = df.drop(columns=columns_to_drop)

    attributes = df_filtered1.columns.to_list()
    columns_to_detect = df_filtered1.columns.to_list()
    z = np.abs(stats.zscore(df[columns_to_detect]))

    print("\n\nZ-score table:")
    print(z)

    # Specify a threshold, which means that the outlier will receive a Z-score of threshold value
    threshold1 = 3

    outliers_indices_1 = np.where(z > threshold1)

    df_cleaned_1 = df.drop(df.index[outliers_indices_1[0]])

    print(f'Shape of data frame with outliers removed when threshold = 3: {df_cleaned_1.shape}')
    print()

    # Draw boxplots for features in the data frame with outliers removed
    columns_to_drop = ['track_id']
    df_cleaned_filtered_1 = df_cleaned_1.drop(columns=columns_to_drop)

    plt.figure(figsize=(16, 10))

    for i in range(len(df_cleaned_filtered_1.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_cleaned_filtered_1[df_cleaned_filtered_1.columns[i]])
        plt.title(df_cleaned_filtered_1.columns[i])

    plt.tight_layout()
    plt.show()

    # Specify threshold = 2.5
    threshold2 = 2.5

    outliers_indices_2 = np.where(z > threshold2)

    df_cleaned_2 = df.drop(df.index[outliers_indices_2[0]])

    print(f'Shape of data frame with outliers removed when threshold = 2.5: {df_cleaned_2.shape}')
    print()

    # Draw boxplots for features in the data frame with outliers removed
    df_cleaned_filtered_2 = df_cleaned_2.drop(columns=columns_to_drop)

    plt.figure(figsize=(16, 10))

    for i in range(len(df_cleaned_filtered_2.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_cleaned_filtered_2[df_cleaned_filtered_2.columns[i]])
        plt.title(df_cleaned_filtered_2.columns[i])

    plt.tight_layout()
    plt.show()

    # Specify threshold = 2
    threshold3 = 2

    outliers_indices_3 = np.where(z > threshold3)

    df_cleaned_3 = df.drop(df.index[outliers_indices_3[0]])

    print(f'Shape of data frame with outliers removed when threshold = 2: {df_cleaned_3.shape}')
    print()

    # Draw boxplots for features in the data frame with outliers removed
    df_cleaned_filtered_3 = df_cleaned_3.drop(columns=columns_to_drop)

    plt.figure(figsize=(16, 10))

    for i in range(len(df_cleaned_filtered_3.columns)):
        plt.subplot(3, 5, i + 1)
        sns.boxplot(x=df_cleaned_filtered_3[df_cleaned_filtered_3.columns[i]])
        plt.title(df_cleaned_filtered_3.columns[i])

    plt.tight_layout()
    plt.show()

    # 3.3 Bins

    # Bins for track duration: short, medium-length and long
    # * 0: 3-minutes songs (short)
    # * 1: 3-5-minutes songs (medium-length)
    # * 2: Songs longer than 5 minutes (long)

    # see the max value of duration_min
    max_duration = df['duration_min'].max()

    print("\n\nMax value for duration_min")
    print(max_duration)

    # name list for duration
    names_duration = [0, 1, 2]

    bins_duration = [0, 3, 5, df["duration_min"].max()]
    df['duration_group'] = pd.cut(df['duration_min'], bins_duration, labels=names_duration)

    print("\n\nNew variable - duration_group")
    print(df['duration_group'])

    # Bins for speechiness
    # * 0: Tracks that are probably made entirely of spoken words.
    # * 1: Tracks that may contain both music and speech, either in sections or layered, including such cases as rap music.
    # * 2: Music and other non-speech-like tracks.

    # name list for speechniess
    bins_speechiness = [-1, 0.33, 0.66, df['speechiness'].max()]
    bins_speechiness = [float(val) for val in bins_speechiness]
    names_speechiness = [0, 1, 2]

    df['speechiness_group'] = pd.cut(df['speechiness'], bins_speechiness, labels=names_speechiness)
    print("\n\nNew variable - speechiness_group")
    print(df['speechiness_group'])

    # **Bins for liveness**
    # * 0: Track is non-live
    # * 1: Track is live

    # name list for speechniess
    bins_liveness = [-1, 0.8, df['liveness'].max()]
    bins_liveness = [float(val) for val in bins_liveness]
    names_liveness = [0, 1]

    df['liveness_group'] = pd.cut(df['liveness'], bins_liveness, labels=names_liveness)

    print("\n\nNew variable - liveness_group")
    print(df['liveness_group'])

    # Save the spotify_api_songs_pop data frame to local path
    df.to_csv("../csv_files/spotify_api_songs_popularity_analysis_result.csv", index=False)

    ###########################
    ##### 4. Histograms and Correlation
    ###########################

    # 4.1 Histogram of frequency distribution for all attributes

    # Draw frequency distribution histogram for features in spotify_api_songs_pop data frame
    columns_to_drop = ['track_id']
    df_filtered2 = df.drop(columns=columns_to_drop)

    attributes = df_filtered2.columns.to_list()

    plt.figure(figsize=(16, 12))
    colours = ['#a7c957', '#d62728', '#386641', '#99621e', '#ff7f0e', '#bc4749',
               '#dd7373', '#2d3047', '#ff9da7', '#59a14f', '#af7aa1', '#bab0ac', '#edc949'
        , '#4e79a7', '#9c755f', '#f28e2b', '#17becf', '#8c271e', '#6a994e']

    for i, attribute in enumerate(attributes):
        plt.subplot(6, 3, i + 1)
        sns.histplot(df[attribute], bins=30, kde=True, color=colours[i])
        plt.title(f'Frequency Distribution of {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')

    plt.tight_layout()

    plt.show()


def main():
    data_analysis()


if __name__ == '__main__':
    main()
