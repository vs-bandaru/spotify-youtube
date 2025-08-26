# import library
import pandas as pd
import numpy as np

# T test
from scipy.stats import ttest_rel

# Data visulization
import matplotlib.pyplot as plt

def main():

    #####################
    # Hypothesis Testing 1
    #####################

    # import spotify_api_songs_popularity_analysis_result.csv data frame
    df_songs = pd.read_csv('../csv_files/spotify_api_songs_popularity_analysis_result.csv')

    # See spotify_api_songs_popularity_analysis_result data frame
    print('Print spotify_api_songs_popularity_analysis_result.csv dataset:')
    print(df_songs)

    # See the distribution of 'valence' attribute with histogram
    valence_data = df_songs['valence']

    # draw histogram
    plt.hist(valence_data, bins=20, edgecolor='k')
    plt.xlabel('Valence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Valence')
    plt.show()

    # Count the number of data whose 'value' is greater and smaller than 0.5.
    valence_lt = df_songs[df_songs['valence'] < 0.5].shape[0]
    valence_gt = df_songs[df_songs['valence'] > 0.5].shape[0]

    print(f'Number of data with valence < 0.5：{valence_lt}')
    print(f'Number of data with Valence > 0.5：{valence_gt}')

    # Create datasets with 'value' less than 0.5 and greater than 0.5
    df_valence_lt = df_songs[df_songs['valence'] < 0.5]
    df_valence_gt = df_songs[df_songs['valence'] > 0.5]

    # 3982 samples from df_valence_lt and df_valence_gt respectively
    df_valence_lt_sample = df_valence_lt.sample(n=3982, random_state=42)
    df_valence_gt_sample = df_valence_gt.sample(n=3982, random_state=42)

    # Paired t-test
    # Run a two sample t-test to compare the two samples
    print('\nCalculate the statistics value and p value for hypothesis:')
    tstat, p = ttest_rel(df_valence_lt_sample['track_popularity'], df_valence_gt_sample['track_popularity'])

    # Display results
    print('Statistics=%.3f, p=%.3f' % (tstat, p))



if '__main__' == __name__:
    main()
