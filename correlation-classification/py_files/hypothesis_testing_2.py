import pandas as pd
import urllib.parse as p

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import statsmodels.api as sm

#####################################################################################
#  Hypothesis Testing
#  H0: The song's popularity on spotify is not related to its popularity on YouTube
#  H1: The song's popularity on spotify is related to its popularity on YouTube
#####################################################################################

# Read files
df_spotify_youtube = pd.read_csv('../csv_files/spotify_youtube.csv', sep=',')
df_youtube = pd.read_csv('../csv_files/youtube_api_data_result.csv', sep=',')
df_spotify = pd.read_csv('../csv_files/spotify_api_songs_popularity_analysis_result.csv', sep=',')


# Merge [track_popularity] in spotify_api_songs_popularity_analysis_result.csv
# and [comment_count],[likeness_count], [view_count] in youtube_api_data_result.csv
# by track_id and Url_youtube in spotify_youtube.csv
def merge_file():
    tracks_id = []
    videos_id = []
    # Get track_id in spotify_youtube.csv
    for uri in df_spotify_youtube['Uri']:
        temp = uri[14:]
        tracks_id.append(temp)

    # Get youtube_id in spotify_youtube.csv
    for url in df_spotify_youtube['Url_youtube']:
        if pd.isna(url):
            videos_id.append(None)
        else:
            parsed_url = p.urlparse(str(url))
            video_id = p.parse_qs(parsed_url.query).get("v")
            if video_id:
                videos_id.append(video_id[0])

    # link data
    df_id = pd.DataFrame(
        {
            'track_id': tracks_id,
            'video_id': videos_id
        }
    )

    # Merge spotify_api_songs_popularity_analysis_result.csv and link data by track_id
    merged_df1 = pd.merge(df_id, df_spotify, on='track_id', how='inner')
    # Merge youtube_api_data_result.csv and result of first merge  by video_id
    merged_df2 = pd.merge(merged_df1, df_youtube, on='video_id', how='inner')
    # Remain fields which will be used in hypothesis testing
    merged_df2 = merged_df2[
        ['track_id', 'video_id', 'track_popularity', 'like_count', 'view_count', 'like_bin', 'view_bin']]

    # Get the number of each song's positive comments
    comment_df = get_positive_comment()
    # Merge positive_comment_count into the dataset
    final_df = pd.merge(merged_df2, comment_df, on='video_id', how='inner')

    # Bin positive_comment_count
    positive_comment_data = []
    for index,row in final_df.iterrows():
        positive_comment_data.append(row['positive_comment_count'])
    data_series_positive_comment = pd.Series(positive_comment_data)
    # Set the number of bins
    num_bins = 5
    # Equal-frequency binning
    bins_positive_comment = pd.qcut(data_series_positive_comment, q=num_bins, duplicates='drop')
    # Get the actual number of bins
    actual_num_bins = len(bins_positive_comment.cat.categories)
    names_duration = list(range(actual_num_bins))
    # Equal-frequency binning again
    bins_positive_comment = pd.qcut(data_series_positive_comment, q=actual_num_bins, labels=names_duration,
                                    duplicates='drop')

    final_df['positive_bin'] = bins_positive_comment

    pd.set_option('display.max_columns', 500)
    print("The Final dataset")
    print(final_df[:10])
    print("===" * 20)
    return final_df


# Calculate the number of each song's positive comments
def get_positive_comment():
    # Read File
    df = pd.read_csv('../csv_files/youtube_api_comments_sentimental_analysis_result.csv', sep=',')

    # Group each song's sentimental label
    comment_df = df.groupby('video_id')['senti_label'].value_counts().unstack().fillna(0).reset_index()
    comment_df['total'] = comment_df[[-1, 0, 1]].sum(axis=1)
    # Calculate the proportions of positive, neural and negative in all labels separately
    for col in [-1, 0, 1]:
        comment_df[col] = comment_df[col] / comment_df['total']
    comment_df.drop(columns='total', inplace=True)
    comment_df['comment_count'] = df_youtube['comment_count']

    # Calculate the relative proportion of positive comments in all comments
    positive_comment_count = []
    for index, row in comment_df.iterrows():
        score = (-1) * row[-1] + 0 * row[0] + 1 * row[1]
        positive_comment_count.append(row['comment_count'] * score)
    comment_df['positive_comment_count'] = positive_comment_count

    final_df = pd.DataFrame({
        'video_id': comment_df['video_id'],
        'positive_comment_count': comment_df['positive_comment_count']
    })

    pd.set_option('display.max_columns', 500)
    print("Dataset with positive_comment_count added")
    print(comment_df[:10])
    print("===" * 20)
    return final_df


# Calculate the total popularity score by setting weights
def weight_set(data):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['positive_comment_count', 'like_count', 'view_count']])

    # Calculate the weights of each feature by linear regression
    model = LinearRegression()
    model.fit(scaled_features, data['track_popularity'])

    print("The weight of positive comments:", model.coef_[0])
    print("The weight of likenesses:", model.coef_[1])
    print("The weight of views:", model.coef_[2])
    print("===" * 20)

    # youtube_popularity_score = weight_comment_count * comment_count
    #                          + weight_likeness_count * likeness_count
    #                          + weight_view_count * view_count
    weighted_popularity = []
    for index, row in data.iterrows():
        temp_popularity = row['positive_comment_count'] * model.coef_[0] + row['like_count'] * model.coef_[1] + row['view_count'] * model.coef_[2]
        weighted_popularity.append(temp_popularity)

    data['weighted_popularity'] = weighted_popularity
    return data


# Use t-test
def hypothesis_t_test(data):
    t_stat, p_value = ttest_ind(data['track_popularity'], data['weighted_popularity'])

    print("The reuslt of t-test")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("We reject the null hypothesis. The means of the two groups are significantly different.")
        print("The song's popularity on spotify is related to its popularity on YouTube")
    else:
        print(
            "We fail to reject the null hypothesis. There is no significant difference between the means of the two groups.")
    print("===" * 20)


# Use linear regression
def hypothesis_regression(data):

    X = sm.add_constant(data['weighted_popularity'])

    model = sm.OLS(data['track_popularity'], X).fit()

    # The result of linear regression
    print("The result of linear regression")
    print(model.summary())
    print("===" * 20)


# Use classification
def hypothesis_classification(data):
    # bin track_popularity
    track_popularity_data = []
    for index, row in data.iterrows():
        track_popularity_data.append(row['track_popularity'])
    data_series_track_popularity = pd.Series(track_popularity_data)
    num_bins = 3
    num_duration = [0, 1, 2]
    bins_track_popularity = pd.qcut(data_series_track_popularity, q=num_bins, labels=num_duration)
    data['track_popularity_bin'] = bins_track_popularity

    # print(data['view_bin'].isnull().sum())
    data.dropna(subset=['like_bin', 'view_bin'], inplace=True)

    # Youtube features
    X = data[['positive_bin', 'like_bin', 'view_bin']]
    y = data['track_popularity_bin']

    # Divide data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    # Use Random Forest
    clf = RandomForestClassifier(n_estimators=10, random_state=42, bootstrap=True, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
    clf.fit(X_train, y_train)

    # Get prediction
    y_pred = clf.predict(X_test)

    # The result of classification
    print("The result of classification")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Features influence
    importances = clf.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f"Feature: {feature}, Importance: {importance}")


def main():
    df = merge_file()
    data = weight_set(df)
    hypothesis_t_test(data)
    hypothesis_regression(data)
    hypothesis_classification(data)


if __name__ == '__main__':
    main()
