import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)

def remove_duplicate(df_youtube):
    df_youtube.drop_duplicates(subset=['video_id'], keep='first', inplace=True)
    print("Successfully remove duplicate data!")
    print("===" * 20)
    return df_youtube


def handle_comment_missing_value(df):
    fit_data = []
    x_data = []
    fit_video_id = []
    fit_comment_count = []
    fit_like_count = []
    fit_view_count = []
    x_video_id = []
    x_comment_count = []
    x_like_count = []
    x_view_count = []
    for index, row in df.iterrows():
        # Find the video whose comment_count is 0 and not 0
        if int(row["comment_count"]) != 0:
            fit_list = []
            fit_list.append(row["comment_count"])
            fit_list.append(row["like_count"])
            fit_list.append(row["view_count"])
            fit_data.append(fit_list)
            fit_video_id.append(row["video_id"])
            fit_comment_count.append(row["comment_count"])
            fit_like_count.append(row["like_count"])
            fit_view_count.append(row["view_count"])
        else:
            x_list = []
            x_list.append(np.nan)
            x_list.append(row["like_count"])
            x_list.append(row["view_count"])
            x_data.append(x_list)
            x_video_id.append(row["video_id"])
            x_like_count.append(row["like_count"])
            x_view_count.append(row["view_count"])

    # Put the videos whose comment_count is not 0 into a new dataframe
    data_train = pd.DataFrame(
        {
            'video_id': fit_video_id,
            'comment_count': fit_comment_count,
            'like_count': fit_like_count,
            'view_count': fit_view_count
        }
    )

    # Use Multiple Imputation to predict the comment_count
    # Get median of predictions from three models: DecisionTree, RandomForest, KNeighbor
    models = [DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor(n_neighbors=3)]
    comment_list_models = []
    for model in models:
        regressor = model
        imp = IterativeImputer(regressor, max_iter=10, random_state=0)
        imp.fit(fit_data)
        comment_list_model = []
        for result in np.round(imp.transform(x_data)):
            comment = result[0]
            comment_list_model.append(comment)
        comment_list_models.append(comment_list_model)
    size = len(comment_list_models[0])
    comment_average_list = []
    index = 0
    while index < size:
        comment_average = int(
            (comment_list_models[0][index] + comment_list_models[1][index] + comment_list_models[2][index]) / 3)
        comment_average_list.append(comment_average)
        index = index + 1

    # Put the prediction of the former videos whose comment_count is not 0 into a new dataframe
    data_test = pd.DataFrame(
        {
            'video_id': x_video_id,
            'comment_count': comment_average_list,
            'like_count': x_like_count,
            'view_count': x_view_count
        }
    )
    print("Successfully predict the comment_count!")
    print("The result of predicted comment_count[1-10]")
    print(data_test[:10])
    print("===" * 20)

    # Concat two dataframes into one dataframe
    df_prediction = pd.concat([data_train, data_test])
    return df_prediction


def normalization(data):
    features_list = []
    rate_v_c = []
    rate_v_l = []
    rate_c_l = []

    # Calculate the rate of view_count to comment_count, view_count to like_count, comment_count to like_count
    for index, row in data.iterrows():
        feature = []
        feature.append((row['view_count'] / (row['comment_count'] + 1)))
        feature.append((row['view_count'] / (row['like_count'] + 1)))
        feature.append((row['comment_count'] / (row['like_count'] + 1)))
        features_list.append(feature)

    # Use Z-score to normalize the three rates
    # Initialize StandardScaler
    scaler = StandardScaler()
    normalized_list = scaler.fit_transform(features_list)
    for feature in normalized_list:
        rate_v_c.append((round(feature[0],3)))
        rate_v_l.append((round(feature[1],3)))
        rate_c_l.append((round(feature[2],3)))
    data['rate_v_c'] = rate_v_c
    data['rate_v_l'] = rate_v_l
    data['rate_c_l'] = rate_c_l

    print("Successfully normalize data! ")
    print("The result of normalization[1-10]")
    print(data[:10])
    print("===" * 20)
    return data


def handler_outliers(normalized_data):
    # Use LOF to remove outliers
    lof_model = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
    outliers = lof_model.fit_predict(normalized_data[['rate_v_c', 'rate_v_l', 'rate_c_l']])
    normalized_data['outlier'] = outliers
    # Outlier_data
    outlier_data = normalized_data[normalized_data['outlier'] == -1]
    # Normalized_data
    normalized_points = normalized_data[normalized_data['outlier'] == 1]
    print("Successfully find outliers!")
    print("The sample of outliers[1-10]")
    print(outlier_data[:10])

    # 3D image of all points, blue is normalized data, red is outliers
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b' if label == 1 else 'r' for label in normalized_data['outlier']]
    x = normalized_data['rate_v_c']
    y = normalized_data['rate_v_l']
    z = normalized_data['rate_c_l']
    ax.scatter(x, y, z, c=colors, marker='o', label='Blue is normalized point, red is outliers')
    ax.set_xlabel('Rate of Views to Comments')
    ax.set_ylabel('Rate of Views to Likes')
    ax.set_zlabel('Rate of Comments to Likes')
    ax.set_title('All points: 3D Scatter Plot of rate of Views to Comments , Views to Likes, and Comments to Likes')

    # 3D image of normalized data
    figg = plt.figure(2, figsize=(10, 10))
    bx = figg.add_subplot(111, projection='3d')
    bx.scatter(normalized_points['rate_v_c'], normalized_points['rate_v_l'], normalized_points['rate_c_l'], c='b', marker='o', label='Data Points')
    bx.set_xlabel('Rate of Views to Comments')
    bx.set_ylabel('Rate of Views to Likes')
    bx.set_zlabel('Rate of Comments to Likes')
    bx.set_title('Normal points: 3D Scatter Plot of rate of Views to Comments , Views to Likes, and Comments to Likes')

    plt.legend()
    plt.show()

    # Remove outliers
    df = normalized_data[normalized_data['outlier'] == 1]
    print("Successfully remove outliers!")
    print("===" * 20)
    return df


def binning_data(data):
    comment_data = []
    like_data = []
    view_data = []
    for index, row in data.iterrows():
        comment_data.append(row['comment_count'])
        like_data.append(row['like_count'])
        view_data.append(row['view_count'])

    # Translate data into pandas series
    data_series_comment = pd.Series(comment_data)
    data_series_like = pd.Series(like_data)
    data_series_view = pd.Series(view_data)

    # Define the number of bins
    num_bins = 5
    names_duration = [0,1,2,3,4]

    # Quantile-based binning
    bins_comment = pd.qcut(data_series_comment, q=num_bins, labels=names_duration)
    bins_like = pd.qcut(data_series_like, q=num_bins, labels=names_duration)
    bins_view = pd.qcut(data_series_view, q=num_bins, labels=names_duration)

    data["comment_bin"] = bins_comment
    data["like_bin"] = bins_like
    data["view_bin"] = bins_view

    print("Successfully bin the data!")
    print("The result of binning data[1-10]")
    print(data[:10])
    print("===" * 20)

    return data


def main():
    # Read in data youtube_api_data directly into pandas
    df_youtube = pd.read_csv('../csv_files/youtube_api_data.csv', sep=',')
    df_remove_duplicate = remove_duplicate(df_youtube)
    df_prediction = handle_comment_missing_value(df_remove_duplicate)
    data_normalized = normalization(df_prediction)
    df_remove_outliers = handler_outliers(data_normalized)
    data = binning_data(df_remove_outliers)
    data.to_csv('../csv_files/youtube_api_data_result.csv', index=False)
    print("Processing completed!")
    print("The result of data[1-10]")
    print(data[:10])


if __name__ == '__main__':
    main()

