# Final Analysis
## csv_files
### spotify_youtube.csv
This is the dataset with about 30,000 songs in the last decade with Spotify ID, URL and YouTube video ID and URL.
### mood.csv
This is the dataset with about 30,000 songs with Spotify uri and labels that affect people’s mood. 0: Sad, 1: Happy, 2: Energetic and 3: Calm.
### spotify_api_mood.csv
This is the dataset with first 10,000 songs in mood.csv with Spotify ID and 16 audio features and mood labels.
### spotify_api_song.csv
This is the dataset with first 10,000 songs in the last decade with Spotify ID in spotify_youtube.csv and 16 audio features and 2 popularity features.
### spotify_api_mood_cleaned.csv
This is the dataset after the data cleaning (plus removing outliers in project 4) for spotify_api_mood.csv. It generated from spotify_outliers_removed.py.
### spotify_api_song_cleaned.csv
This is the dataset after the data cleaning (plus removing outliers in project 4) for spotify_api_song.csv. It generated from spotify_outliers_removed.py.
### spotify_popular_songs.csv
This dataset is a generated dataset that contains popular songs and its audio features along with 'pop_label' which indicates popularity. 0: Low, 1: Medium, 2: High.
### spotify_popular_songs_mood_predicted.csv
This dataset is a generated dataset and contains moods predicted by the XGBoost Classifier for the popular songs and it's features in spotify_popular_songs.csv along with 'predicted_mood'. 0: Sad, 1: Happy, 2: Energetic and 3: Calm.
### comment_preprocessing.csv
This is the result of sentimental analysis preprocessing.
### comment_sentilabels.csv
This is the dataset with sentimental labels. It is used as training set for sentimental analysis.
This dataset is from https://www.kaggle.com/datasets/advaypatil/youtube-statistics
### youtube_comments_sentimental_analysis.csv
This is the dataset used to predict sentimental label.
### youtube_comments_sentimental_analysis_result.csv
This is the dataset with predicted sentimental score.
### popularity_train_set.csv
This is the dataset with manual popularitu labels. It is used as training set for popularity analysis.
The manual popularity labels are based on Chartmetric(https://chartmetric.com).
### spotify_youtube_popularity_features.csv
This is the dataset used to predict popularity.
### popularity_prediction_result.csv
This is the dataset with predicted popularity labels.


## py_files
### histograms_and_correlations.py
This py file uses Plotly to produce interoperable frequency distribution histograms and correlation heatmap of audio features in spotify_api_song_cleaned.csv. 
1. main() - This function is the main function that imports the datasets and implements all the other functions.
2. histogram_correlation(df) - This function imports Plotly to generate histograms and correlation heatmap of song's audio features and upload it to Mandy's chart-studio.
### new_decision_tree.py
This py file implements Decision Tree Classifier on the spotify_api_mood_cleaned.csv dataset. Unlike project 3, this time more hyperparameters are introduced to do the grid search to find the best model. Parameters include: criterion, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, ccp_alpha.
1. main() - This function is the main function that imports the datasets and implements all the other functions.
2. decisionTree(df) - This function uses Decision Tree classifiers to train a model to predict mood labels on spotify_api_mood_cleaned.csv dataset.
3. roc_validation(y_test,y_pred) - This function generates a Roc curve graph to evaluate the performance of Decision Tree.
4. confusion_matrix_validation(y_test,y_pred) - This function generates a confusion matrix to evaluate the performance of Decision Tree.
### spotify_outliers_removed.py
This py file uses Z-score statistics method to remove outliers in spotify_api_mood.csv and spotify_api_song.csv.
1. main() - This function is the main function that imports the datasets and implements all the other functions.
2. remove_outliers(df_mood,df_song) - This function specifys 3 threshold = 2, 2.5 and 3 to remove outliers with Z-score statistic method and uses box-plot to evaluate the performance.
### clustering.py
This py file contains different clustering algorithms like Kmeans, Ward, DBScan and GMM on the spotify_api_mood_cleaned.csv dataset. It has the following functions:
1. main() - It is a main function that imports the datasets required and implements the other functions.
2. pca(data) - It is used to scale the data using Standard Scaler and perform PCA on the data to reduce dimensionality.
3. clusters(df) - It performs Kmeans, Ward, DBScan and GMM clustering algorithm on the reduced data and prints Calinski-Harabasz score along with Silhouette score for the clusters formed.
### xgboostClassifier.py
This py file contains XGBoost Classifier model used to model the spotify_api_mood_cleaned.csv dataset and predict moods for the songs from the spotify_popular_songs.csv dataset. It generates a csv with popular songs with its audio features, pop_label (popularity label) and predicted_mood (predicted mood label). It has the following functions:
1. main() - It is a main function that imports the datasets required and implements the other functions.
2. xgboostClassifier(df) - It is used to model the data using XGBoost Classifier and plots ROC curve, confusion matrix heatmap and prints accuracy along with classification report.
3. predictMood(pop, model) - It is used to predict mood of the popular songs using the XGBoost model and returns those predictions.
4. predictMoodCSV(pop, pred) - It is used to generate a csv of the popular songs dataset with predicted moods.
5. moods(csv) - It is used to plot the predicted moods for the popular songs using Plotly and uploads it to chart-studio.
### findings_violin_plot_and_box_plot.py
This py file uses violin plots to analyze how 4 mood features (danceability, energy, valence and tempo) of popular songs in spotify_popular_songs.csv present and box plots to how popular songs performed on all audio features for 3 categories of mood (0: Sad, 1: Happy, 2: Energetic).
1. main() - It is a main function that imports the datasets required and implements the other functions.
2. violin_plot_for_mood_features(df) - It generates 4 violin plots to show how danceability, energy, valence and tempo presents on popuplar songs.
3. box_plot_for_mood_features(df) - It generates 10 box plots for all audio features to show how they presents on popular songs classified by 3 mood labels.
### sentimental_model.py
1. proprecessing() - Proprecessing for comments context.
2. pipeline() - Combine TF-IDF and Random Forest to generate model.
3. prediction_labels() - It predicts sentimental labels.
4. calculate_sentimental_score() - It calculates sentimental scores.
### calculate_popularity.py
1. popularity_model() - It generates popularity model by random forest.
2. predict_model() - It predicts popularity labels. 0:Less popular, 1:Medium popular, 2:Very popular
### popularity_visualization.py
It generates visualization for popularity analysis. You can visit by https://chromatic-idea-406803.an.r.appspot.com

## popularity_predict.joblib
This is the model of predicting popularity. The accuracy of this model is 65%. You can use it directly to predict one song's popularity.
