# Data Analysis
## csv_files
### mood.csv
This is the downloaded dataset that has moods for each song labelled - 0: Sad, 1: Happy, 2: Energetic, 3: Calm.
### spotify_api_mood.csv
This is the dataset that has the features of SpotifyAPI data for each song in the mood.csv dataset.
### comment_preprocessing.csv
The result of youtube comment text data proprecessing. It is generated from youtube_api_comments.csv by def preprocessing() in youtube_api_comment.py
### spotify_api_popularity.csv
This is the dataset that has the features of popularity and artist popularity for each song in spotify_youtube.csv dataset.
### spotify_api_songs.csv
This is the dataset that has the features of SpotifyAPI data for each song in spotify_youtube.csv dataset.
### spotify_api_songs_popularity_result.csv
This is the dataset after the data cleaning and analysis for spotify_api_songs.csv and spotify_api_popularity.csv datasets. It generated from spotify_songs_popularity.py 
### spotify_youtube.csv
This is the downloaded dataset that has the features for songs in both Spotify and Youtube platforms.
### youtube_api_comments.csv
This is the dataset that has comments of each video in youtube_api_data.csv.
### youtube_api_comments_sentimental_analysis_result.csv
The final result of youtube comment sentimental analysis. It is generated from youtube_api_comments.csv in youtube_api_comment.py
### youtube_api_data.csv
This is the dataset that has the features of YouTube videoes.
### youtube_api_data_result.csv
The final result of youtube songs data preprocessing and cleaning. It is generated from youtube_api_data.csv in youtube_api_song.py
### youtube_comment_test.csv
It is a manually labeled test dataset. 'true_label_a','true_label_b' is the manual label.

## py_files
### spotify_mood.py
This py file has Exploratory Data Analysis, correlation and clustering for the mood.csv and spotify_api_mood.csv. It has the following functions:
1. main() - This function is the main function that implements all the other functions.
2. data() - This function merges mood.csv and spotify_api_mood.csv. The merged data is df and is returned to the main().
3. printdata(df) - This function is used to print the data head rows, tail rows etc.
4. univariateAnalysis(df) - This function is used to resperesnt the plots for univariate analysis in Exploratory Data Analysis.
5. bivariateAnalysis(df) - This function is used to resperesnt the plots for bivariate analysis in Exploratory Data Analysis.
6. missingValuesTreatment(df) - This function is used to identify missing values.
7. describeData(df) - This function prints data description.
8. statisticalOutlier(df) - This function is used to identify outliers using IQR.
9. plots(df) - This function is used to plot different fileds to display outliers.
10. correlation(df) - This function is used to correlate values.
11. pca(df) - This function is used to implement PCA and the data is returned as df1.
12. clusters(df1) - This function is used to implement different clusters like Ward, K-means, DBSCAN, GMM.
### spotify_songs_popularity.py
The py file spotify_songs_popularity.py contains data preprocessing, data cleaning, basic statistical analysis, and histograms for the datasets spotify_api_songs.csv and spotify_api_popularity.csv.<br>


Usage
- Import datasets spotify_api_songs.csv and spotify_api_popularity.csv
- Run the code to view the output<br>


Method
- data_analysis(): The function is divided into the 4 main sections: Data preprocess, Clean & Organize the data, Basic Statistical Analysis and data cleaning and Histograms and Correlation.<br>


Section Details
- Data preprocessing: We explored the spotify_api_songs and spotify_api_popularity datasets in multiple ways.<br>
- Clean & Organize the data: We dropped unnecessary columns, handled duplicate values, merged two datasets, handled missing values and converted 'duration_ms' into 'duration_min'.<br>
- Basic Statistical Analysis and data cleaning: We viewed the mean, median and standard deviation of the attributes, handled the outliers and binned the data.<br>
- Histograms and Correlation: We generated the histograms for all attributes.<br>

### youtube_api_song.py
Youtube songs attributes preprocessing and cleaning. It generates youtube_api_data_result.csv.
### youtube_api_comment.py
Youtube comment text sentimental analysis. It analyze each comment sentiment: positive, negative, neural.<br>


Usage
- Before run it, you can delete comment_preprocessing.py and youtube_api_comments_analysis_result.csv
- When the programme is running, it needs to download packages below:
  nltk.download('stopwords')<br>
  nltk.download('punkt')<br>
  nltk.download('wordnet')<br>
  nltk.download('averaged_perceptron_tagger') <br>
  It takes a few minutes to download them. If your nltk version is not 3.8, please change        nltk.download('wordnet') to nltk.download('sentiwordnet').
- If it shows that 'No Module chardet', please install 'chardet' firstly: pip install chardet. 

Method
- is_english(): Judge language of comment.<br>
- preprocessing(): Data preprocessing, including remove emoji, remove punctuation and special characters, l lowercasing, split words, remove stop_words, stemming. It generates comment_preprocessing.csv.<br>
- sentimental_analysis(): Use VADER, AFINN, SentiWordNet to predict sentimental score of comments.<br>
- set_label(): According to sentimental scores, set labels to comments.(1-positive, 0-neural, -1-negative)<br>
- evaluation(): Use youtube_comment_test.csv to evaluate accuracy of VADER, AFINN, SentiWordNet.<br>
- get_sentiment_score(): SentiWordNet calculate each word score<br>
- get_sentence_sentiment_score(): SentiWordNet calculate each sentence score<br>
- SentiWordNet(): Calculate each comment score.
### youtube_api_song.py
Youtube songs attributes preprocessing and cleaning. It generates youtube_api_data_result.csv.

Method

- remove_duplicate(): According 'video_id' remove duplicate data.<br>
- handle_comment_missing_value(): Evaluate comment_count which is 0 in raw dataset.<br>
- normalization(): Normalize the rate of view_count to comment_count, view_count to like_count, comment_count to like_count.<br>
- handler_outliers(): Find outliers<br>
- binning_data(): Set comment_count, like_count, view_count into 5 bins.
