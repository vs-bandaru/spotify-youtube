import pandas as pd
from joblib import load

# Read File spotify_api_songs_popularity_analysis_result.csv
df = pd.read_csv('../csv_files/spotify_api_songs_popularity_analysis_result.csv')
X = df.drop(columns=['track_id', 'track_popularity', 'artist_popularity', 'duration_group', 'speechiness_group', 'liveness_group'])


def main():
    model = load('../random_forest_model.joblib')
    predictions = model.predict(X)
    df['predicted_label'] = predictions
    df.to_csv("../csv_files/predictionResult.csv", index= False)


if __name__ == '__main__':
    main()