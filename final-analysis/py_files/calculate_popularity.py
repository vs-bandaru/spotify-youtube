import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from joblib import dump
from joblib import load


# Generate popularity model by SVM
def popularity_model():
    # load dataset with popularity labels
    df = pd.read_csv('../csv_files/popularity_train_set.csv')
    popularity_label = []

    # Bin the popularity score: 0-less popular 1-medium popular 2-very popular
    for index, row in df.iterrows():
        popularity_score = row["popularity_score"]
        if popularity_score > 20:
            popularity_label.append(2)
        elif popularity_score > 10:
            popularity_label.append(1)
        else:
            popularity_label.append(0)
    df["popularity_label"] = popularity_label

    # Standard features
    scaler = StandardScaler()
    df[['track_popularity_scale', 'artist_popularity_scale', 'like_count_scale',
        'view_count_scale', 'senti_score_scale']] = scaler.fit_transform(
        df[['track_popularity', 'artist_popularity', 'like_count', 'view_count', 'senti_score']])

    X = df[['track_popularity_scale', 'artist_popularity_scale', 'like_count_scale', 'view_count_scale', 'senti_score_scale']]
    y = df['popularity_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set random forest params
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize rf model
    model = RandomForestClassifier()
    # Initialize grid_search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    # Generate best model
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Successfully train model!")
    print(report)
    dump(best_model, '../popularity_predict.joblib')


# Predict popularity label
def predict_label():
    model = load('../popularity_predict.joblib')
    df = pd.read_csv('../csv_files/spotify_youtube_popularity_features.csv')
    # Standard features
    scaler = StandardScaler()
    df[['track_popularity_scale', 'artist_popularity_scale', 'like_count_scale',
        'view_count_scale', 'senti_score_scale']] = scaler.fit_transform(
        df[['track_popularity', 'artist_popularity', 'like_count', 'view_count', 'senti_score']])

    # Train set
    X = df[['track_popularity_scale', 'artist_popularity_scale', 'like_count_scale',
        'view_count_scale', 'senti_score_scale']]
    predictions = model.predict(X)
    df['pop_label'] = predictions
    df.to_csv("../csv_files/popularity_prediction_result.csv", index=False)

    print("Successfully predict popularity labels!")
    print(df[:10])


def main():
    # popularity_model()
    predict_label()


if __name__ == '__main__':
    main()