import pandas as pd
from langdetect import detect
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from joblib import dump
from joblib import load
from sklearn.pipeline import Pipeline


# Judge the comment is english or not
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


# Proprecess comment text
def proprecessing():
    df = pd.read_csv('../csv_files/comments_sentilabels.csv')
    df = df[df['comment_text'].apply(is_english)]

    text_list = []
    text_clean_list = []

    for index, row in df.iterrows():
        text = row['comment_text']
        text_list.append(text)
    for text in text_list:
        # Remove emoji
        text = re.sub(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FE00-\U0001FE0F\U0001F004-\U0001F0CF\U0001F170-\U0001F19A\U0001F200-\U0001F251\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FE00-\U0001FE0F\U0001F004-\U0001F0CF\U0001F170-\U0001F19A\U0001F200-\U0001F251]+',
            '', text)

        # Remove punctuation and special characters
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])

        # Lowercasing
        text = text.lower()

        # Split words
        words = word_tokenize(text)

        # Remove stop_words
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        text = stemmed_words
        text_clean_list.append(text)

    df["context_clean_text"] = text_clean_list
    df.to_csv("../csv_files/comment_preprocessing.csv", index=False)

    print("Successfully data preprocessing!")
    print("The result of data preprocessing")
    print(df[:10])
    print("==="*20)
    
    return df


def pipeline_model():
    df = pd.read_csv('../csv_files/comment_preprocessing.csv')
    X = df["context_clean_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge TF-IDF and Random Forest into one pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('rf', RandomForestClassifier())
    ])

    # Set gridsearch params
    param_grid = {
        'tfidf__max_df': [0.5, 0.75],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'rf__n_estimators': [100, 200],
        'rf__max_features': ['sqrt'],
        'rf__max_depth': [None, 20, 40],
        'rf__min_samples_split': [2, 10],
        'rf__min_samples_leaf': [1, 4],
        'rf__bootstrap': [True]
    }

    # Create GridSearchCV instance
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

    # GridSearch
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross Validation Score:", grid_search.best_score_)

    # Use best model to predict labels
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    dump(best_model, '../randomForest.joblib')

    report = classification_report(y_test, y_pred)
    print(report)

    print("Successfully generate Random Forest model!")
    print("RF report:")
    print("===" * 20)


def prediction_label():
    df = pd.read_csv("../csv_files/youtube_comments_sentimental_analysis.csv")
    data = pd.DataFrame(
        {
            'video_id': df["video_id"],
            'comment_text': df["comment_text"],
            'context_clean_text': df["context_clean_text"]
        }
    )
    X = data["context_clean_text"]

    # Load Random Forest model
    model = load('../randomForest.joblib')
    # Predict sentimental labels
    predictions = model.predict(X)
    data['predicted_label'] = predictions

    print("Successfully predict labels!")
    print(data[:10])
    print("===" * 20)
    
    return data


def calculate_sentimental_score(df):
    # Group each video's sentimental label
    comment_df = df.groupby('video_id')['predicted_label'].value_counts().unstack().fillna(0).reset_index()
    # 0:negative 1:neural 2:positive
    comment_df['total'] = comment_df[[0, 1, 2]].sum(axis=1)
    # Calculate the proportions of positive, neural and negative in all labels separately
    for col in [0, 1, 2]:
        comment_df[col] = comment_df[col] / comment_df['total']

    comment_df.drop(columns='total', inplace=True)

    sentiment_score = []
    # Calculate sentimental score of each video
    for index, row in comment_df.iterrows():
        score = (-1) * float(row[1]) + 0 * float(row[2]) + 1 * float(row[3])
        sentiment_score.append(score)
    comment_df['senti_score'] = sentiment_score

    final_df = pd.DataFrame({
        'video_id': comment_df['video_id'],
        'senti_score': comment_df['senti_score']
    })

    final_df.to_csv("../csv_files/youtube_comments_sentimental_analysis_result.csv", index=False)
    print("Successfully calculate sentimental score!")
    print(final_df[:10])


def main():
    # Generate comment_preprocessing.csv
    proprecessing()

    # Generate randomForest.joblib
    pipeline_model()

    # Predict sentimental labels
    df_predicted = prediction_label()

    # Calculate sentimental score
    calculate_sentimental_score(df_predicted)


if __name__ == '__main__':
    main()