import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk.corpus import sentiwordnet as swn
import ast

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

pd.set_option('mode.chained_assignment', None)

# Judge the language of the comment
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


def preprocessing(df):
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

    print("Successfully data preprocessing!")
    print("The result of data preprocessing")
    print(df[:10])
    print("===" * 20)
    df.to_csv("../csv_files/comment_preprocessing.csv",index = False)


def sentimental_analysis():
    df = pd.read_csv("../csv_files/comment_preprocessing.csv")
    comment_list = []
    vader_score_list = []
    afinn_score_list = []
    for index, row in df.iterrows():
        comment = row["context_clean_text"]
        comment_list.append(comment)
        total_sentiment_score = get_sentence_sentiment_score(comment)
    # Initialize VADER
    vader = SentimentIntensityAnalyzer()
    # Initialize AFINN
    afinn = Afinn()

    vader_scores = [vader.polarity_scores(comment)['compound'] for comment in comment_list]
    afinn_scores = [afinn.score(comment) for comment in comment_list]

    for i, comment in enumerate(comment_list):
        vader_score_list.append(vader_scores[i])
        afinn_score_list.append(afinn_scores[i])
    df["vader_score"] = vader_score_list
    df["afinn_score"] = afinn_score_list
    df["senti_score"] = SentiWordNet()
    print("Successfully get sentimental score from VADER, AFINN, SentiWordNet!")
    return df


def set_label(df):
    # Set threshold
    vader_pos_threshold = 0.3
    vader_neg_threshold = 0

    afinn_pos_threshold = 1.0
    afinn_neg_threshold = 0

    senti_pos_threshold = 0.5
    senti_neg_threshold = 0

    vader_label_list = []
    afinn_label_list = []
    senti_label_list = []

    # Set label
    # 1: positive, 0: neutral, -1: negative
    for index, row in df.iterrows():
        vader_score = row["vader_score"]
        if vader_score > vader_pos_threshold:
            vader_label_list.append(1)
        elif vader_score < vader_neg_threshold:
            vader_label_list.append(-1)
        else:
            vader_label_list.append(0)

        afinn_score = row["afinn_score"]
        if afinn_score > afinn_pos_threshold:
            afinn_label_list.append(1)
        elif afinn_score < afinn_neg_threshold:
            afinn_label_list.append(-1)
        else:
            afinn_label_list.append(0)

        senti_score = row["senti_score"]
        if senti_score > senti_pos_threshold:
            senti_label_list.append(1)
        elif senti_score < senti_neg_threshold:
            senti_label_list.append(-1)
        else:
            senti_label_list.append(0)

    df["vader_label"] = vader_label_list
    df["afinn_label"] = afinn_label_list
    df["senti_label"] = senti_label_list

    print("Successfully set label according to the score")
    print("The result of setting labels")
    print(df[:10])
    print("===" * 20)

    return df


def evaluation():
    df_sample = pd.read_csv("../csv_files/youtube_comment_test.csv")
    vader_predict_labels = []
    afinn_predict_labels = []
    senti_predict_labels = []
    true_labels = []
    for index, row in df_sample.iterrows():
        vader_predict_labels.append(row["vader_label"])
        afinn_predict_labels.append(row["afinn_label"])
        senti_predict_labels.append(row["senti_label"])
        true_labels.append(row["true_label"])
    vader_correct_predictions = [1 for true, predicted in zip(true_labels, vader_predict_labels) if true == predicted]
    afinn_correct_predictions = [1 for true, predicted in zip(true_labels, afinn_predict_labels) if true == predicted]
    senti_correct_predictions = [1 for true, predicted in zip(true_labels, senti_predict_labels) if true == predicted]
    vader_accuracy = sum(vader_correct_predictions) / len(true_labels)
    afinn_accuracy = sum(afinn_correct_predictions) / len(true_labels)
    senti_accuracy = sum(senti_correct_predictions) / len(true_labels)
    vader_accuracy_percentage = "{:.2%}".format(vader_accuracy)
    afinn_accuracy_percentage = "{:.2%}".format(afinn_accuracy)
    senti_accuracy_percentage = "{:.2%}".format(senti_accuracy)
    print(f"The accuracy of VADER is {vader_accuracy_percentage}.")
    print(f"The accuracy of AFINN is {afinn_accuracy_percentage}.")
    print(f"The accuracy of AFINN is {senti_accuracy_percentage}.")


def get_sentiment_score(word, pos):
    pos = {
        'N': 'n',
        'V': 'v',
        'R': 'r',
        'J': 'a'
    }.get(pos[0], 'n')

    synsets = list(swn.senti_synsets(word, pos))
    if synsets:
        sentiment_score = synsets[0].pos_score() - synsets[0].neg_score()
        return sentiment_score
    else:
        return 0.0


def get_sentence_sentiment_score(text):
    total_sentiment_score = 0.0
    word_list = ast.literal_eval(text)
    for word in word_list:
        pos = nltk.pos_tag([word])[0][1]
        sentiment_score = get_sentiment_score(word, pos)
        total_sentiment_score += sentiment_score

    return total_sentiment_score


def SentiWordNet():
    df = pd.read_csv("../csv_files/comment_preprocessing.csv")
    score_list = []
    sentiwordnet_predict_list = []
    for index, row in df.iterrows():
        text = row["context_clean_text"]
        total_sentiment_score = get_sentence_sentiment_score(text)
        score_list.append(total_sentiment_score)
    return score_list


def main():
    df = pd.read_csv('../csv_files/youtube_api_comments.csv')
    preprocessing(df)
    df_score = sentimental_analysis()
    df_label = set_label(df_score)
    df_label.to_csv("../csv_files/youtube_api_comments_sentimental_analysis_result.csv", index=False)
    evaluation()


if __name__ == '__main__':
    main()
