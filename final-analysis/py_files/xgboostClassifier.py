# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import plotly.express as px
import chart_studio.plotly as py

# Function to model the data using XGBoost Classifier
def xgboostClassifier(df):
    print("XGBoost Classifier")
    # X includes audio features of songs
    X = df.drop(columns=['labels', 'track_id'])
    # y includes mood labels
    y = df['labels']
    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # XGBoost Classifier
    model = XGBClassifier()
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Test Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", test_accuracy)
    print()
    # Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])
    # Calculate the ROC curve and AUC for each class
    n_classes = 4
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Calculate the micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Draw multiclass ROC curves
    print("ROC Curve")
    plt.figure(figsize=(10, 6))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Visualize the confusion matrix
    print()
    print()
    print("Heatmap of Confusion Matrix")
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues",
                xticklabels=["Sad", "Happy", "Energetic", "Calm"],
                yticklabels=["Sad", "Happy", "Energetic", "Calm"])
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), ha='center', va='center')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    print()
    print()
    return model

# Function to predict mood of the popular Spotify songs
def predictMood(pop, model):
    pred = pop
    pred = pred.drop('track_popularity', axis=1)
    pred = pred.drop('artist_popularity', axis=1)
    pred = pred.drop('track_id', axis=1)
    pred = pred.drop('pop_label', axis=1)
    # Predicting mood for the popular songs dataset
    pred = model.predict(pred)
    # Converting the predictions to a pandas DataFrame
    pred = pd.DataFrame(pred, columns=['predicted_mood'])
    print('Moods predicted')
    return pred

# Function to generate a CSV of the moods predicted for popular songs
def predictMoodCSV(pop, pred):
    # Merging the popularity song dataset with predicted moods
    csv = pd.concat([pop, pred], axis=1)
    # Printing the count of null values in the merged dataset
    print('Null values for each field of the dataset')
    print(csv.isnull().sum())
    # Saving the merged csv file that contains popular songs with predicted moods
    csv.to_csv('../csv_files/spotify_popular_songs_mood_predicted.csv', index=False)
    return csv

# Plot of popular song with generated moods
def moods(csv):
    # Create a countplot using Plotly Express
    fig = px.histogram(csv, x='pop_label', color='predicted_mood', barmode='group',
                   labels={'pop_label': 'Popularity', 'predicted_mood': 'Predicted Mood'},
                   title='Popularity and Mood')
    fig.update_layout(
        xaxis_title='Popularity',
        yaxis_title='Count',
        legend_title='Mood'
    )
    fig.show()
    # Plotly Chart Studio credentials
    username = 'vb506'
    api_key = 'DFX8R3Dqc8k5L2KXbFlA'
    # Set up Plotly Chart Studio credentials
    py.sign_in(username, api_key)
    # Upload the figure to Plotly Chart Studio
    url = py.plot(fig, filename='Moods of Popular Songs', auto_open=False)
    # Print the generated web link
    print("Web Link:", url)
    print()
    return

# Main function
def main():
    # Spotify API Mood CSV - downloaded dataset
    df = pd.read_csv('../csv_files/spotify_api_mood_cleaned.csv')
    # Spotify Popular Songs CSV - generated dataset
    pop = pd.read_csv('../csv_files/spotify_popular_songs.csv')
    print("Data imported")
    # Modelling the data
    print('Classifying data')
    model = xgboostClassifier(df)
    # Predicting moods for the data
    print('Predicting moods')
    pred = predictMood(pop, model)
    # Generating a CSV file for the predicted moods
    print('Generating CSV')
    csv = predictMoodCSV(pop, pred)
    print('CSV generated and saved')
    # Plotting graphs for moods generated
    moods(csv)
    print("Completed")
    return

if __name__ == '__main__':
  main()