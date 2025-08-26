import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Roc curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # confusion matrix
import seaborn as sns

from joblib import dump

df = pd.read_csv('../csv_files/spotify_mood.csv')
# X includes audio featrues of songs，y includes mood labels
# get audio features of songs
X = df.drop(columns=['labels', 'tracks_id'])
# get labels
y = df['labels']

# Divide dataset into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#####################################
# Random Forest: Cross validation with grid search
#####################################
def cross_validation():

    # Set parameters of grid search
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier()

    # Set grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Parameters of best model:")
    print(grid_search.best_params_)

    # Get best model from grid search
    best_model = grid_search.best_estimator_
    # Get prediction from best model
    y_pred = best_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("f1-score:", f1_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print(classification_report(y_test, y_pred))

    return best_model


def random_forest_model():
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_pred


def roc_validation(y_pred):

    # Binary the labels
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])

    # Calculate the ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 4

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate the micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Draw multiclass ROC curves
    plt.figure()
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


def confusion_matrix_validation(y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(6, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sad", "Happy", "Energetic", "Calm"],
                yticklabels=["Sad", "Happy", "Energetic", "Calm"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


def main():
    best_model = cross_validation()
    y_pred = random_forest_model()
    roc_validation(y_pred)
    confusion_matrix_validation(y_pred)
    dump(best_model, '../random_forest_model.joblib')


if __name__ == '__main__':
    main()