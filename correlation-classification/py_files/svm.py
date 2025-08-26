# import library
import pandas as pd
import numpy as np

# Data visulization
import matplotlib.pyplot as plt
import seaborn as sns

# Data classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score # cross validation
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.model_selection import GridSearchCV # GridSearch
from sklearn.metrics import classification_report

# Roc curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def main():


    # import spotify_mood.csv dataset
    df = pd.read_csv('../csv_files/spotify_mood.csv')

    # See spotify_mood data frame
    print('Print spotify_mood.csv dataset:')
    print(df)

    #####################
    # SVM: Cross Validation & Grid Search
    #####################


    # X includes audio featrues of songs，y includes mood labels
    X = df.drop(columns=['labels', 'tracks_id'])  # get audio features of songs
    y = df['labels']  # get labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Score kernels using cross validation
    # 1. linear
    # 2. Gaussian
    # 3. Polynomial
    # 4. Sigmoid

    C = 1.0  # SVM regularization parameters
    models = (svm.SVC(kernel='linear',  C=C),
              svm.SVC(kernel='rbf',  C=C),
              svm.SVC(kernel='poly', degree=3, C=C),
              svm.SVC(kernel='sigmoid', C=C))

    score_max = 0  # max score
    index_max = -1  # index of model with max score

    for index, model in enumerate(models):

        scores = cross_val_score(model, X_train, y_train)
        score = np.mean(scores)
        if score > score_max:
            score_max = score
            index_max = index

    kernels = ['linear', 'Gaussian', 'Polynomial', 'Sigmoid']  # list of kernals
    max_model = kernels[index_max]

    print("\nThe best kernel is: " + max_model)
    print(f"The score of the best kernel is: {score_max}")

    svm_model = models[index_max]

    # Since the best kernel is 'linear', we will perform gridSearch in svm model with 'lineal' kernel and \
    # because the most important parameter of linear svm model is C, we will find the best 'C' value by gridSearch

    # Parameters of GridSearch
    param_grid = {'C': [0.1, 1, 100]}

    # Create a GridSearch object with 5-fold cross validation
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)

    # Start Searching
    grid_search.fit(X_train, y_train)
    grid_search.score(X_train, y_train)  # output the socres

    print("\nThe best parameters and scores after gridSearch are: ")
    print(grid_search.best_params_, grid_search.best_score_)  # best parameters and scores

    # Get the best model
    best_model = grid_search.best_estimator_

    # Use the best model to make predictions
    y_pred = best_model.predict(X_test)

    print("\nModel has predicted the test data successfully!")

    # Show performance of svm model with classification_report
    report = classification_report(y_test, y_pred)
    print("\nShow performance of svm model: ")
    print(report)

    #####################
    # Roc Curve for svm model
    #####################

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

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

    #####################
    # Confusion Matrix for svm model
    #####################

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


if '__main__' == __name__:
    main()

