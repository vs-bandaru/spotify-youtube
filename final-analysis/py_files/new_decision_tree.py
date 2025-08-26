# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def decisionTree(df):

    # Cross Validation and Grid Search

    # X includes audio featrues of songs
    X = df.drop(columns=['labels', 'track_id'])
    # y includes mood labels
    y = df['labels']  # get labels

    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Classifier
    decision_tree = DecisionTreeClassifier()

    # Define the hyperparameter grid
    param_grid = {
    'criterion': ['entropy'], # Measures the quality of the split
    'max_depth': [3, 5, 10, None], # The maximum depth of the tree. Deeper trees can capture more complex patterns, but can also lead to overfitting
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split internal nodes. Increasing this value reduces the complexity of the model and reduces the risk of overfitting
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required on leaf nodes, which helps control model complexity and overfitting
    'max_features': ['auto', 'sqrt', 'log2', None], # The number of features to consider when finding the best split. This can improve efficiency and reduce the risk of overfitting
    'class_weight': [None, 'balanced'], # Weight of class
    'min_weight_fraction_leaf': [0.0], # The fraction of the minimum weighted sample total required for leaf nodes. This can help prevent the tree from becoming too complex and resulting in overfitting.
    'max_leaf_nodes': [None, 30], # Maximum number of leaf nodes. Limiting the number of leaf nodes can make the tree more concise.
    'min_impurity_decrease': [0.0, 0.01], # The minimum impurity that a node should reduce after splitting.
    'ccp_alpha': [0.0, 0.01] # Cost complexity pruning parameter used to minimize the complexity of the tree.
    }

    # GridSearch object with 5-fold cross validation
    grid_search = GridSearchCV(decision_tree, param_grid, cv=5)

    # Start Searching
    grid_search.fit(X_train, y_train)
    grid_search.score(X_train, y_train)

    # Output the best parameters and score
    print("\nThe best parameters and scores after gridSearch are: ")
    print(grid_search.best_params_, grid_search.best_score_)
    print()

    # Get the best model
    best_model = grid_search.best_estimator_

    # Use the best model to make predictions
    y_pred = best_model.predict(X_test)

    print("\nModel has predicted the test data successfully!")
    print()

    # Test Accuracy
    test_accuracy = best_model.score(X_test, y_test)
    print("Test set accuracy", test_accuracy)
    print()

    # Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy {accuracy:.2f}")
    print()
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print()

    # roc curve
    roc_validation(y_test,y_pred)

    # confusion matrix
    confusion_matrix_validation(y_test,y_pred)


def roc_validation(y_test,y_pred):
    # ROC
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
    # Calculate the micro - average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Draw multiclass ROC curves
    print("ROC Curve")
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
    print()

def confusion_matrix_validation(y_test,y_pred):

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

    return

def main():

    df = pd.read_csv('../csv_files/spotify_api_mood_cleaned.csv')

    decisionTree(df)

    return

if '__main__' == __name__:
    main()