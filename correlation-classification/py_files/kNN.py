# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def kNN(df):
  # Cross Validation and Grid Search
  # X includes audio featrues of songs
  X = df.drop(columns=['labels', 'tracks_id'])
  # y includes mood labels
  y = df['labels']  # get labels
  # Splitting into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # KNN Classifier
  knn = KNeighborsClassifier()
  # 5 fold cross-validation
  cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
  print("Cross-validation scores:", cv_scores)
  print()
  # Define the hyperparameter grid
  param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
  }
  # GridSearch object with 5 fold cross validation
  grid_search = GridSearchCV(knn, param_grid, cv=5)
  # Start Searching
  grid_search.fit(X_train, y_train)
  grid_search.score(X_train, y_train)
  # Output best parameters and score
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
  # Confusion matrix
  conf_matrix = confusion_matrix(y_test, y_pred)
  # Visualize of confusion matrix
  print("Heatmap of Confusion Matrix")
  plt.figure(figsize=(6, 3))
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
              xticklabels=["Sad", "Happy", "Energetic", "Calm"],
              yticklabels=["Sad", "Happy", "Energetic", "Calm"])
  plt.ylabel("Actual")
  plt.xlabel("Predicted")
  plt.show()
  print()
  return

def main():
  print("Calling main")
  df = pd.read_csv('../csv_files/spotify_mood.csv')
  print("Data imported")
  print("KNN Classifier: ")
  kNN(df)
  print("Completed")
  return

if '__main__' == __name__:
    main()