import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

file = "G:/My Drive/ML Q2 Project/Dataset/diabetes.csv"  
data = pd.read_csv(file)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

def adaptive_feature_selection_gb(X_train, y_train, cumulative_importance_threshold=0.95):
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    cumulative_importances = np.cumsum(importances[sorted_indices])
    n_features = np.where(cumulative_importances >= cumulative_importance_threshold)[0][0] + 1
    return sorted_indices[:n_features]

best_accuracy = 0
best_threshold = 0
best_features_indices = []

for i in range(100):
    threshold = i / 100
    selected_features_indices = adaptive_feature_selection_gb(X_train, y_train, threshold)
    X_train_selected = X_train.iloc[:, selected_features_indices]
    X_test_selected = X_test.iloc[:, selected_features_indices]

    nb_model = GaussianNB()
    cv_scores = cross_val_score(nb_model, X_train_selected, y_train, cv=10)
    accuracy = np.mean(cv_scores)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_features_indices = selected_features_indices

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Threshold: {best_threshold}")

nb_model.fit(X_train.iloc[:, best_features_indices], y_train)
y_pred = nb_model.predict(X_test.iloc[:, best_features_indices])

print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

y_pred_prob = nb_model.predict_proba(X_test.iloc[:, best_features_indices])[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='---')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
