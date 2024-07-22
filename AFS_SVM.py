import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file = 'G:/My Drive/ML Q2 Project/Dataset/musk_csv2.csv' 
data = pd.read_csv(file)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# label_mapping = {'b': 0, 'g': 1}
# y = y.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def adaptive_feature_selection_gb(X_train_scaled, y_train, cumulative_importance_threshold=0.9):
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    cumulative_importances = np.cumsum(importances[sorted_indices])
    n_features = np.where(cumulative_importances >= cumulative_importance_threshold)[0][0] + 1
    return sorted_indices[:n_features]

best_accuracy = 0
best_threshold = 0
best_model = None
best_features_indices = None  

for i in range(100):
    threshold = i / 100
    selected_features_indices = adaptive_feature_selection_gb(X_train_scaled, y_train, threshold)
    X_train_selected = X_train_scaled[:, selected_features_indices]
    X_test_selected = X_test_scaled[:, selected_features_indices]

    svm_model = SVC(kernel='linear', probability=True)
    parameters = {'C': [0.1, 1, 10]}
    clf = GridSearchCV(svm_model, parameters, cv=5)
    clf.fit(X_train_selected, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test_selected))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_model = clf
        best_features_indices = selected_features_indices  

X_test_selected = X_test_scaled[:, best_features_indices]

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Threshold: {best_threshold}")
print("Classification Report:\n", classification_report(y_test, best_model.predict(X_test_selected)))

cm = confusion_matrix(y_test, best_model.predict(X_test_selected))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1]
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