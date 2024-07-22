from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = 'G:/My Drive/ML Q2 Project/Dataset/musk_csv2.csv' 
data = pd.read_csv(file)

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]

label_mapping = {'b': 0, 'g': 1}
if y.dtype == 'object':
    y = y.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

svm_model = SVC(kernel='linear', probability=True)  
svm_model.fit(X_train, y_train)

cv_scores = cross_val_score(svm_model, X_test, y_test, cv=10)

print("Cross-Validation Scores on Test Set:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
print("Classification Report:\n", classification_report(y_test, svm_model.predict(X_test)))

cm = confusion_matrix(y_test, svm_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

y_pred_prob = svm_model.predict_proba(X_test)[:, 1]
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
