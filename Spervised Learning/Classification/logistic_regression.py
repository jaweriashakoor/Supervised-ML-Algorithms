from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"D:\python\Machine Learning\Spervised Learning\Classification\diabetes_risk_dataset.csv")
# print(df.head())
df['Target'] = (df['diabetes_risk_category'] == 'High Risk').astype(int)

features = ['age', 'waist_circumference_cm', 'diabetes_risk_score']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict
# This will output 0 or 1 for each patient in the test set
predictions = model.predict(X_test)
# print("Here is the final prediction:")
# print(predictions.reshape(-1, 1))

# readable_predictions = np.where(predictions == 1, "Yes", "No")
# print("Final Predictions (Yes/No):")
# print(readable_predictions.reshape(-1, 1))

# Convert 1/0 to True/False
# boolean_predictions = predictions.astype(bool)
# print("Final Predictions (True/False):")
# print(boolean_predictions.reshape(-1, 1))

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
plt.plot([0, 1], [0, 1], '--') # Diagonal line (random guessing)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not High Risk', 'High Risk'])
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix for Diabetes Risk')
# plt.show()