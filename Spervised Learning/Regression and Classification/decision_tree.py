import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Load the data
file_path = r"D:\python\Machine Learning\Spervised Learning\Regression and Classification\tech_mental_health_burnout.csv"
df = pd.read_csv(file_path).dropna()

# 2. ENCODING: Convert 'job_role' into numbers
# pd.get_dummies creates new 0/1 columns for every job role
X = df[['age','job_role']]
X = pd.get_dummies(X, columns=['job_role'])

y_reg = df['burnout_score']  # Continuous numbers
y_clf = df['burnout_level']  # Text categories

# 4. Initialize both models
reg_model = DecisionTreeRegressor(max_depth=3)
clf_model = DecisionTreeClassifier(max_depth=3)

# 5. Train
reg_model.fit(X, y_reg)
clf_model.fit(X, y_clf)

# 6. Visualize both (in different windows)
# FIGURE 1: Regression (Numeric) 
plt.figure("Regression Analysis", figsize=(20, 10)) 
plot_tree(reg_model, feature_names=list(X.columns), filled=True, precision=2)
plt.title("1. Regression Tree: Predicting Burnout Score (Numeric)", fontsize=16)

# FIGURE 2: Classification (Category) 
plt.figure("Classification Analysis", figsize=(20, 10))
plot_tree(clf_model, feature_names=list(X.columns), class_names=list(y_clf.unique()), filled=True)
plt.title("2. Classification Tree: Predicting Burnout Level (Category)", fontsize=16)

plt.show()