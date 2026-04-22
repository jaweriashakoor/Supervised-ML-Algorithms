from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split # Added this tool
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv(r"D:\python\Machine Learning\Spervised Learning\Regression and Classification\istanbul_apartment_prices_2026.csv")

# 2. Basic Cleaning
cols_to_use = ['price', 'price_per_sqm', 'district', 'exchange']
df = df[cols_to_use]
df_encoded = pd.get_dummies(df, columns=['district', 'exchange'])

# 3. Create Features (X) and Targets (y)
X = df_encoded.drop('price', axis=1)
y_reg = df_encoded['price']
y_clf = pd.qcut(df_encoded['price'], q=3, labels=['Cheap', 'Average', 'Expensive'])

# 4. THE SPLIT (80% for learning, 20% for testing)
# We do this for regression first as an example
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Also split for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# 5. Train Models (Fit only on the TRAIN data)
model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
model_reg.fit(X_train, y_train)

model_clf = RandomForestClassifier(n_estimators=50, random_state=42)
model_clf.fit(X_train_c, y_train_c)

# 6. Predict on the TEST data (The 20% the model has NEVER seen)
reg_predictions = model_reg.predict(X_test.head())
clf_predictions = model_clf.predict(X_test_c.head())

print("="*60)
print("RESULTS ON UNSEEN DATA (20% TEST SET)")
print("="*60)

for i in range(5):
    print(f"New House {i+1}:")
    print(f"  -> Predicted Price (Regression): {reg_predictions[i]:,.0f} TRY")
    print(f"  -> Predicted Category (Classification): {clf_predictions[i]}")
    print("-" * 30)

# 7. Visualizing the "Exam" Results
plt.figure(figsize=(8, 5))
plt.bar(['New 1', 'New 2', 'New 3', 'New 4', 'New 5'], reg_predictions, color='skyblue')
plt.title('Regression: Predictions on New Houses')
plt.ylabel('Price in TRY')
plt.show()

all_test_preds = model_clf.predict(X_test_c) # Predict for all unseen houses
category_counts = pd.Series(all_test_preds).value_counts()

plt.figure(figsize=(8, 5))
category_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'gold', 'tomato'])
plt.title('Classification: How Model Labeled Unseen Houses')
plt.ylabel('')
plt.show()