import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# 1. Load the data
file_path = r"D:\python\Machine Learning\Spervised Learning\Regression and Classification\Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
df = pd.read_csv(file_path)

# 2. Data Cleaning
df = df.drop(['transaction_id', 'user_id'], axis=1)

le = LabelEncoder()
df['stress_level'] = le.fit_transform(df['stress_level'])
df['academic_work_impact'] = le.fit_transform(df['academic_work_impact'])
df['gender'] = le.fit_transform(df['gender'])

# Exact feature names from your dataset
features = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'notifications_per_day', 'stress_level', 'academic_work_impact']
X = df[features]

# --- TASK 1: CLASSIFICATION ---
y_class = df['addicted_label']
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)

# --- TASK 2: REGRESSION ---
y_reg = df['sleep_hours']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train_r) 

# --- VISUALIZATION ---
plt.figure(figsize=(14, 5))

# Plot 1: Classification Clusters
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['daily_screen_time_hours'], y=df['notifications_per_day'], hue=df['addicted_label'], palette='coolwarm', alpha=0.6)
plt.title('Phone Addiction Clusters')
plt.xlabel('Daily Screen Time (Hours)')
plt.ylabel('Notifications per Day')

# Plot 2: Regression (Actual vs Predicted Sleep)
plt.subplot(1, 2, 2)
reg_preds = knn_reg.predict(X_test_scaled)
plt.scatter(y_test_r, reg_preds, color='teal', alpha=0.3)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2) # Diagonal line
plt.title('Sleep Hours: Actual vs Predicted')
plt.xlabel('Actual Sleep Hours')
plt.ylabel('Predicted Sleep Hours')

plt.tight_layout()
plt.show()

# --- PREDICTION FOR SAMPLE USER  ---
# We use a DataFrame with column names to keep the Scaler happy
sample_user = pd.DataFrame([[8, 4, 2, 150, 1, 1]], columns=features)
sample_scaled = pd.DataFrame(scaler.transform(sample_user), columns=features)

pred_label = knn_clf.predict(sample_scaled)
pred_sleep = knn_reg.predict(sample_scaled)

print(f"Classification Accuracy: {accuracy_score(y_test, class_preds := knn_clf.predict(X_test_scaled)) * 100:.2f}%")
print(f"Regression Mean Error: {mean_absolute_error(y_test_r, reg_preds):.2f} hours")
print(f"\nSample User Results:\nAddicted: {'Yes' if pred_label[0] == 1 else 'No'}\nPredicted Sleep: {pred_sleep[0]:.1f} hours")