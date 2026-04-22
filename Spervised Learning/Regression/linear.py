from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r'D:\python\Machine Learning\Spervised Learning\Linear Regression\Fish.csv')
#print(df.head())
X=df['Length1'].values.reshape(-1,1)
y=df['Weight'].values

model=LinearRegression()
model.fit(X,y)

new_fish_length=[[30]]
predicted_weight=model.predict(new_fish_length)
print(f"Predicted weight for 30cm Fish is : {predicted_weight[0]:.2f}")

# Visualization
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Fish Data')
plt.plot(X, model.predict(X), color='red', label='Linear Trend')
plt.scatter(new_fish_length, predicted_weight, color='green', marker='X', s=150, label='Our Prediction for given length')

plt.title('Fish Weight vs. Length')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()