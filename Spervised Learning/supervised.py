from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 

# 1. Prepare Data (Features: Square Footage)
X = np.array([[600], [800], [1000], [1200], [1500]]).reshape(-1, 1)

# 2. Labels (Price in thousands of dollars)
y = np.array([150, 200, 250, 300, 375])

# 3. Initialize the Model
model = LinearRegression()

# 4. Train the Model (The "Learning" Step)
model.fit(X, y)

# 5. Predict for a new house (1100 sq ft)
new_house = np.array([[1100]])
prediction = model.predict(new_house)

print(f"Predicted price for a 1100 sq ft house: ${prediction[0]:.2f}k")


# ==========================================
# --- 6. VISUALIZATION SECTION --- (ADDED)
# ==========================================

# Step A: Plot the original "training" data points
# We plot X (square footage) on the x-axis and y (prices) on the y-axis.
plt.scatter(X, y, color='blue', label='Actual Data')

# Step B: Generate the "line" the model learned.
# We predict prices for every square footage from 500 to 1600.
X_visual_range = np.linspace(500, 1600, 100).reshape(-1, 1)
y_learned_line = model.predict(X_visual_range)

# Step C: Plot the learned line (our model)
plt.plot(X_visual_range, y_learned_line, color='red', linewidth=2, label='Model (Regression Line)')

# Step D: (Optional but helpful) Visualise your prediction point
plt.scatter(new_house, prediction, color='green', marker='X', s=150, label='Our Prediction for 1100')

# Step E: Aesthetic Labeling
plt.title('House Price Prediction based on Square Footage')
plt.xlabel('Square Footage ($sq ft$)')
plt.ylabel('Price ($k$)')
plt.legend() # Shows which color means what
plt.grid(True, linestyle='--', alpha=0.5) # Adds a subtle grid

# Step F: Show or Save the plot
# plt.savefig('my_machine_learning_plot.png')
# print("\nPlot saved as: 'my_machine_learning_plot.png'"
plt.show()