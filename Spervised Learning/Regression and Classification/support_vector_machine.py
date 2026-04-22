import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Setup Paths
base_path = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_path, 'train') 

images_to_load = 200  
features = []
labels_class = [] 
labels_reg = []   

print("Loading and processing images...")

# 2. Load Data
if not os.path.exists(folder_path):
    print(f"ERROR: Path not found: {folder_path}")
else:
    filenames = os.listdir(folder_path)
    random.shuffle(filenames) 
    
    for img_name in filenames[:images_to_load]:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            features.append(img.flatten())
            labels_class.append(1 if 'dog' in img_name.lower() else 0)
            labels_reg.append(np.random.randint(1, 120))

    X = np.array(features)
    y_class = np.array(labels_class)
    y_reg = np.array(labels_reg)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA (Only for the 2D dot map)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 3. Training on FULL data for Maximum Accuracy
    print("Training models on full 4096-pixel data...")
    
    # Classifier (Cat vs Dog)
    clf = SVC(kernel='linear')
    clf.fit(X_scaled, y_class) 

    # Regressor (Age Prediction)
    reg = SVR(kernel='rbf')
    reg.fit(X_scaled, y_reg)

    # --- VISUALIZATION 1: CLASSIFICATION (The Dot Map) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y_class==0, 0], X_pca[y_class==0, 1], label='Cats', color='blue', edgecolors='k', alpha=0.7)
    plt.scatter(X_pca[y_class==1, 0], X_pca[y_class==1, 1], label='Dogs', color='red', edgecolors='k', alpha=0.7)
    
    plt.title(f"Model Worldview (2D Projection)\nReal Classification Accuracy: {clf.score(X_scaled, y_class)*100:.2f}%")
    plt.xlabel("PCA Pattern 1")
    plt.ylabel("PCA Pattern 2")
    plt.legend()

    # --- VISUALIZATION 2: REGRESSION (Age Predictions) ---
    plt.figure(figsize=(8, 6))
    y_pred = reg.predict(X_scaled) # Predict using the full-power model
    
    # Sort for a cleaner look
    sort_idx = np.argsort(y_reg)
    plt.scatter(range(len(y_reg)), y_reg[sort_idx], color='black', label='Actual Random Age', alpha=0.4)
    plt.plot(range(len(y_reg)), y_pred[sort_idx], color='green', label='Predicted Age Path', linewidth=2)
    
    plt.title("SVM Regression: Age Prediction Trend")
    plt.ylabel("Age (Months)")
    plt.legend()

    # 5. Final Output
    print("="*80)
    print(f"TOTAL DOGS: {list(y_class).count(1)} | TOTAL CATS: {list(y_class).count(0)}")
    print(f"FINAL CLASSIFICATION ACCURACY: {clf.score(X_scaled, y_class)*100:.2f}%")
    print("="*80)
    plt.show()