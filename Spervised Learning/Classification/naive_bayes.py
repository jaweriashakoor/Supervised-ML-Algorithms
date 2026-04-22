import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load data with the 'latin-1' encoding fix
path = r"D:\python\Machine Learning\Spervised Learning\Classification\spam.csv"
df = pd.read_csv(path, encoding='latin-1')

# 2. Clean: Select only the two columns we need
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']

# 3. Convert Text to Numbers (Vectorization)
cv = CountVectorizer()
X = cv.fit_transform(df['Message'])
y = df['Category']

# 4. Split and Train (using test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict a New Message
# Try changing this message to see the graph update!
my_message = ["WINNER! You have won a 1000 cash prize. Call now to claim!"]
message_vector = cv.transform(my_message)

# Get the probabilities (the % scores)
probs = model.predict_proba(message_vector)[0]
categories = model.classes_ # This is ['ham', 'spam']

# 6. Visualization
plt.figure(figsize=(8, 5))
plt.bar(categories, probs, color=['skyblue', 'salmon'])
plt.title(f'Prediction Probability for:\n"{my_message[0][:50]}..."')
plt.ylabel('Probability (0.0 to 1.0)')
plt.ylim(0, 1.1) # Set limit slightly higher for better view

# Add the percentage text on top of the bars
for i, prob in enumerate(probs):
    plt.text(i, prob + 0.02, f'{prob*100:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.show()

# Print results to the terminal as well
print(f"Message: {my_message[0]}")
print(f"Ham Probability: {probs[0]*100:.2f}%")
print(f"Spam Probability: {probs[1]*100:.2f}%")