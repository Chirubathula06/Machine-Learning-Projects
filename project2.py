# Spam Classification using Naive Bayes (Final Version)

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Create Improved Dataset
data = {
    "Message": [
        "Win money now",
        "Hello friend how are you",
        "Get free lottery tickets",
        "Important meeting tomorrow",
        "Earn cash quickly",
        "Let's have lunch",
        "Free gift card available",
        "Project submission deadline",
        "Congratulations you won prize",
        "Call me later",
        "Limited offer buy now",
        "Are you coming today",
        "Claim your free reward",
        "Meeting rescheduled",
        "Urgent response needed",
        "Dinner tonight?"
    ],
    "Label": [
        "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)

# Step 3: Define Features and Target
X = df["Message"]
y = df["Label"]

# Step 4: Split Data (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Convert Text to Numerical (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predict
y_pred = model.predict(X_test_vec)

# Step 8: Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=1))

# Step 9: Test with New Message
new_msg = ["Congratulations! You have won a free iPhone"]
new_msg_vec = vectorizer.transform(new_msg)

prediction = model.predict(new_msg_vec)
print("\nNew Message:", new_msg[0])
print("Prediction:", prediction[0])
