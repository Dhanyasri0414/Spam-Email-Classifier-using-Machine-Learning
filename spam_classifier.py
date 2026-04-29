# ================================
# Spam Email Classifier Project
# ================================

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (run once)
nltk.download('stopwords')

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")

# Rename columns (your dataset has v1 = label, v2 = message)
data = data.rename(columns={"v1": "label", "v2": "message"})

# Keep only required columns
data = data[["label", "message"]]

print("Dataset shape:", data.shape)
print(data.head())

# -------------------------------
# 2. Text Preprocessing Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Apply cleaning
data["clean_message"] = data["message"].apply(clean_text)

# -------------------------------
# 3. Convert Labels to Numbers
# spam = 1, ham = 0
# -------------------------------
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# -------------------------------
# 4. Feature Extraction (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_message"])
y = data["label_num"]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Train Model (Naive Bayes)
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# 7. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 9. Test with Custom Email
# -------------------------------
def predict_spam(email_text):
    email_text = clean_text(email_text)
    email_vector = vectorizer.transform([email_text])
    result = model.predict(email_vector)
    if result[0] == 1:
        return "SPAM"
    else:
        return "NOT SPAM"

# Example test
sample_email = "Congratulations! You won 1 lakh rupees. Click now"
print("\nSample Email:", sample_email)
print("Prediction:", predict_spam(sample_email))

# -------------------------------
# 10. User Input
# -------------------------------
while True:
    user_input = input("\nEnter an email message (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_spam(user_input))
