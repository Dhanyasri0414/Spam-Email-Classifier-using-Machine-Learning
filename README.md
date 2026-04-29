# 📧 Spam Email Classifier using Machine Learning

A Python-based Machine Learning project that classifies email or SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **TF-IDF Vectorization** with the **Naive Bayes algorithm**.  
This project helps in detecting unwanted, promotional, or fraudulent messages automatically with high accuracy. Built from your implementation using dataset preprocessing, TF-IDF feature extraction, and Multinomial Naive Bayes. :contentReference[oaicite:0]{index=0}

---

## 🚀 Project Overview
Spam emails and messages are one of the biggest challenges in digital communication. This project uses machine learning to analyze text patterns and predict whether a message is spam or legitimate.

The system:
- Cleans and preprocesses email/message text
- Removes stopwords and unwanted characters
- Converts text into numerical vectors using TF-IDF
- Trains a Naive Bayes classifier
- Predicts spam or ham for new custom messages

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- **TF-IDF Vectorizer**
- **Multinomial Naive Bayes**
- **Regex**

---

## 📂 Dataset
- Dataset File: `spam.csv`
- Main columns used:
  - `v1` → Label (`spam` / `ham`)
  - `v2` → Message Content

---

## ⚙️ Workflow
### 1️⃣ Data Loading
- Reads dataset using Pandas
- Renames columns for clarity

### 2️⃣ Text Preprocessing
- Converts text to lowercase
- Removes special characters
- Removes stopwords using NLTK

### 3️⃣ Label Encoding
- Spam → `1`
- Ham → `0`

### 4️⃣ Feature Extraction
- Uses **TF-IDF Vectorizer** to transform text into machine-readable numerical format

### 5️⃣ Model Training
- Splits data into training and testing sets
- Trains using **Multinomial Naive Bayes**

### 6️⃣ Evaluation
- Accuracy Score
- Classification Report
- Precision / Recall / F1-Score

### 7️⃣ Prediction
- Predicts custom user input messages in real-time

---

## 📌 Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
