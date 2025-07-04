import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import sys
import os

# Verify environment
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Check if dataset exists
dataset_path = 'IMDB_Dataset.csv'
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found in {os.getcwd()}. Please download it from Kaggle and place it in the project folder.")
    sys.exit(1)

# Load dataset
try:
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

X = data['review']
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Plot accuracy
plt.plot([1], [accuracy], 'ro')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
try:
    plt.savefig('accuracy_plot.png')
    print("Accuracy plot saved as accuracy_plot.png")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.close()

# NER with spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    text = "Apple is launching a new product in San Francisco next month."
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Named Entities:", entities)
except Exception as e:
    print(f"Error with spaCy: {e}. Ensure 'en_core_web_sm' is installed.")