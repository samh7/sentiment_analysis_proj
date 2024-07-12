import pickle  # For model persistence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Assuming LogisticRegression model

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    # Consider additional steps like stemming/lemmatization
    return text

new_text = "This movie is fantastic!"
new_text_preprocessed = preprocess_text(new_text)

# Load the trained model (replace with your actual filename)
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)

vectorizer = TfidfVectorizer(max_features=2000)  # Adjust max_features if needed

# Transform the new text
new_text_features = vectorizer.transform([new_text_preprocessed])

# Predict sentiment
sentiment_prediction = model.predict(new_text_features)[0]
print(f"Predicted sentiment for '{new_text}': {sentiment_prediction}")
