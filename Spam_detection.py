from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Training data
messages = [
    "win a free iphone now",
    "meeting scheduled at 10AM",
    "claim your lottery prize",
    "project discussion tomorrow"
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 0, 1, 0]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Test message
text_message = ["Free prize waiting for you"]

# Vectorize test message
test_vector = vectorizer.transform(text_message)

# Predict
prediction = model.predict(test_vector)

# Output
if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")