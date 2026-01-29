from sklearn.feature_extraction.text import TfidfVectorizer

#input text(list of documents)

text = ["Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."]

#initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

#fit and transform the text
tfidf_matrix = vectorizer.fit_transform(text)

#extract keywords(feature name)
keywords = vectorizer.get_feature_names_out()

#print keywords
print("Extracted Keywords:", keywords)