import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

#Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

#Input text
text = """Arttificial Intelligence is transforming industries across the globe.
It is used in healthcare, finance, transportation, and many other sectors.
AI impproves efficiency and accuracy"""


#sentence tokenization
sentences = sent_tokenize(text)

#word tokenization
words = word_tokenize(text.lower())

#stop words removal
stop_words = set(stopwords.words("english"))

#word frequency dictionary
word_frequencies = {}

for word in words:
    if word.isalnum() and word not in stop_words:
        if word not in word_frequencies:
            word_frequencies[word] =1

    #Sentence scoring
sentence_scores = {}

for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in word_frequencies:
            if sentence not in sentence_scores:
                sentence_scores[sentence] = word_frequencies[word]
            else:
                sentence_scores[sentence] += word_frequencies[word]

#select the best sentence
summary = max(sentence_scores, key=sentence_scores.get)

print("Original summary:", text)
print("\n Extracted summary:", summary)