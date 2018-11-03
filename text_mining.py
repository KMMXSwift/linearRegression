from nltk.corpus import stopwords
from string import punctuation
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

chapter_1 = open("chapter1", "r").read()

def only_not_empty(sentence):
    return sentence != ""

def preprocessing(sentence):
    return " ".join(["*number*" if word.isdigit() else word for word in ["".join([character for character in unidecode(word) if character not in punctuation]) for word in sentence.lower().split(" ") if word not in stopwords.words("english") and word != ""]])

sentences = list(filter(only_not_empty , chapter_1.split("\n")))
words = list(map(preprocessing, sentences))

print(words)

vectorizer = CountVectorizer()
vectorized_data = vectorizer.fit_transform(words)

tf_idf_vectorizer = TfidfVectorizer()
tf_idf_data = tf_idf_vectorizer.fit_transform(words)
