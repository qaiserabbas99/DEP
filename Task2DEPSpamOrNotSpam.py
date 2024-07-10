
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

file_path = 'D:\Intership\spam_or_not_spam\spam_or_not_spam.csv'
df = pd.read_csv(file_path)

df['email'] = df['email'].astype(str)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

df['email'] = df['email'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['email']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

nb_model = MultinomialNB()
svm_model = SVC(kernel='linear')

nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

print("Naive Bayes Model")
print(classification_report(y_test, nb_pred))
print("Accuracy:", accuracy_score(y_test, nb_pred))

print("SVM Model")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))

print("Confusion Matrix for Naive Bayes")
print(confusion_matrix(y_test, nb_pred))

print("Confusion Matrix for SVM")
print(confusion_matrix(y_test, svm_pred))
