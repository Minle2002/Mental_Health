import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('cleaned_dataset.csv')

def clean(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['selftext'] = df['selftext'].apply(clean)

X = df['selftext']
Y = df['subreddit']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=50)

vectorization = TfidfVectorizer()
X_Train_Vector = vectorization.fit_transform(X_Train)
X_Test_Vector = vectorization.transform(X_Test)

model = LogisticRegression(solver='sag')
model.fit(X_Train_Vector, Y_Train)

joblib.dump(vectorization, 'vectorizer.pkl')
joblib.dump(model, 'trained_model.pkl')