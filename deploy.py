import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
nltk.download('stopwords')

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
print(df.head())

corpus = []
ps = PorterStemmer()

for i in range(0,df.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]',repl=' ',string=df.message[i])
    message = message.lower()
    words = message.split()

    # Remove the stop words
    words = [w for w in words if w not in set(stopwords.words('english'))]

    # Stemming
    words = [ps.stem(w) for w in words]
    message = ' '.join(words)
    corpus.append(message)

# BAG OF WORDS

cv = CountVectorizer(max_features=2500)

# Independent Variable
X = cv.fit_transform(corpus).toarray()

# Dependent Variable
y = pd.get_dummies(df['label']).iloc[:,1].values

# Creating a Pickle file for Count Vectorizer
pickle.dump(cv,open('cv.pkl','wb'))

# Create a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set

classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Naive Bayes model
filename = 'naive_bias_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))



