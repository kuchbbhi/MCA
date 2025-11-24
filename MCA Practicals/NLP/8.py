# Lab 8 : Train a sentiment classifier (Naive Bayes, LSTM, or BERT) on IMDb reviews.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data = {
    "review": [
        "I loved this movie, it was fantastic!",
        "Terrible film. I will never watch again.",
        "A masterpiece, acting and story were brilliant.",
        "Waste of time, boring and predictable."
    ],
    "sentiment": ["pos", "neg", "pos", "neg"]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.25, random_state=42)
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)
print("\n--- Naive Bayes Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
lstm_model = Sequential()
lstm_model.add(Embedding(max_features, 128))
lstm_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation="sigmoid"))
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("\nTraining LSTM model (2 epochs for demo)...")
lstm_model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
loss, acc = lstm_model.evaluate(x_test, y_test, batch_size=64)
print("\n--- LSTM Results ---")
print("Test Accuracy:", acc)