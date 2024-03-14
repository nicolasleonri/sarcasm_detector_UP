from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import basic_preprocess_text
import tensorflow as tf
import pandas as pd
import numpy as np
import os


class LSTM():

    def __init__(self, train_file, test_file, epochs, hidden_size, n_layers, preprocessing_method="Basic"):
        # Hyperparameters
        self.train_file = train_file
        self.test_file = test_file
        self.epochs = epochs
        self.preprocessing_method = preprocessing_method
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def train(self):
        data = pd.read_csv(self.train_file)

        data['tweet'] = data['tweet'].astype(str)  # Convert to string type

        # Pre-process the data:
        if self.preprocessing_method == "Basic":
            data['tweet'] = data['tweet'].apply(basic_preprocess_text)
        # ToDo: An if-else statement as we try different preprocessing methods

        # Tokenize text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['tweet'])
        vocab_size = len(tokenizer.word_index) + 1

        X = tokenizer.texts_to_sequences(data['tweet'])
        X = pad_sequences(X, padding='post')

        # Create train-test split
        y = np.array(data['sarcastic'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Define LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size, 100, input_length=X.shape[1]),
            tf.keras.layers.LSTM(self.hiddensize),
            tf.keras.layers.Dense(self.n_layers, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        # Define early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)

        # Train the model
        history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(
            X_test, y_test), callbacks=[early_stopping])

        # Save model
        filename = "Model_{}_{}_{}_{}_{}.keras".format(
            self.epochs, self.hiddensize, self.n_layers, self.preprocessing_method)
        filename = os.path.join('./results/', filename)
        model.save(filename)

        shape_size = X.shape[1]

        return model, history, shape_size
