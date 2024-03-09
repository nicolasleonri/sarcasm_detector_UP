import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
from utils import preprocess_text

class LSTM():

    def __init__(self, train_file, test_file, epochs, threshold):
        #super(LSTM, self).__init__()
        
        # Hyperparameters
        self.train_file = train_file
        self.test_file = test_file
        self.epochs = epochs
        self.threshold = threshold

    def train(self):
        data = pd.read_csv(self.train_file)

        data['tweet'] = data['tweet'].astype(str)  # Convert to string type
        data['tweet'] = data['tweet'].apply(preprocess_text)

        # Tokenize text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['tweet'])
        vocab_size = len(tokenizer.word_index) + 1

        X = tokenizer.texts_to_sequences(data['tweet'])
        X = pad_sequences(X, padding='post')

        # Create train-test split
        y = np.array(data['sarcastic'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print(y)
