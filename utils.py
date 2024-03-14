import time
import math
import re
import json
import demoji
import seaborn as sns
import matplotlib.pyplot as plt


def time_since(since):
    """
    A helper to print the amount of time passed.
    """
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def basic_preprocess_text(text):
    # Map contractions
    with open("./data/contractions.json", 'r') as file:
        contractions = json.load(file)

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Replace emojis with their meanings
    text = demoji.replace(text)

    return text


def plot_loss(history):
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={'size': 16}, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
