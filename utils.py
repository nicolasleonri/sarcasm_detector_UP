import time
import math
import re
import json
import demoji

def time_since(since):
    """
    A helper to print the amount of time passed.
    """
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def preprocess_text(text):
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