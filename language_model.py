from utils import basic_preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inference method


def predict_sarcasm(decoder, threshold, sentences):
    # Preprocess input sentences
    preprocessed_sentences = [basic_preprocess_text(
        sentence) for sentence in sentences]

    model, _, shape_size = decoder

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    sequences = tokenizer.texts_to_sequences(preprocessed_sentences)
    padded_sequences = pad_sequences(
        sequences, padding='post', maxlen=shape_size)

    # Predict
    predictions = model.predict(padded_sequences)

    # Apply threshold
    binary_predictions = (predictions > threshold).astype(int)

    return binary_predictions
