import argparse
import time

import pandas as pd
from model.model import LSTM
from utils import time_since, plot_loss, plot_confusion_matrix
from language_model import predict_sarcasm
from sklearn.metrics import confusion_matrix


"""
Comments: we are using tensorflow instead of pytorch. We also should use pytorch
"""

def main():

    parser = argparse.ArgumentParser(
        description='Train LSTM model'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--evaluate', dest='evaluate',
        help='Evaluates the model using f1 score',
        action='store_true'
    )

    args = parser.parse_args()

    # We have 1 unique dataset, that we can vary.
    train_file = './data/dataset_train.csv'
    test_file = './data/dataset_test.csv'

    if args.default_train:
        decoder = LSTM(train_file, test_file, 1000, 128, 1)

        start = time.time()

        model, history, _ = decoder.train()

        print("Time needed to train:", time_since(start))

        plot_loss(history) # ToDo: maybe save results automatically as well

        # Basic inference
        sentences = ["I'm soooo excited! #sarcasm",
                     "This is the best day ever!"]
        predictions = predict_sarcasm(model, 0.65, sentences)
        
        print(predictions)

    """
    ToDo: Add different kinds of trainings using:
        a) Different pre-processing
        b) Different combination of labels (by changing dataset)
    """

    if args.evaluate: # ToDo: Make it obligatory to need a training argument
        sentences = pd.read_csv(test_file)
        sentences['Predicted_sarcastic'] = pd.DataFrame(
            predict_sarcasm(model, 0.65, sentences.iloc[:, 0]))
        sentences.sarcastic.value_counts(dropna = False, normalize= True).reset_index()
        sentences.Predicted_sarcastic.value_counts(dropna = False, normalize= True).reset_index()
        sentences.Predicted_sarcastic.value_counts(dropna = False).reset_index()

        correct = sentences[sentences.sarcastic==sentences.Predicted_sarcastic]
        accuracy = correct.shape[0] / sentences.shape[0]
        print(round(accuracy, 2),"%")
        wrong = sentences[sentences.sarcastic!=sentences.Predicted_sarcastic]
        sentences[['sarcastic', 'Predicted_sarcastic']].value_counts().reset_index()

        cm = confusion_matrix(sentences['sarcastic'], sentences['Predicted_sarcastic'])
        plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()
