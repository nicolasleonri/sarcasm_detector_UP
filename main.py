import argparse
import time

from utils import time_since
from model.model import LSTM

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )
    
    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    args = parser.parse_args()

    train_file = './data/dataset_train.csv'
    test_file = './data/dataset_test.csv'

    if args.default_train:

        # Training and Testing files as separated inputs, so we can change them and test different labels easier
        decoder = LSTM(train_file, test_file, 3000, 0.35)

        # print_every = 100
        # plot_every = 10

        start = time.time()

        decoder.train()

        print("Running time:", time_since(start))

if __name__ == "__main__":
    main()