import argparse
import json
import os

# for python2
# import model
# for python3
from . import model

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--bucket',
        required=True)
    PARSER.add_argument(
        '--train_path',
        required=True)
    PARSER.add_argument(
        '--eval_path',
        required=True)
    PARSER.add_argument(
        '--learning_rate',
        default=.00001)
    PARSER.add_argument(
        '--num_epochs',
        default=200)
    PARSER.add_argument(
        '--train_batch_size',
        default=2)
    PARSER.add_argument(
        '--job-dir',
        default = 'junk')
    PARSER.add_argument(
        '--buffer_size',
        default=50)
    PARSER.add_argument(
        '--val_batch_size',
        default=2)
    PARSER.add_argument(
        '--max_steps',
        required = True)
    PARSER.add_argument(
        '--eval_steps',
        default=100)
    PARSER.add_argument(
        '--prefetch_buffer_size',
        default=500)
    PARSER.add_argument(
        '--model_path',
        required = True)

    ARGUMENTS, _ = PARSER.parse_known_args()
    model.train_and_evaluate(ARGUMENTS)
