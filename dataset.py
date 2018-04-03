import os
import pandas as pd
from collections import namedtuple
from pandas import read_csv

POLITIFACT_TRAIN = 'politifact_train.csv'
POLITIFACT_TEST = 'politifact_test.csv'
ARTICLES_TRAIN = 'articles_train.csv'
ARTICLES_TEST = 'articles_test.csv'
MOVIE_REVIEWS_TRAIN = 'movie_reviews_train.csv'
MOVIE_REVIEWS_TEST = 'movie_reviews_test.csv'
current_path = os.path.dirname(__file__)

Corpus = namedtuple('Corpus', 'data target')
Dataset = namedtuple('Dataset', 'data target labels')


def get_corpus(path: str):
    # If the given path does not exist then try appending data/
    if not os.path.isfile(path):
        path = os.path.join(current_path, 'data', path)

    data = read_csv(path)
    return Corpus(data.text.tolist(), pd.get_dummies(data.label).as_matrix())
