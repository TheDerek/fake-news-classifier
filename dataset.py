import os
import pandas as pd
from collections import namedtuple
from pandas import read_csv


POLTIFACT = 'poltifact.csv'
POLTIFACT_TRAIN = 'poltifact_train.csv'
POLTIFACT_TEST = 'poltifact_test.csv'
ARTICLES = 'fake_or_real_news.csv'
IMPROVED_ARTICLES = 'improved_articles.csv'
current_path = os.path.dirname(__file__)

Corpus: ([str], [str]) = namedtuple('Corpus', 'data target')
Dataset = namedtuple('Dataset', 'data target labels')

def get_corpus(file_name: str) -> Corpus:
    path = os.path.join(current_path, 'data', file_name)
    data = read_csv(path)
    return Corpus(data.text.tolist(), pd.get_dummies(data.label).as_matrix())
