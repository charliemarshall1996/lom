import logging
import concurrent.futures
import itertools

from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel, CoherenceModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from clean import clean_lyrics

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class Pipe:

    def __init__(self, kept_pos=['N']):

        # Clean Kwargs
        self.clean_params = {
            'kept_pos': kept_pos,
            'remove_sqaure_brackets': True,
            'remove_brackets': True,
            'remove_adlibs': False,
            'remove_min_len_adlibs': False,
            'remove_punctuation': True,
            'map_vocab': True,
            'lemmatize': True,
            'filter_pos': True,
            'remove_stopwords': True,
        }

        # Vectorizer
        self.vec = TfidfVectorizer(lowercase=False)

        self.model = None
        self.corpus = None
        self.dictionary = None
        self.dtm = None
        self.vocab = None
        self.X = None
        self.texts = None

        self.all_params = {
            'kept_pos': kept_pos,
            'remove_sqaure_brackets': True,
            'remove_brackets': True,
            'remove_adlibs': False,
            'remove_min_len_adlibs': False,
            'remove_punctuation': True,
            'map_vocab': True,
            'lemmatize': True,
            'filter_pos': True,
            'remove_stopwords': True,
        }

    def run(self, data):
        X, texts = clean_lyrics(data, **self.clean_params)
        self.X = X
        self.texts = texts
        self.extract()
        self.train()
        return self.score()

    def extract(self):
        self.dtm = self.vec.fit_transform(self.X)
        self.vocab = self.vec.vocabulary_.items()
        self.corpus = Sparse2Corpus(self.dtm, documents_columns=False)
        self.dictionary = Dictionary.from_corpus(
            self.corpus, id2word={v: k for k, v in self.vocab})

    def train(self):
        self.model = LdaModel(
            corpus=self.corpus, id2word=self.dictionary, random_state=42, dtype=np.float64)

    def score(self):
        coherence_model = CoherenceModel(
            model=self.model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')

        self.all_params['coherence'] = coherence_model.get_coherence()


def fit(data, kept_pos):
    logger.info("Running with params:\n %s", kept_pos)

    pipe = Pipe(kept_pos=kept_pos)
    pipe.run(data)
    msg = ""
    for key, val in pipe.all_params.items():
        text = f"({key}={str(val)})"
        msg += text
    print(msg)
    return pipe.all_params


def run(data, genre):

    # Search Space
    pos = ['N', 'V', 'J', 'R']
    all_pos_combinations = []
    for r in range(1, 5):
        combinations = itertools.combinations(pos, r)
        all_pos_combinations.extend(combinations)

    pos_combinations = [list(combination)
                        for combination in all_pos_combinations]

    # Run
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_optimized = [executor.submit(
            fit, data, combination) for combination in pos_combinations]
        for future in concurrent.futures.as_completed(future_to_optimized):
            params = future.result()
            results.append(params)

    results = pd.DataFrame(results)
    results.to_csv(
        f"./data/optimization/preprocessing_{genre}_grid_search.csv", index=False)


if __name__ == "__main__":
    data = pd.read_csv("./data/raw/song_lyrics_sampled.csv")
    data = data.dropna(how="any")
    data = data.sample(n=1000).lyrics
    run(data, 'all')
