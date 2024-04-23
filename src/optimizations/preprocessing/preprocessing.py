
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

    def __init__(self, kept_pos=['N'], remove_sqaure_brackets=True, remove_brackets=True, remove_adlibs=True, remove_min_len_adlibs=True, remove_punctuation=True, map_vocab=True, lemmatize=True, filter_pos=True, remove_stopwords=True, max_features=10000, max_df=1, min_df=1, ngram_range=(1, 1), num_topics=5, passes=1, iterations=50, eta=None, decay=0.5, offset=1, gamma_threshold=0.001, min_probability=0.01, update_every=10000, eval_every=None, random_state=None):

        # Clean Kwargs
        self.clean_params = {
            'kept_pos': kept_pos,
            'remove_sqaure_brackets': remove_sqaure_brackets,
            'remove_brackets': remove_brackets,
            'remove_adlibs': remove_adlibs,
            'remove_min_len_adlibs': remove_min_len_adlibs,
            'remove_punctuation': remove_punctuation,
            'map_vocab': map_vocab,
            'lemmatize': lemmatize,
            'filter_pos': filter_pos,
            'remove_stopwords': remove_stopwords
        }

        # Vectorizer
        self.vec = TfidfVectorizer(
            max_features=max_features, max_df=max_df, min_df=min_df, ngram_range=ngram_range, lowercase=False)

        # Model params
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.gamma_threshold = gamma_threshold
        self.min_probability = min_probability
        self.update_every = update_every
        self.eval_every = eval_every
        self.random_state = random_state

        self.model = None
        self.corpus = None
        self.dictionary = None
        self.dtm = None
        self.vocab = None
        self.X = None
        self.texts = None

        self.all_params = {
            'kept_pos': kept_pos,
            'remove_sqaure_brackets': remove_sqaure_brackets,
            'remove_brackets': remove_brackets,
            'remove_adlibs': remove_adlibs,
            'remove_min_len_adlibs': remove_min_len_adlibs,
            'remove_punctuation': remove_punctuation,
            'map_vocab': map_vocab,
            'lemmatize': lemmatize,
            'filter_pos': filter_pos,
            'remove_stopwords': remove_stopwords,
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
        self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics,
                              passes=self.passes, iterations=self.iterations, eta=self.eta, decay=self.decay,
                              offset=self.offset, gamma_threshold=self.gamma_threshold,
                              minimum_probability=self.min_probability, update_every=self.update_every,
                              eval_every=self.eval_every, random_state=42, dtype=np.float64)

    def score(self):
        coherence_model = CoherenceModel(
            model=self.model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')

        self.all_params['coherence'] = coherence_model.get_coherence()


def fit(data, params):
    logger.info("Running with params:\n %s", params)
    pipe = Pipe(kept_pos=params[9], remove_sqaure_brackets=params[0], remove_punctuation=params[1], remove_stopwords=params[2], remove_brackets=params[3],
                remove_adlibs=params[4], remove_min_len_adlibs=params[5], map_vocab=params[6], lemmatize=params[7], filter_pos=params[8])
    pipe.run(data)
    msg = ""
    for key, val in pipe.all_params.items():
        text = f"({key}={str(val)})"
        msg += text
    print(msg)
    return pipe.all_params


def run_multiprocessed(data, genre):

    # Search Space
    remove_sqaure_brackets = [True, False]
    remove_punctuation = [True, False]
    remove_stopwords = [True, False]
    remove_brackets = [True, False]
    remove_adlibs = [True, False]
    remove_brackets = [True, False]
    remove_min_len_adlibs = [True, False]
    map_vocabulary = [True, False]
    lemmatize = [True, False]
    filter_pos = [True, False]

    pos = ['N', 'V', 'J', 'R']
    all_pos_combinations = []
    for r in range(1, 5):
        combinations = itertools.combinations(pos, r)
        all_pos_combinations.extend(combinations)

    pos_combinations = [list(combination)
                        for combination in all_pos_combinations]
    search_space = list(itertools.product(remove_sqaure_brackets, remove_punctuation, remove_stopwords, remove_brackets,
                        remove_adlibs, remove_min_len_adlibs, map_vocabulary, lemmatize, filter_pos, pos_combinations))

    # Run
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_optimized = [executor.submit(
            fit, data, params) for params in search_space]
        for future in concurrent.futures.as_completed(future_to_optimized):
            params = future.result()
            results.append(params)

    results = pd.DataFrame(results)
    results.to_csv(
        f"./data/optimization/preprocessing_{genre}_grid_search.csv", index=False)


def run(data, genre):

    # Search Space
    remove_sqaure_brackets = [True, False]
    remove_punctuation = [True, False]
    remove_stopwords = [True, False]
    remove_brackets = [True, False]
    remove_adlibs = [True, False]
    remove_brackets = [True, False]
    remove_min_len_adlibs = [True, False]
    map_vocabulary = [True, False]
    lemmatize = [True, False]
    filter_pos = [True, False]
    search_space = list(itertools.product(remove_sqaure_brackets, remove_punctuation, remove_stopwords, remove_brackets,
                        remove_adlibs, remove_min_len_adlibs, map_vocabulary, lemmatize, filter_pos))

    # Run
    results = []
    for params in search_space:
        result = fit(data, params)
        results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(
        f"./data/optimization/preprocessing_{genre}_grid_search.csv", index=False)


if __name__ == "__main__":
    data = pd.read_csv("./data/raw/song_lyrics_sampled.csv")
    data = data.dropna(how="any")
    data = data.sample(n=1000).lyrics
    run_multiprocessed(data, 'all')
