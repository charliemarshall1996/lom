
import logging

from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel, CoherenceModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from clean import clean_lyrics

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def run(data, genre):

    # Create search space
    logger.info("Initializing search space...")
    space = [Integer(1000, len(data), prior='log-uniform', name="samples")]

    # Define objective function
    @use_named_args(space)
    def lda_optimizer(**params):

        # Prepare lyrics
        lyrics = data.sample(n=params.get('samples')).lyrics
        X, cleaned_lyrics = clean_lyrics(
            lyrics, remove_adlibs=False, remove_min_len_adlibs=False)

        # Extract lyric features
        vec = TfidfVectorizer(lowercase=False)
        dtm = vec.fit_transform(X)
        vocab = vec.vocabulary_.items()
        corpus = Sparse2Corpus(dtm, documents_columns=False)
        dictionary = Dictionary.from_corpus(
            corpus, id2word={v: k for k, v in vocab})

        model = LdaModel(corpus=corpus, id2word=dictionary,
                         dtype=np.float64, alpha='auto', random_state=42)
        coherence_model = CoherenceModel(
            model=model, texts=cleaned_lyrics, dictionary=dictionary)
        return -coherence_model.get_coherence()

    # Run optimizer
    logger.info("Running optimizer for %s...", genre)
    result = gp_minimize(lda_optimizer, space, n_calls=20,
                         random_state=0, verbose=True, initial_point_generator='lhs', n_jobs=3)

    # Extract results into a DataFrame

    params_df = pd.DataFrame(result.x_iters, columns=[
                             dim.name for dim in space])
    params_df['score'] = -result.func_vals

    # Save to .csv
    params_df.to_csv(f"./data/optimization/sample_size_{genre}_optimized.csv")


if __name__ == "__main__":

    df = pd.DataFrame()
    for chunk in pd.read_csv("./data/raw/song_lyrics.csv", encoding='utf-8-sig', encoding_errors='ignore', chunksize=100000, usecols=['tag', 'lyrics', 'language_cld3']):
        df = pd.concat([df, chunk])
    df = df[df['language_cld3'] == 'en']
    df = df.dropna(how="any")
    for genre in ['country', 'rap', 'rb', 'rock', 'pop']:
        data = df[df['tag'] == genre]
        run(data, genre)
