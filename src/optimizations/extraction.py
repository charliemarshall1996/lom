
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import LdaModel, CoherenceModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args

from src.clean.clean import clean_lyrics

search_space = [Categorical(['tfidf', 'bow'], name="method"),
                Integer(1, 100, prior="log-uniform", name="num_topics")]


def tfidf_extract(lyrics):
    # initialize vectorizer
    vec = TfidfVectorizer(lowercase=False)

    # Get doc-term matrix
    dtm = vec.fit_transform(lyrics)

    # Build corpus
    corpus = Sparse2Corpus(dtm, documents_columns=False)

    # Build dictionary from corpus and vocab
    vocab = vec.vocabulary_.items()
    dictionary = Dictionary.from_corpus(
        corpus, id2word={v: k for k, v in vocab})

    return dictionary, corpus


def bow_extract(texts):

    # Create dictionary
    dictionary = Dictionary(texts)

    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # Return
    return dictionary, corpus


def run():

    # Import data
    data = pd.read_csv("./data/raw/song_lyrics_sampled.csv")
    data = data.sample(n=10000)

    # Prepare lyrics data
    lyrics = data.lyrics
    lyrics, text = clean_lyrics(
        lyrics, remove_adlibs=False, remove_min_len_adlibs=False)

    # Initialize KFold
    kf = KFold(n_splits=11, shuffle=True, random_state=42)
    kf.get_n_splits(lyrics, text)

    # Define objective function
    def objective(**params):

        # Initialize results dictionary
        results = params.copy()

        # Initialize coherences list
        coherences = []

        # Iterate over splits
        for i, (train_i, _) in enumerate(kf.split(lyrics, text)):

            # Get lyrics & Text for splits
            train_lyrics = [lyrics[i] for i in train_i]
            train_text = [text[i] for i in train_i]

            print(f"Scoring folds {i+1}/11...")

            # Perform feature extraction based on params
            if params.get("method") == "tfidf":
                dictionary, corpus = tfidf_extract(train_lyrics)

            elif params.get("method") == "bow":
                dictionary, corpus = bow_extract(train_text)

            model = LdaModel(corpus=corpus, id2word=dictionary,
                             num_topics=params.get("num_topics"))

            # Score model
            coherence_model = CoherenceModel(
                model, texts=train_text, dictionary=dictionary)
            score = coherence_model.get_coherence()
            print(f"Score received for fold {i+1}: {score}\n")
            print(f"Params:\n-------")
            print(f"method: {params['method']}\nnum_topics: {params['num_topics']}\n")  # noqa

            # Append score to coherences
            coherences.append(score)

            # Add fold score to results
            results[f'fold {i+1} score'] = score

        # Add mean_score variable to observation
        results['mean score'] = np.mean(coherences)

        # Return results dict
        return results

    # Define search space
    space = {
        'methods': ['tfidf', 'bow'],
        'num_topics': [1, 50, 100]
    }

    # Iterate over search space
    results_list = []
    for method in space["methods"]:
        for num_topics in space["num_topics"]:
            results_list.append(
                objective(method=method, num_topics=num_topics))

    # Extract results into a DataFrame
    params_df = pd.DataFrame(results_list)
    params_df.to_csv("./data/optimization/extraction_optimization.csv")


if __name__ == "__main__":
    run()
