import time

from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import numpy as np
import pandas as pd

from src.utils import load_genre_dataset

# Extract features


def bow_extract(texts):

    # Create dictionary
    dictionary = Dictionary(texts)

    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # Return
    return dictionary, corpus


def train_baseline_model(genre, dct, corpus, lyrics):

    print(f"Training baseline {genre} model...")

    baseline_data = {'genre': genre}

    start = time.time()
    model = LdaModel(corpus=corpus, id2word=dct,
                     num_topics=20, dtype=np.float64)
    end = time.time()

    print(f"Trained {genre} model after {end-start}...")

    model.save(f"./models/full/baseline/{genre}_baseline.model")

    baseline_data['time'] = end - start

    coh_model = CoherenceModel(model, texts=lyrics, dictionary=dct)
    baseline_data['coherence'] = coh_model.get_coherence()
    baseline_data['perplexity'] = model.log_perplexity(corpus)
    print(f"{genre} baseline performance: ", baseline_data)
    return baseline_data


if __name__ == "__main__":
    baselines = []
    for genre in ['country', 'rock', 'rb', 'rap', 'pop']:
        lyrics = load_genre_dataset(genre)
        dct, corpus = bow_extract(lyrics)
        baselines.append(
            train_baseline_model(genre, dct, corpus, lyrics))

    baseline_df = pd.DataFrame(baselines)
    baseline_df.to_csv(
        "./data/optimization/full/baseline/baseline_performances.csv")
