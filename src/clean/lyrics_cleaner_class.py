
import os
import logging
import unicodedata

from nltk import pos_tag, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pandas as pd

from src.utils import read_json_mapping, txt_to_set

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

wn_pos = {'J': wordnet.ADJ, 'V': wordnet.VERB,
          'N': wordnet.NOUN, 'R': wordnet.ADV}

wn = WordNetLemmatizer()


def remove_stopwords(tokens, stop_words):
    stop_words = set(stopwords.words('english')).union(stop_words)
    return [token for token in tokens if token not in stop_words]


def pos_lemmatize(tokens):
    return [wn.lemmatize(token, wn_pos.get(tag[0], wordnet.NOUN)) for token, tag in tokens]


def filter_pos(tokens, kept_pos):
    return [(token, tag) for token, tag in tokens if tag[0] in kept_pos]


def map_vocab(tokens, contractions, dropped_gs):

    # Initialize list of mapped
    # contractions
    mapped_contractions = []

    for token in tokens:

        # If the token is in the
        # CONTRACTIONS dictionary
        if token in contractions:

            # split the bi-gram and
            # add individual tokens
            # to mapped_contractions
            mapped_contractions.extend(contractions[token].split())

        else:  # If not a contraction
            # just add the token as it is to mapped_contractions
            mapped_contractions.append(token)

    # Map and return dropped g's
    return [dropped_gs.get(token, token) for token in mapped_contractions]


def remove_pos_tags(tokens):
    return [token for token, _ in tokens]


class LyricsCleaner:

    def __init__(self, stop_words, contractions, dropped_gs, verbose=False, **kwargs):
        self.stop_words = stop_words
        self.contractions = contractions
        self.dropped_gs = dropped_gs
        self.verbose = verbose
        self.kwargs = kwargs

    def clean_lyrics(self, series: pd.Series):
        if self.verbose:
            print("Starting cleaning...")

        # lowercase lyrics
        if self.verbose:
            print("Lowercasing lyrics...")
        series = series.str.lower()
        series.apply(lambda x: self.debug(x, "lowercase"))

        series.dropna(how="any", inplace=True)
        # normalize encoding
        if self.verbose:
            print("Normalizing unicode...")
        series.apply(lambda x: unicodedata.normalize('NFKD', x))
        series.apply

        # remove square brackets
        if self.kwargs.get("remove_square_brackets", True):
            if self.verbose:
                print("Removing square brackets...")
            square_brackets_pattern = r"\[([^\[\]]*+(?:\[[^\[\]]*+])*+)\]"
            series = series.str.replace(
                square_brackets_pattern, " ", regex=True)

        # remove regular brackets
        if self.kwargs.get("remove_regular_brackets", True):
            if self.verbose:
                print("Removing regular brackets...")
            regular_brackets_pattern = r"\([^)]*\)"
            series = series.str.replace(
                regular_brackets_pattern, " ", regex=True)

        # remove newline characters
        if self.verbose:
            print("Removing newline characters...")
        series = series.str.replace("\n", " ")

        # remove carriage return characters
        if self.verbose:
            print("Removing carriage return characters...")
        series = series.str.replace("\r", " ")

        # remove adlibs
        if self.kwargs.get("remove_adlibs", True):
            if self.verbose:
                print("Removing adlibs...")
            adlibs = {'ah', 'aw', 'anh', 'ay', 'ayo', 'ayoh', 'aye',
                      'br', 'da', 'dae', 'do', 'er', 'goh', 'he',
                      'ho', 'lad', 'ladi', 'ladium', 'li', 'm',
                      'mh', 'na', 'nah', 'naw', 'noh', 'nouh', 'sh', 'uh',
                      'woah', 'wo' 'h', 'wo', 'unh', 'uho',
                      'umah', 'yo', 'yuh'}
            adlibs_pattern = r'\b(?:' + '|'.join(adlibs) + r')\b'
            series = series.str.replace(adlibs_pattern, "", regex=True)

            minimum_length_adlibs = {('\b[he]{3,}\b'), ('\b[hey]{4,}\b'), ('\b[i]{2,}\b'), ('\b[la]{3,}\b'),
                                     ('\b[na]{3,}\b'), ('\b[no]{5,}\b'), ('\b[ops]{4,}\b'), ('\b[bra]{4,}\b')}
            minimum_length_adlibs_pattern = fr"{'|'.join([sub_pattern for sub_pattern in minimum_length_adlibs])}"  # noqa
            series = series.str.replace(
                minimum_length_adlibs_pattern, "", regex=True)

        # remove whitespace
        if self.verbose:
            print("Removing whitespace...")
        whitespace_pattern = r"\s{2,}"
        series = series.str.replace(whitespace_pattern, " ", regex=True)
        series = series.str.strip()

        # remove punctuation
        if self.verbose:
            print("Removing punctuation...")
        punctuation_pattern = r"[^\w\s]"
        series = series.str.replace(punctuation_pattern, "", regex=True)

        # word tokenize
        if self.verbose:
            print("Tokenizing lyrics...")
        series = series.apply(word_tokenize)

        # map vocabulary
        if self.kwargs.get("map_vocab", True):
            if self.verbose:
                print("Mapping vocabulary...")
            series = series.apply(map_vocab, args=(
                self.contractions, self.dropped_gs))

        # tag pos
        if self.kwargs.get("filter_pos", True) or self.kwargs.get("pos_lemmatize", True):
            if self.verbose:
                print("Tagging part-of-speech...")
            series = series.apply(pos_tag)

        # pos filter
        if self.kwargs.get("filter_pos", True):
            if self.verbose:
                print("Filtering part-of-speech...")
            series = series.apply(filter_pos, args=(
                self.kwargs.get("kept_pos", "N"),))

            if not self.kwargs.get("pos_lemmatize", True):
                series = series.apply(remove_pos_tags)

        # pos lemmatize
        if self.kwargs.get("pos_lemmatize", True):
            if self.verbose:
                print("Lemmatizing part-of-speech...")
            series = series.apply(pos_lemmatize)

        # remove stopwords
        if self.kwargs.get("remove_stopwords", True):
            if self.verbose:
                print("Removing stopwords...")
            series = series.apply(remove_stopwords, stop_words=self.stop_words)

        # rejoin to string
        joined_lyrics = series.str.join(" ").to_list()

        tokenized_lyrics = series.to_list()

        if self.verbose:
            print("Finished cleaning lyrics!")
        return joined_lyrics, tokenized_lyrics

    def debug(self, doc, place):

        if 'tearsand' in doc:
            print(f"Error found at: {place}")
