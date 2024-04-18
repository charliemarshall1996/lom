

import logging
import unicodedata

from nltk import pos_tag, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pandas as pd

from utils import read_json_mapping, txt_to_set

wn_pos = {'J': wordnet.ADJ, 'V': wordnet.VERB,
          'N': wordnet.NOUN, 'R': wordnet.ADV}

wn = WordNetLemmatizer()

stopwords = set(stopwords.words('english')).union(
    txt_to_set("../data/vocab/stopwords.txt"))


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopwords]


def pos_lemmatize(tokens):
    return [wn.lemmatize(token, wn_pos.get(tag[0], wordnet.NOUN)) for token, tag in tokens]


def filter_pos(tokens, kept_pos):
    return [(token, tag) for token, tag in tokens if tag[0] in kept_pos]


def map_vocab(tokens):

    contractions = read_json_mapping("../data/vocab/contractions.json")
    dropped_gs = read_json_mapping("../data/vocab/contractions.json")

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


def clean_lyrics(df: pd.DataFrame, **kwargs):

    # lowercase lyrics
    df.lyrics = df.lyrics.str.lower()

    # normalize encoding
    df.lyrics.apply(lambda x: unicodedata.normalize('NFKD', x))

    # remove square brackets
    if kwargs.get("remove_square_brackets", True):
        square_brackets_pattern = r"\[([^\[\]]*+(?:\[[^\[\]]*+])*+)\]"
        df.lyrics = df.lyrics.str.replace(
            square_brackets_pattern, "", regex=True)

    # remove regular brackets
    if kwargs.get("remove_regular_brackets", True):
        regular_brackets_pattern = r"\([^)]*\)"
        df.lyrics = df.lyrics.str.replace(
            regular_brackets_pattern, "", regex=True)

    # remove newline characters
    df.lyrics = df.lyrics.str.replace("\n", " ")

    # remove carriage return characters
    df.lyrics = df.lyrics.str.replace("\r", " ")

    # remove adlibs
    if kwargs.get("remove_adlibs", True):
        adlibs = {'ah', 'aw', 'anh', 'ay', 'ayo', 'ayoh', 'aye',
                  'br', 'da', 'dae', 'do', 'er', 'goh', 'he',
                  'ho', 'lad', 'ladi', 'ladium', 'li', 'm',
                  'mh', 'na', 'nah', 'naw', 'noh', 'nouh', 'sh', 'uh',
                  'woah', 'wo' 'h', 'wo', 'unh', 'uho',
                  'umah', 'yo', 'yuh'}
        adlibs_pattern = r'\b(?:' + '|'.join(adlibs) + r')\b'
        df.lyrics = df.lyrics.str.replace(adlibs_pattern, "", regex=True)

        minimum_length_adlibs = {('\b[he]{3,}\b'), ('\b[hey]{4,}\b'), ('\b[i]{2,}\b'), ('\b[la]{3,}\b'),
                                 ('\b[na]{3,}\b'), ('\b[no]{5,}\b'), ('\b[ops]{4,}\b'), ('\b[bra]{4,}\b')}
        minimum_length_adlibs_pattern = fr"{'|'.join([sub_pattern for sub_pattern in minimum_length_adlibs])}"  # noqa
        df.lyrics = df.lyrics.str.replace(
            minimum_length_adlibs_pattern, "", regex=True)

    # remove whitespace
    whitespace_pattern = r"\s{2,}"
    df.lyrics = df.lyrics.str.replace(whitespace_pattern, "", regex=True)
    df.lyrics = df.lyrics.str.strip()

    # remove punctuation
    punctuation_pattern = r"[^\w\s]"
    df.lyrics = df.lyrics.str.replace(punctuation_pattern, "", regex=True)

    # word tokenize
    df.lyrics = df.lyrics.apply(word_tokenize)

    # map vocabulary
    if kwargs.get("map_vocab", True):
        df.lyrics = df.lyrics.apply(map_vocab)

    # tag pos
    if kwargs.get("filter_pos", True) or kwargs.get("pos_lemmatize", True):
        df.lyrics = df.lyrics.apply(pos_tag)

    # pos filter
    if kwargs.get("filter_pos", True):
        df.lyrics = df.lyrics.apply(
            filter_pos, True, (kwargs.get("kept_pos", "N")))

        if not kwargs.get("pos_lemmatize", True):
            df.lyrics = df.lyrics.apply(remove_pos_tags)

    # pos lemmatize
    if kwargs.get("pos_lemmatize", True):
        df.lyrics = df.lyrics.apply(pos_lemmatize)

    # remove stopwords
    if kwargs.get("remove_stopwords", True):
        df.lyrics = df.lyrics.apply(remove_stopwords)

    # rejoin to string
    df.lyrics = df.lyrics.str.join(" ")

    return df
