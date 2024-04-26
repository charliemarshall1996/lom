

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

stopwords = set(stopwords.words('english')).union(
    txt_to_set("./data/vocab/stopwords.txt"))


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopwords]


def pos_lemmatize(tokens):
    return [wn.lemmatize(token, wn_pos.get(tag[0], wordnet.NOUN)) for token, tag in tokens]


def filter_pos(tokens, kept_pos):
    return [(token, tag) for token, tag in tokens if tag[0] in kept_pos]


def map_vocab(tokens):

    contractions = read_json_mapping("./data/vocab/contractions.json")
    dropped_gs = read_json_mapping("./data/vocab/dropped_gs.json")

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


def clean_lyrics(series: pd.DataFrame, **kwargs):

    # lowercase lyrics
    logger.info("Lowercasing lyrics...")
    series = series.str.lower()

    series.dropna(how="any", inplace=True)
    # normalize encoding
    logger.info("Normalizing unicode...")
    series.apply(lambda x: unicodedata.normalize('NFKD', x))

    # remove square brackets
    if kwargs.get("remove_square_brackets", True):
        logger.info("Removing square brackets...")
        square_brackets_pattern = r"\[([^\[\]]*+(?:\[[^\[\]]*+])*+)\]"
        series = series.str.replace(
            square_brackets_pattern, "", regex=True)

    # remove regular brackets
    if kwargs.get("remove_regular_brackets", True):
        logger.info("Removing regular brackets...")
        regular_brackets_pattern = r"\([^)]*\)"
        series = series.str.replace(
            regular_brackets_pattern, "", regex=True)

    # remove newline characters
    logger.info("Removing newline characters...")
    series = series.str.replace("\n", " ")

    # remove carriage return characters
    logger.info("Removing carriage return characters...")
    series = series.str.replace("\r", " ")

    # remove adlibs
    if kwargs.get("remove_adlibs", True):
        logger.info("Removing adlibs...")
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
    logger.info("Removing whitespace...")
    whitespace_pattern = r"\s{2,}"
    series = series.str.replace(whitespace_pattern, " ", regex=True)
    series = series.str.strip()

    # remove punctuation
    logger.info("Removing punctuation...")
    punctuation_pattern = r"[^\w\s]"
    series = series.str.replace(punctuation_pattern, "", regex=True)

    # word tokenize
    logger.info("Tokenizing lyrics...")
    series = series.apply(word_tokenize)

    # map vocabulary
    if kwargs.get("map_vocab", True):
        logger.info("Mapping vocabulary...")
        series = series.apply(map_vocab)

    # tag pos
    if kwargs.get("filter_pos", True) or kwargs.get("pos_lemmatize", True):
        logger.info("Tagging part-of-speech...")
        series = series.apply(pos_tag)

    # pos filter
    if kwargs.get("filter_pos", True):
        logger.info("Filtering part-of-speech...")
        series = series.apply(filter_pos, args=(kwargs.get("kept_pos", "N"),))

        if not kwargs.get("pos_lemmatize", True):
            series = series.apply(remove_pos_tags)

    # pos lemmatize
    if kwargs.get("pos_lemmatize", True):
        logger.info("Lemmatizing part-of-speech...")
        series = series.apply(pos_lemmatize)

    # remove stopwords
    if kwargs.get("remove_stopwords", True):
        logger.info("Removing stopwords...")
        series = series.apply(remove_stopwords)

    # rejoin to string
    joined_lyrics = series.str.join(" ").to_list()

    tokenized_lyrics = series.to_list()

    logger.info("Finished cleaning lyrics!")
    return joined_lyrics, tokenized_lyrics
