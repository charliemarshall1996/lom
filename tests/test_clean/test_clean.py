
import unittest
from src.clean.clean import *


class TestRemoveStopWords(unittest.TestCase):

    def test_remove_stop_words_happy(self):
        tokens = ["hello", "by", "world", "i"]
        expected = ["hello", "world"]
        actual = remove_stopwords(tokens)
        self.assertEqual(actual, expected)

    def test_remove_stop_words_just_stops(self):
        tokens = ["by", "i"]
        expected = []
        actual = remove_stopwords(tokens)
        self.assertEqual(actual, expected)

    def test_remove_stop_words_no_stops(self):
        tokens = ["hello", "world"]
        expected = ["hello", "world"]
        actual = remove_stopwords(tokens)
        self.assertEqual(actual, expected)

    def test_remove_stop_words_no_tokens(self):
        tokens = []
        expected = []
        actual = remove_stopwords(tokens)
        self.assertEqual(actual, expected)


class TestPosLemmatize(unittest.TestCase):

    def test_pos_lemmatize_valid_tags(self):
        tokens = [("running", "VBG"), ("cats", "NNS"), ("happier", "JJR")]
        expected_output = ["run", "cat", "happy"]
        output = pos_lemmatize(tokens)
        self.assertEqual(output, expected_output)

    def test_pos_lemmatize_invalid_tags(self):
        tokens = [("running", "ABC"), ("cats", "XYZ")]
        expected_output = ["running", "cat"]  # Default to NOUN
        output = pos_lemmatize(tokens)
        self.assertEqual(output, expected_output)

    def test_pos_lemmatize_empty_list(self):
        tokens = []
        expected_output = []
        output = pos_lemmatize(tokens)
        self.assertEqual(output, expected_output)


class TestFilterPOS(unittest.TestCase):

    def test_filter_pos_empty_list(self):
        tokens = []
        kept_pos = ['N']
        expected_output = []
        output = filter_pos(tokens, kept_pos)
        self.assertEqual(output, expected_output)

    def test_filter_pos_valid_pos_tags(self):
        tokens = [("running", "VBG"), ("cats", "NNS"), ("happier", "JJR")]
        kept_pos = ['N', 'V']
        expected_output = [("running", "VBG"), ("cats", "NNS")]
        output = filter_pos(tokens, kept_pos)
        self.assertEqual(output, expected_output)

    def test_filter_pos_empty_kept_pos(self):
        tokens = [("running", "VBG"), ("cats", "NNS"), ("happier", "JJR")]
        kept_pos = []
        expected_output = []
        output = filter_pos(tokens, kept_pos)
        self.assertEqual(output, expected_output)


class TestMapVocab(unittest.TestCase):

    def test_map_vocab_dropped_gs_and_contractions(self):
        tokens = ["doin", "didnt"]
        expected = ["doing", "did", "not"]
        actual = map_vocab(tokens)
        self.assertEqual(actual, expected)


class RemovePOSTags(unittest.TestCase):

    def test_remove_postags(self):
        tokens = [("running", "VBG"), ("cats", "NNS")]
        expected = ["running", "cats"]
        actual = remove_pos_tags(tokens)
        self.assertEqual(actual, expected)


class TestCleanLyrics(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
