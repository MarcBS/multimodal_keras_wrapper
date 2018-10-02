# -*- coding: utf-8 -*-
import pytest
from six import iteritems
from keras_wrapper.extra.tokenizers import *


def test_tokenize_basic():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This , ¿ is a , . sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 , ! ! ! '
    tokenized_string = tokenize_basic(untokenized_string, lowercase=False)
    tokenized_string_lower = tokenize_basic(untokenized_string, lowercase=True)
    assert expected_string == tokenized_string
    assert expected_string.lower() == tokenized_string_lower


def test_tokenize_aggressive():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This is a sentence with weird\xbb symbols ù ä ë ï ö ü ^首先'
    tokenized_string = tokenize_aggressive(untokenized_string, lowercase=False)
    tokenized_string_lower = tokenize_aggressive(untokenized_string, lowercase=True)
    assert expected_string == tokenized_string
    assert expected_string.lower() == tokenized_string_lower


def test_tokenize_icann():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This , ¿is a , . sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 , ! '
    tokenized_string_lower = tokenize_icann(untokenized_string)
    assert expected_string.lower() == tokenized_string_lower


def test_tokenize_montreal():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This ¿is a sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 !!!'
    tokenized_string_lower = tokenize_montreal(untokenized_string)
    assert expected_string.lower() == tokenized_string_lower


def test_tokenize_soft():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This , ¿is a , . sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 , ! '
    tokenized_string = tokenize_soft(untokenized_string, lowercase=False)
    tokenized_string_lower = tokenize_soft(untokenized_string, lowercase=True)
    assert expected_string == tokenized_string
    assert expected_string.lower() == tokenized_string_lower


def test_tokenize_none():
    untokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!\n\n'
    expected_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!'
    tokenized_string = tokenize_none(untokenized_string)
    assert expected_string == tokenized_string


def test_tokenize_none_char():
    untokenized_string = u'This, ¿is a > <     , .sentence with weird\xbb symbols'
    expected_string = u'T h i s , <space> ¿ i s <space> a <space> > <space> < <space> , <space> . s e n t e n c e <space> w i t h <space> w e i r d \xbb <space> s y m b o l s'
    tokenized_string = tokenize_none_char(untokenized_string)
    assert expected_string == tokenized_string


def test_tokenize_CNN_sentence():
    # TODO
    pass


def test_tokenize_questions():
    # TODO
    pass


def test_tokenize_bpe():
    # TODO
    pass


def test_detokenize_none():
    tokenized_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!'
    expected_string = u'This, ¿is a      , .sentence with weird\xbb symbols ù ä ë ï ö ü ^首先 ,!!!'
    detokenized_string = detokenize_none(tokenized_string)
    assert expected_string == detokenized_string


def test_detokenize_none_char():
    tokenized_string = u'T h i s , <space> ¿ i s <space> a <space> > <space> < <space> , <space> . s e n t e n c e <space> w i t h <space> w e i r d \xbb <space> s y m b o l s'
    expected_string = u'This, ¿is a > < , .sentence with weird\xbb symbols'
    detokenized_string = detokenize_none_char(tokenized_string)
    assert expected_string == detokenized_string


if __name__ == '__main__':
    pytest.main([__file__])
