# -*- coding: utf-8 -*-
import pytest
from keras_wrapper.utils import *

def test_checkParameters():
    # TODO
    pass


def test_bbox():
    # TODO
    pass


def test_build_OneVsOneECOC_Stage():
    # TODO
    pass


def test_build_OneVsAllECOC_Stage():
    # TODO
    pass


def test_build_Specific_OneVsOneECOC_Stage():
    # TODO
    pass


def test_build_Specific_OneVsOneVsRestECOC_Stage():
    # TODO
    pass


def test_build_Specific_OneVsOneECOC_loss_Stage():
    # TODO
    pass


def test_prepareECOCLossOutputs():
    # TODO
    pass


def test_loadGoogleNetForFood101():
    # TODO
    pass


def test_prepareGoogleNet_Food101():
    # TODO
    pass


def test_prepareGoogleNet_Food101_ECOC_loss():
    # TODO
    pass


def test_prepareGoogleNet_Food101_Stage1():
    # TODO
    pass


def test_prepareGoogleNet_Stage2():
    # TODO
    pass


def test_simplifyDataset():
    # TODO
    pass


def test_average_models():
    # TODO
    pass


def test_one_hot_2_indices():
    # TODO
    pass


def test_indices_2_one_hot():
    # TODO
    pass


def test_to_categorical():
    # TODO
    pass


def test_categorical_probas_to_classes():
    # TODO
    pass


def test_decode_predictions_one_hot():
    # TODO
    pass


def test_decode_predictions():
    # TODO
    pass


def test_decode_multilabel():
    # TODO
    pass


def test_replace_unknown_words():
    # TODO
    pass


def test_decode_predictions_beam_search():
    # TODO
    pass


def test_sample():
    # TODO
    pass


def test_sampling():
    scores = [0.06, 0.1, 0.04, 0.4, 0.3, 0.3]
    sampled_idx = sampling(scores, sampling_type='max_likelihood', temperature=1)
    assert sampled_idx == 3

def test_flatten_list_of_lists():
    list_of_lists = [[1, 2, 3], [4, 5], [6]]
    flatten_list = flatten(list_of_lists)
    desired_list = [1, 2, 3, 4, 5, 6]
    assert flatten_list == desired_list


def test_flatten():
    list_of_lists = [[1, 2, 3], [4, 5], [6]]
    flatten_list = flatten(list_of_lists)
    desired_list = [1, 2, 3, 4, 5, 6]
    assert flatten_list == desired_list

    nested_list_of_lists = [[[[[1, 2]], 3]], [[4], [[[5]]]], [[[6]]]]
    flatten_list = flatten(nested_list_of_lists)
    desired_list = [1, 2, 3, 4, 5, 6]
    assert flatten_list == desired_list

def test_key_with_max_val():
    test_dict = {1: 1, 2: 2, 3: 4, 4: 1, 5: 4444}
    key = key_with_max_val(test_dict)
    assert key == 5

    test_dict = {1: 'a', 2: 'b', 3: 'z', 4: 1, 5: 4444}
    key = key_with_max_val(test_dict)
    assert key == 3


if __name__ == '__main__':
    pytest.main([__file__])
