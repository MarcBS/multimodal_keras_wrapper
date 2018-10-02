# -*- coding: utf-8 -*-
import pytest
from keras_wrapper.utils import *


def test_checkParameters():

    input_params = {'a': True, 'b': False, 'z': 16}
    default_params = {'a': False, 'b': False, 'c': 5}

    checked_params = checkParameters(input_params, default_params)
    assert checked_params['a']
    assert not checked_params['b']
    assert checked_params['c'] == 5
    assert checked_params.get('z') is None
    # Invalid configuration with hard_check
    try:
        _ = checkParameters(input_params, default_params, hard_check=True)
    except ValueError:
        assert True


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
    indices = np.array([2, 3, 5])
    one_hot = np.array([
        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    desired_indices = one_hot_2_indices(one_hot)
    assert np.all(desired_indices == indices)
    # Add some padding. It should be removed all but last
    indices = np.array([2, 3, 5, 0])
    one_hot = np.array([
        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    desired_indices = one_hot_2_indices(one_hot)
    assert np.all(desired_indices == indices)

    # We may also want to keep the padding
    indices = np.array([2, 3, 5, 0, 0, 0, 0, 0, 0])
    desired_indices = one_hot_2_indices(one_hot, pad_sequences=False)
    assert np.all(desired_indices == indices)


def test_indices_2_one_hot():
    indices = np.array([2, 3, 5])
    desired_one_hot = np.array(
        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    one_hot = indices_2_one_hot(indices, 16)
    assert np.all(desired_one_hot == one_hot)


def test_to_categorical():
    num_classes = 5
    shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
    expected_shapes = [(1, num_classes),
                       (3, num_classes),
                       (4, 3, num_classes),
                       (5, 4, 3, num_classes),
                       (3, num_classes),
                       (3, 2, num_classes)]
    labels = [np.random.randint(0, num_classes, shape) for shape in shapes]
    one_hots = [to_categorical(label, num_classes) for label in labels]
    for label, one_hot, expected_shape in zip(labels,
                                              one_hots,
                                              expected_shapes):
        # Check shape
        assert one_hot.shape == expected_shape
        # Make sure there are only 0s and 1s
        assert np.array_equal(one_hot, one_hot.astype(bool))
        # Make sure there is exactly one 1 in a row
        assert np.all(one_hot.sum(axis=-1) == 1)
        # Get original labels back from one hots
    assert np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)


def test_categorical_probas_to_classes():
    one_hot_probas = np.array(
        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    categorical_probas = categorical_probas_to_classes(one_hot_probas)
    desired_probas = np.array([2, 3, 5])
    assert np.all(categorical_probas == desired_probas)


def test_decode_predictions_one_hot():
    index2word = {
        0: u'<pad>',
        1: u'<unk>',
        2: u'This',
        3: u'is',
        4: u'a',
        5: u'text',
        6: u'file',
        7: u'.',
        8: u'Containing',
        9: u'characters',
        10: u'ẁñ',
        11: u'ü',
        12: u'^',
        13: u'首',
        14: u'先',
        15: u'，'
    }
    one_hot_preds = np.array([[[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

    decoded_preds = decode_predictions_one_hot(one_hot_preds, index2word)
    desired_preds = [u'This is text']
    assert desired_preds == decoded_preds


def test_decode_predictions():
    index2word = {
        0: u'<pad>',
        1: u'<unk>',
        2: u'This',
        3: u'is',
        4: u'a',
        5: u'text',
        6: u'file',
        7: u'.',
        8: u'Containing',
        9: u'characters',
        10: u'ẁñ',
        11: u'ü',
        12: u'^',
        13: u'首',
        14: u'先',
        15: u'，'
    }
    temperature = 1.
    preds = np.array([[0.1, 0.2, 0.7, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0.1, 0.2, 0.3, 0.7, 0.6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    decoded_preds = decode_predictions(preds, temperature, index2word, 'max_likelihood')
    desired_preds = [u'This', u'is', u'text']
    assert desired_preds == decoded_preds


def test_decode_multilabel():
    index2word = {
        0: u'<pad>',
        1: u'<unk>',
        2: u'This',
        3: u'is',
        4: u'a',
        5: u'text',
        6: u'file',
        7: u'.',
        8: u'Containing',
        9: u'characters',
        10: u'ẁñ',
        11: u'ü',
        12: u'^',
        13: u'首',
        14: u'先',
        15: u'，'
    }

    preds = [[0.1, 0.2, 0.7, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.1, 0.2, 0.3, 0.7, 0.6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]

    decoded_preds = decode_multilabel(preds, index2word, min_val=0.5)
    desired_preds = [[u'This'], [u'is', u'a'], [u'text']]
    assert desired_preds == decoded_preds
    decoded_preds, probs_pred = decode_multilabel(preds, index2word, min_val=0.5,
                                                  get_probs=True)
    desired_preds = [[u'This'], [u'is', u'a'], [u'text']]
    desired_probs = [[0.7], [0.7, 0.6], [1.0]]
    assert desired_preds == decoded_preds
    assert desired_probs == probs_pred

    decoded_preds, probs_pred = decode_multilabel(preds, index2word, min_val=0.7,
                                                  get_probs=True)
    desired_preds = [[u'This'], [u'is'], [u'text']]
    desired_probs = [[0.7], [0.7], [1.0]]
    assert desired_preds == decoded_preds
    assert desired_probs == probs_pred


def test_replace_unknown_words():
    src_word_seq = [[u'Ejemplo', u'de', u'texto', u'en', u'castellano', u'para', u'hacer', u'sustituciones'],
                    [u'Ejemplo', u'de', u'texto', u'en', u'castellano', u'para', u'hacer', u'sustituciones']]

    trg_word_seq = [[u'This', u'is', u'a', u'text', u'file', u'.', u'Containing', u'首', u'先', u'，'],
                    [u'This', u'<unk>', u'a', u'text', u'file', u'.', u'Containing', u'<unk>', u'先', u'，']]
    hard_alignment = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 4, 0, 0]])
    unk_symbol = u'<unk>'
    mapping = {u'castellano': u'Spanish', u'Ejemplo': u'example'}

    desired_trans_words_h0 = [[u'This', u'is', u'a', u'text', u'file', u'.', u'Containing', u'首', u'先', u'，'],
                              [u'This', u'Ejemplo', u'a', u'text', u'file', u'.', u'Containing', u'castellano', u'先', u'，']]
    desired_trans_words_h1 = [[u'This', u'is', u'a', u'text', u'file', u'.', u'Containing', u'首', u'先', u'，'],
                              [u'This', u'example', u'a', u'text', u'file', u'.', u'Containing', u'Spanish', u'先', u'，']]
    desired_trans_words_h2 = [[u'This', u'is', u'a', u'text', u'file', u'.', u'Containing', u'首', u'先', u'，'],
                              [u'This', u'Ejemplo', u'a', u'text', u'file', u'.', u'Containing', u'Spanish', u'先', u'，']]

    for i in range(len(src_word_seq)):
        new_trans_words = replace_unknown_words(src_word_seq[i], trg_word_seq[i], hard_alignment[i], unk_symbol, heuristic=0, mapping=None)
        assert new_trans_words == desired_trans_words_h0[i]

        new_trans_words = replace_unknown_words(src_word_seq[i], trg_word_seq[i], hard_alignment[i], unk_symbol, heuristic=1, mapping=mapping)
        assert new_trans_words == desired_trans_words_h1[i]

        new_trans_words = replace_unknown_words(src_word_seq[i], trg_word_seq[i], hard_alignment[i], unk_symbol, heuristic=2, mapping=mapping)
        assert new_trans_words == desired_trans_words_h2[i]


def test_decode_predictions_beam_search():
    index2word = {
        0: u'<pad>',
        1: u'<unk>',
        2: u'This',
        3: u'is',
        4: u'a',
        5: u'text',
        6: u'file',
        7: u'.',
        8: u'Containing',
        9: u'characters',
        10: u'ẁñ',
        11: u'ü',
        12: u'^',
        13: u'首',
        14: u'先',
        15: u'，'
    }
    preds = [[2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 0, 0, 0, 0, 0, 0],
             [2, 1, 4, 5, 6, 7, 8, 1, 14, 15, 0, 0, 0, 0, 0, 0]]

    # Test regular decoding without padding
    desired_predictions = [
        u'This is a text file . Containing 首 先 ， <pad> <pad> <pad> <pad> <pad>',
        u'This <unk> a text file . Containing <unk> 先 ， <pad> <pad> <pad> <pad> <pad>'
    ]
    decoded_predictions = decode_predictions_beam_search(preds, index2word)
    assert decoded_predictions == desired_predictions

    # Test regular decoding with padding
    desired_predictions = [u'This is a text file . Containing 首 先 ，',
                           u'This <unk> a text file . Containing <unk> 先 ，',
                           ]
    decoded_predictions = decode_predictions_beam_search(preds, index2word,
                                                         pad_sequences=True)
    assert decoded_predictions == desired_predictions

    # Test unk replace - Heuristic 0
    x_text = [u'Ejemplo de texto en castellano para hacer sustituciones',
              u'Ejemplo de texto en castellano para hacer sustituciones']
    alphas = np.array([
        [
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]],
        [
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.7, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            [0.5, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]]])

    decoded_predictions = decode_predictions_beam_search(preds, index2word,
                                                         pad_sequences=True,
                                                         alphas=alphas,
                                                         x_text=x_text, heuristic=0,
                                                         mapping=None)
    # Test regular decoding with padding
    desired_predictions = [u'This is a text file . Containing 首 先 ，',
                           u'This Ejemplo a text file . Containing castellano 先 ，',
                           ]
    assert decoded_predictions == desired_predictions

    # Test unk replace - Heuristic 1
    mapping = {u'castellano': u'Spanish', u'Ejemplo': u'example'}

    decoded_predictions = decode_predictions_beam_search(preds, index2word,
                                                         pad_sequences=True,
                                                         alphas=alphas,
                                                         x_text=x_text,
                                                         heuristic=1,
                                                         mapping=mapping)
    # Test regular decoding with padding
    desired_predictions = [u'This is a text file . Containing 首 先 ，',
                           u'This example a text file . Containing Spanish 先 ，',
                           ]
    assert decoded_predictions == desired_predictions

    # Test unk replace - Heuristic 2: Copy if source starts with capital letter.
    decoded_predictions = decode_predictions_beam_search(preds, index2word,
                                                         pad_sequences=True,
                                                         alphas=alphas,
                                                         x_text=x_text,
                                                         heuristic=2,
                                                         mapping=mapping)
    # Test regular decoding with padding
    desired_predictions = [u'This is a text file . Containing 首 先 ，',
                           u'This Ejemplo a text file . Containing Spanish 先 ，',
                           ]
    assert decoded_predictions == desired_predictions


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
    assert key == 5


if __name__ == '__main__':
    pytest.main([__file__])
