import pytest
from keras_wrapper.extra.evaluation import *


def test_get_coco_score():
    pred_list = ['Prediction 1', 'Prediction 2', 'Prediction 3']
    extra_vars = {'val': {'references': {0: ['Prediction 1'], 1: ['Prediction 2'],
                                         2: ['Prediction 3', 'Prediction 5']}},
                  'test': {'references': {0: ['Prediction 2'], 1: ['Prediction 3'],
                                          2: ['Prediction 1']}}
                  }
    val_scores = get_coco_score(pred_list, 0, extra_vars, 'val')
    assert np.allclose(val_scores['Bleu_1'], 1.0, atol=1e6)
    assert np.allclose(val_scores['Bleu_2'], 1.0, atol=1e6)
    assert np.allclose(val_scores['Bleu_3'], 1.0, atol=1e6)
    assert np.allclose(val_scores['Bleu_4'], 1.0, atol=1e6)
    assert np.allclose(val_scores['ROUGE_L'], 1.0, atol=1e6)
    assert np.allclose(val_scores['CIDEr'], 5.0, atol=1e6)
    assert np.allclose(val_scores['TER'], 0., atol=1e6)
    assert np.allclose(val_scores['METEOR'], 1.0, atol=1e6)
    test_scores = get_coco_score(pred_list, 0, extra_vars, 'test')

    assert np.allclose(test_scores['Bleu_1'], 0.5, atol=1e6)
    assert np.allclose(test_scores['Bleu_2'], 0., atol=1e6)
    assert np.allclose(test_scores['Bleu_3'], 0., atol=1e6)
    assert np.allclose(test_scores['Bleu_4'], 0., atol=1e6)
    assert np.allclose(test_scores['ROUGE_L'], 0.5, atol=1e6)
    assert np.allclose(test_scores['CIDEr'], 0., atol=1e6)
    assert np.allclose(test_scores['TER'], 0.5, atol=1e6)
    assert np.allclose(test_scores['METEOR'], 0.2, atol=1e6)


def test_multilabel_metrics():
    pred_list = [['w1'], ['w2'], ['w3']]
    extra_vars = {
        'val': {'references': [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                'word2idx': {'w1': 0, 'w2': 1, 'w3': 3, 'w4': 3, 'w5': 4}
                },
        'test': {'references': [[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
                 'word2idx': {'w1': 0, 'w2': 1, 'w3': 2, 'w4': 3, 'w5': 4}
                 }
    }
    val_scores = multilabel_metrics(pred_list, 0, extra_vars, 'val')

    assert np.allclose(val_scores['f1'], 0.66, atol=1e6)
    assert np.allclose(val_scores['recall'], 0.66, atol=1e6)
    assert np.allclose(val_scores['precision'], 0.66, atol=1e6)
    assert np.allclose(val_scores['ranking_loss'], 0.33, atol=1e6)
    assert np.allclose(val_scores['coverage_error'], 2.33, atol=1e6)
    assert np.allclose(val_scores['average_precision'], 0.73, atol=1e6)

    test_scores = multilabel_metrics(pred_list, 0, extra_vars, 'test')
    assert np.allclose(test_scores['f1'], 0.33, atol=1e6)
    assert np.allclose(test_scores['recall'], 0.33, atol=1e6)
    assert np.allclose(test_scores['precision'], 0.22, atol=1e6)
    assert np.allclose(test_scores['ranking_loss'], 0.66, atol=1e6)
    assert np.allclose(test_scores['coverage_error'], 3.66, atol=1e6)
    assert np.allclose(test_scores['average_precision'], 0.466, atol=1e6)


def test_multiclass_metrics():
    # TODO
    pass


def test_compute_perplexity():
    y_pred = [[1, 2, 3, 3, 4], [1, 1, 2, 3, 4]]
    y_true = [[1, 2, 3, 3, 4], [1, 1, 2, 3, 4]]
    ppl = compute_perplexity(y_pred, y_true, 0, 'val')
    assert np.allclose(ppl, 0.47, atol=1e6)


def test_semantic_segmentation_accuracy():
    # TODO
    pass


def test_semantic_segmentation_meaniou():
    # TODO
    pass


def test_averagePrecision():
    # TODO
    pass


if __name__ == '__main__':
    pytest.main([__file__])
