# -*- coding: utf-8 -*-
import copy
import itertools
import sys
import time
from six import iteritems
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
if sys.version_info.major == 2:
    from itertools import imap as map


def checkParameters(input_params,
                    default_params,
                    hard_check=False):
    """Validates a set of input parameters and uses the default ones if not specified.

    :param input_params: Input parameters.
    :param default_params: Default parameters
    :param hard_check: If True, raise exception if a parameter is not valid.
    :return:
    """
    valid_params = [key for key in default_params]
    params = dict()

    # Check input parameters' validity
    for key, val in iteritems(input_params):
        if key in valid_params:
            params[key] = val
        elif hard_check:
            raise ValueError("Parameter '" + key + "' is not a valid parameter.")

    # Use default parameters if not provided
    for key, default_val in iteritems(default_params):
        if key not in params:
            params[key] = default_val

    return params


class MultiprocessQueue():
    """
        Wrapper class for encapsulating the behaviour of some multiprocessing
        communication structures.

        See how Queues and Pipes work in the following link
        https://docs.python.org/2/library/multiprocessing.html#multiprocessing-examples
    """

    def __init__(self,
                 manager,
                 multiprocess_type='Queue'):
        if multiprocess_type != 'Queue' and multiprocess_type != 'Pipe':
            raise NotImplementedError(
                'Not valid multiprocessing queue of type ' + multiprocess_type)

        self.type = multiprocess_type
        if multiprocess_type == 'Queue':
            self.queue = eval('manager.' + multiprocess_type + '()')
        else:
            self.queue = eval(multiprocess_type + '()')

    def put(self,
            elem):
        if self.type == 'Queue':
            self.queue.put(elem)
        elif self.type == 'Pipe':
            self.queue[1].send(elem)

    def get(self):
        if self.type == 'Queue':
            return self.queue.get()
        elif self.type == 'Pipe':
            return self.queue[0].recv()

    def qsize(self):
        if self.type == 'Queue':
            return self.queue.qsize()
        elif self.type == 'Pipe':
            return -1

    def empty(self):
        if self.type == 'Queue':
            return self.queue.empty()
        elif self.type == 'Pipe':
            return not self.queue[0].poll()


def bbox(img,
         mode='max'):
    """
    Returns a bounding box covering all the non-zero area in the image.

    :param img: Image on which print the bounding box
    :param mode:  "width_height" returns width in [2] and height in [3], "max" returns xmax in [2] and ymax in [3]
    :return:
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y, ymax = np.where(rows)[0][[0, -1]]
    x, xmax = np.where(cols)[0][[0, -1]]

    if mode == 'width_height':
        return x, y, xmax - x, ymax - y
    elif mode == 'max':
        return x, y, xmax, ymax


def simplifyDataset(ds,
                    id_classes,
                    n_classes=50):
    """

    :param ds:
    :param id_classes:
    :param n_classes:
    :return:
    """
    logger.info("Simplifying %s from %d to %d classes." % (str(ds.name), len(ds.classes), n_classes))
    ds.classes[id_classes] = ds.classes[id_classes][:n_classes]

    id_labels = ds.ids_outputs[ds.types_outputs.index('categorical')]

    # reduce each data split
    for s in ['train', 'val', 'test']:
        kept_Y = dict()
        kept_X = dict()
        labels_set = getattr(ds, 'Y_' + s)[id_labels]
        for i, y in list(enumerate(labels_set)):
            if y < n_classes:
                for id_out in ds.ids_outputs:
                    y_split = getattr(ds, 'Y_' + s)
                    sample = y_split[id_out][i]
                    try:
                        kept_Y[id_out].append(sample)
                    except Exception:
                        kept_Y[id_out] = []
                        kept_Y[id_out].append(sample)
                for id_in in ds.ids_inputs:
                    # exec ('sample = ds.X_' + s + '[id_in][i]')
                    x_split = getattr(ds, 'X_' + s)
                    sample = x_split[id_in][i]
                    try:
                        kept_X[id_in].append(sample)
                    except Exception:
                        kept_X[id_in] = []
                        kept_X[id_in].append(sample)
        setattr(ds, 'X_' + s, copy.copy(kept_X))
        setattr(ds, 'Y_' + s, copy.copy(kept_Y))
        setattr(ds, 'len_' + s, len(kept_Y[id_labels]))


def average_models(models,
                   output_model,
                   weights=None,
                   custom_objects=None):
    from keras_wrapper.saving import loadModel, saveModel
    if not isinstance(models, list):
        raise AssertionError('You must give a list of models to average.')
    if len(models) == 0:
        raise AssertionError('You provided an empty list of models to average!')

    model_weights = np.asarray([1. / len(models)] * len(models),
                               dtype=np.float32) if (weights is None) or (weights == []) else np.asarray(weights,
                                                                                                         dtype=np.float32)
    if len(model_weights) != len(models):
        raise AssertionError(
            'You must give a list of weights of the same size than the list of models.')
    loaded_models = [loadModel(m,
                               -1,
                               full_path=True,
                               custom_objects=custom_objects) for m in models]

    # Check that all models are compatible
    if not all([hasattr(loaded_model, 'model') for loaded_model in loaded_models]):
        raise AssertionError('Not all models have the attribute "model".')
    if not (all([hasattr(loaded_model, 'model_init') for loaded_model in
                 loaded_models]) or all(
        [not hasattr(loaded_model, 'model_init') for loaded_model in
         loaded_models])):
        raise AssertionError('Not all models have the attribute "model_init".')

    if not (all([hasattr(loaded_model, 'model_next') for loaded_model in
                 loaded_models]) or all(
        [not hasattr(loaded_model, 'model_next') for loaded_model in
         loaded_models])):
        raise AssertionError('Not all models have the attribute "model_next".')

    # Check all layers are the same

    if not (all([[str(loaded_models[0].model.weights[i]) == str(loaded_model.model.weights[i]) for i in
                  range(len(loaded_models[0].model.weights))] for loaded_model in loaded_models])):
        raise AssertionError('Not all models have the same weights!')

    if hasattr(loaded_models[0], 'model_init') and getattr(loaded_models[0], 'model_init') is not None:
        if not all([[str(loaded_models[0].model.weights[i]) == str(loaded_model.model.weights[i]) for i in
                     range(len(loaded_models[0].model_init.weights))] for loaded_model in loaded_models]):
            raise AssertionError('Not all model_inits have the same weights!')

    if hasattr(loaded_models[0], 'model_next') and getattr(loaded_models[0], 'model_next') is not None:
        if not all([[str(loaded_models[0].model_next.weights[i]) == str(loaded_model.model_next.weights[i]) for i in
                     range(len(loaded_models[0].model_next.weights))] for loaded_model in loaded_models]):
            raise AssertionError('Not all model_nexts have the same weights!')

    # Retrieve weights, weigh them and overwrite in model[0].
    current_weights = loaded_models[0].model.get_weights()
    loaded_models[0].model.set_weights(
        [current_weights[matrix_index] * model_weights[0] for matrix_index in range(len(current_weights))])
    # We have model_init
    if hasattr(loaded_models[0], 'model_init') and getattr(loaded_models[0], 'model_init') is not None:
        current_weights = loaded_models[0].model_init.get_weights()
        loaded_models[0].model_init.set_weights(
            [current_weights[matrix_index] * model_weights[0] for matrix_index in range(len(current_weights))])

    # We have model_next
    if hasattr(loaded_models[0], 'model_next') and getattr(loaded_models[0], 'model_next') is not None:
        current_weights = loaded_models[0].model_next.get_weights()
        loaded_models[0].model_next.set_weights(
            [current_weights[matrix_index] * model_weights[0] for matrix_index in range(len(current_weights))])

    # Weighted sum of all models
    for m in range(1, len(models)):
        current_weights = loaded_models[m].model.get_weights()
        prev_weights = loaded_models[0].model.get_weights()
        loaded_models[0].model.set_weights(
            [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in
             range(len(current_weights))])

        # We have model_init
        if hasattr(loaded_models[0], 'model_init') and getattr(loaded_models[0], 'model_init') is not None:
            current_weights = loaded_models[m].model_init.get_weights()
            prev_weights = loaded_models[0].model_init.get_weights()
            loaded_models[0].model_init.set_weights(
                [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in
                 range(len(current_weights))])

        # We have model_next
        if hasattr(loaded_models[0], 'model_next') and getattr(loaded_models[0], 'model_next') is not None:
            current_weights = loaded_models[m].model_next.get_weights()
            prev_weights = loaded_models[0].model_next.get_weights()
            loaded_models[0].model_next.set_weights(
                [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in
                 range(len(current_weights))])

    # Save averaged model
    saveModel(loaded_models[0], -1, path=output_model, full_path=True,
              store_iter=False)


# Text-related utils
def one_hot_2_indices(preds,
                      pad_sequences=True,
                      verbose=0):
    """
    Converts a one-hot codification into a index-based one
    :param preds: Predictions codified as one-hot vectors.
    :param pad_sequences: Whether we should pad sequence or not
    :param verbose: Verbosity level, by default 0.
    :return: List of convertedpredictions
    """
    if verbose > 0:
        logger.info('Converting one hot prediction into indices...')
    preds = list(map(lambda x: np.argmax(x, axis=1), preds))
    if pad_sequences:
        preds = [pred[:sum([int(elem > 0) for elem in pred]) + 1] for pred in preds]
    return preds


def indices_2_one_hot(indices,
                      n):
    """
    Converts a list of indices into one hot codification

    :param indices: list of indices
    :param n: integer. Size of the vocabulary
    :return: numpy array with shape (len(indices),
n)
    """
    one_hot = np.zeros((len(indices), n), dtype=np.int8)
    for i in range(len(indices)):
        if indices[i] >= n:
            raise ValueError("Index out of bounds when converting to one hot")
        one_hot[i, indices[i]] = 1

    return one_hot


# From keras.utils.np_utils
def to_categorical(y,
                   num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y,
                 dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n,
                            num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def categorical_probas_to_classes(p):
    return np.argmax(p,
                     axis=1)


# ------------------------------------------------------- #
#       DECODING FUNCTIONS
#           Functions for decoding predictions
# ------------------------------------------------------- #

def decode_predictions_one_hot(preds,
                               index2word,
                               pad_sequences=True,
                               verbose=0):
    """
    Decodes predictions following a one-hot codification.
    :param preds: Predictions codified as one-hot vectors.
    :param index2word: Mapping from word indices into word characters.
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions
    """
    if verbose > 0:
        logger.info('Decoding one hot prediction ...')
    preds = list(map(lambda prediction: np.argmax(prediction, axis=1), preds))
    PAD = '<pad>'
    flattened_answer_pred = [list(map(lambda index: index2word[index], pred)) for
                             pred in preds]
    answer_pred_matrix = np.asarray(flattened_answer_pred)
    answer_pred = []

    for a_no in answer_pred_matrix:
        end_token_pos = [j for j, x in list(enumerate(a_no)) if x == PAD]
        end_token_pos = None if len(end_token_pos) == 0 or not pad_sequences else end_token_pos[0]
        a_no = [a.decode('utf-8') if isinstance(a,
                                                str) and sys.version_info.major == 2 else a
                for a in a_no]
        tmp = u' '.join(a_no[:end_token_pos])
        answer_pred.append(tmp)
    return answer_pred


def decode_predictions(preds,
                       temperature,
                       index2word,
                       sampling_type,
                       verbose=0):
    """
    Decodes predictions
    :param preds: Predictions codified as the output of a softmax activation function.
    :param temperature: Temperature for sampling.
    :param index2word: Mapping from word indices into word characters.
    :param sampling_type: 'max_likelihood' or 'multinomial'.
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions.
    """

    if verbose > 0:
        logger.info('Decoding prediction ...')
    flattened_preds = preds.reshape(-1, preds.shape[-1])
    flattened_answer_pred = list(map(lambda index: index2word[index], sampling(scores=flattened_preds,
                                                                               sampling_type=sampling_type,
                                                                               temperature=temperature)))
    answer_pred_matrix = np.asarray(flattened_answer_pred).reshape(preds.shape[:-1])

    answer_pred = []
    EOS = '<eos>'
    PAD = '<pad>'

    for a_no in answer_pred_matrix:
        if len(a_no.shape) > 1:  # only process word by word if our prediction has more than one output
            init_token_pos = 0
            end_token_pos = [j for j, x in list(enumerate(a_no)) if x == EOS or x == PAD]
            end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
            a_no = [a.decode('utf-8') if isinstance(a, str) and sys.version_info.major == 2 else a
                    for a in a_no]
            tmp = u' '.join(a_no[init_token_pos:end_token_pos])
        else:
            tmp = a_no[:-1]
        answer_pred.append(tmp)
    return answer_pred


def decode_categorical(preds,
                       index2word,
                       verbose=0):
    """
    Decodes predictions
    :param preds: Predictions codified as the output of a softmax activation function.
    :param index2word: Mapping from word indices into word characters.
    :return: List of decoded predictions.
    """

    if verbose > 0:
        logger.info('Decoding prediction ...')

    word_indices = categorical_probas_to_classes(preds)
    return [index2word.get(word) for word in word_indices]


def decode_multilabel(preds,
                      index2word,
                      min_val=0.5,
                      get_probs=False,
                      verbose=0):
    """
    Decodes predictions
    :param preds: Predictions codified as the output of a softmax activation function.
    :param index2word: Mapping from word indices into word characters.
    :param min_val: Minimum value needed for considering a positive prediction.
    :param get_probs: additionally return probability for each predicted label
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions.
    """

    if verbose > 0:
        logger.info('Decoding prediction ...')

    answer_pred = []
    probs_pred = []
    for pred in preds:
        current_pred = []
        current_probs = []
        for ind, word in list(enumerate(pred)):
            if word >= min_val:
                current_pred.append(index2word[ind])
                current_probs.append(word)
        answer_pred.append(current_pred)
        probs_pred.append(current_probs)

    if get_probs:
        return answer_pred, probs_pred
    else:
        return answer_pred


def replace_unknown_words(src_word_seq,
                          trg_word_seq,
                          hard_alignment,
                          unk_symbol,
                          glossary=None,
                          heuristic=0,
                          mapping=None,
                          verbose=0):
    """
    Replaces unknown words from the target sentence according to some heuristic.
    Borrowed from: https://github.com/sebastien-j/LV_groundhog/blob/master/experiments/nmt/replace_UNK.py
    :param src_word_seq: Source sentence words
    :param trg_word_seq: Hypothesis words
    :param hard_alignment: Target-Source alignments
    :param glossary: Hard-coded substitutions.
    :param unk_symbol: Symbol in trg_word_seq to replace
    :param heuristic: Heuristic (0, 1, 2)
    :param mapping: External alignment dictionary
    :param verbose: Verbosity level
    :return: trg_word_seq with replaced unknown words
    """
    trans_words = trg_word_seq
    new_trans_words = []
    mapping = mapping or {}
    for j in range(len(trans_words)):
        current_word = trans_words[j]
        if glossary is not None and glossary.get(
                src_word_seq[hard_alignment[j]]) is not None:
            current_word = glossary.get(src_word_seq[hard_alignment[j]])
            new_trans_words.append(current_word)
        elif current_word == unk_symbol:
            current_src = src_word_seq[hard_alignment[j]]
            if isinstance(current_src, str) and sys.version_info.major == 2:
                current_src = current_src.decode('utf-8')
            if heuristic == 0:  # Copy (ok when training with large vocabularies on en->fr, en->de)
                new_trans_words.append(current_src)
            elif heuristic == 1:
                # Use the most likely translation (with t-table). If not found, copy the source word.
                # Ok for small vocabulary (~30k) models
                if mapping.get(current_src) is not None:
                    new_trans_words.append(mapping[current_src])
                else:
                    new_trans_words.append(current_src)
            elif heuristic == 2:
                # Use t-table if the source word starts with a lowercase letter. Otherwise copy
                # Sometimes works better than other heuristics
                if mapping.get(current_src) is not None and current_src[0].islower():
                    new_trans_words.append(mapping[current_src])
                else:
                    new_trans_words.append(current_src)
        else:
            new_trans_words.append(current_word)
    return new_trans_words


def decode_predictions_beam_search(preds,
                                   index2word,
                                   glossary=None,
                                   alphas=None,
                                   heuristic=0,
                                   x_text=None,
                                   unk_symbol='<unk>',
                                   pad_sequences=False,
                                   mapping=None,
                                   verbose=0):
    """
    Decodes predictions from the BeamSearch method.

    :param preds: Predictions codified as word indices.
    :param index2word: Mapping from word indices into word characters.
    :param alphas: Attention model weights: Float matrix with shape (I, J) (I: number of target items; J: number of source items).
    :param heuristic: Replace unknown words heuristic (0, 1 or 2)
    :param x_text: Source text (for unk replacement)
    :param unk_symbol: Unknown words symbol
    :param pad_sequences: Whether we should make a zero-pad on the input sequence.
    :param mapping: Source-target dictionary (for unk_replace heuristics 1 and 2)
    :param verbose: Verbosity level, by default 0.
    :return: List of decoded predictions
    """
    if verbose > 0:
        logger.info('Decoding beam search prediction ...')

    if alphas is not None:
        if x_text is None:
            raise AssertionError('When using POS_UNK, you must provide the input '
                                 'text to decode_predictions_beam_search!')
        if verbose > 0:
            logger.info('Using heuristic %d' % heuristic)
    if pad_sequences:
        preds = [pred[:sum([int(elem > 0) for elem in pred]) + 1] for pred in preds]
    flattened_predictions = [list(map(lambda x: index2word[x], pred)) for pred in
                             preds]
    final_predictions = []

    if alphas is not None:
        x_text = list(map(lambda x: x.split(), x_text))
        hard_alignments = list(
            map(lambda alignment, x_sentence: np.argmax(
                alignment[:, :max(1, len(x_sentence))], axis=1), alphas,
                x_text))

        for i, a_no in list(enumerate(flattened_predictions)):
            if unk_symbol in a_no or glossary is not None:
                a_no = replace_unknown_words(x_text[i],
                                             a_no,
                                             hard_alignments[i],
                                             unk_symbol,
                                             glossary=glossary,
                                             heuristic=heuristic,
                                             mapping=mapping,
                                             verbose=verbose)
            a_no = [a.decode('utf-8') if isinstance(a,
                                                    str) and sys.version_info.major == 2 else a
                    for a in a_no]
            tmp = u' '.join(a_no[:-1])
            final_predictions.append(tmp)
    else:
        for a_no in flattened_predictions:
            a_no = [a.decode('utf-8') if isinstance(a,
                                                    str) and sys.version_info.major == 2 else a
                    for a in a_no]
            tmp = u' '.join(a_no[:-1])
            final_predictions.append(tmp)
    return final_predictions


def sampling(scores,
             sampling_type='max_likelihood',
             temperature=1.):
    """
    Sampling words (each sample is drawn from a categorical distribution).
    Or picks up words that maximize the likelihood.
    :param scores: array of size #samples x #classes;
    every entry determines a score for sample i having class j
    :param sampling_type:
    :param temperature: Predictions temperature. The higher, the flatter probabilities. Hence more random outputs.
    :return: set of indices chosen as output, a vector of size #samples
    """
    if isinstance(scores, dict):
        scores = scores['output']

    if sampling_type == 'multinomial':
        preds = np.asarray(scores).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    elif sampling_type == 'max_likelihood':
        return np.argmax(scores, axis=-1)
    else:
        raise NotImplementedError()


# Data structures-related utils
def flatten_list_of_lists(list_of_lists):
    """
    Flattens a list of lists
    :param list_of_lists: List of lists
    :return: Flatten list of lists
    """
    return [item for sublist in list_of_lists for item in sublist]


def flatten(l):
    """
    Flatten a list (more general than flatten_list_of_lists, but also more inefficient
    :param l:
    :return:
    """
    if not l:
        return l
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if isinstance(l,
                                                                                list) else [
        l]


def key_with_max_val(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""

    d = dict((k, v) for k, v in iteritems(d) if isinstance(v, (int, float, complex)))
    v = list(d.values())
    k = list(d.keys())
    if d == {}:
        return -1
    else:
        return k[v.index(max(v))]


def print_dict(d, header=''):
    """
    Formats a dictionary for printing.
    :param d: Dictionary to print.
    :return: String containing the formatted dictionary.
    """
    obj_str = str(header) + '{ \n\t'
    obj_str += "\n\t".join([str(key) + ": " + str(d[key]) for key in sorted(d.keys())])
    obj_str += '\n'
    obj_str += '}'
    return obj_str
