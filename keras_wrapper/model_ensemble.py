# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import math
import sys
import time
import numpy as np

from keras_wrapper.dataset import Data_Batch_Generator
from keras_wrapper.utils import one_hot_2_indices, checkParameters
from keras_wrapper.search import beam_search

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
try:
    import cupy as cp
    cupy = True
except:
    import numpy as cp
    logger.info('<<< Cupy not available. Using numpy. >>>')
    cupy = False


class BeamSearchEnsemble:
    """
    Beam search with one or more autoreggressive models.
    """

    def __init__(self, models, dataset, params_prediction, model_weights=None, n_best=False, verbose=0):
        """
        Initialize the models, dataset and params of the method.
        :param models: Models for provide the probabilities.
        :param dataset: Dataset instance for the model.
        :param params_prediction: Prediction parameters.
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction.get('optimized_search', False)
        self.return_alphas = params_prediction.get('coverage_penalty', False) or params_prediction.get('pos_unk', False)
        self.n_best = n_best
        self.verbose = verbose
        self.model_weights = np.asarray([1. / len(models)] * len(models), dtype='float32') if (model_weights is None) or (model_weights == []) else np.asarray(model_weights, dtype='float32')
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or 'ipykernel' in sys.modules)
        if self.verbose > 0:
            logger.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples
    def predict_cond_optimized(self, X, states_below, params, ii, prev_outs):
        """
        Call the prediction functions of all models, according to their inputs
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """
        probs_list = None
        alphas_list = None
        prev_outs_list = []
        for i, model in list(enumerate(self.models)):
            [model_probs, next_outs] = model.predict_cond_optimized(X,
                                                                    states_below,
                                                                    params,
                                                                    ii,
                                                                    prev_out=prev_outs[i])
            # We introduce an additional dimension to the output of each model for stacking probs
            if probs_list is None:
                probs_list = model_probs[None]
            else:
                probs_list = cp.vstack((probs_list, model_probs[None]))
            if self.return_alphas:
                if alphas_list is None:
                    alphas_list = next_outs[-1][0][None]
                else:
                    alphas_list = np.vstack((alphas_list, next_outs[-1][0][None]))
                next_outs = next_outs[:-1]
            prev_outs_list.append(next_outs)
        probs = cp.sum(cp.asarray(self.model_weights[:, None, None]) * probs_list, axis=0)
        alphas = np.sum(self.model_weights[:, None, None] * alphas_list, axis=0) if self.return_alphas else None
        return probs, prev_outs_list, alphas

    def predict_cond(self, X, states_below, params, ii):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in list(enumerate(self.models)):
            probs_list.append(model.predict_cond(X, states_below, params, ii))

        probs = sum(probs_list[i] * self.model_weights[i] for i in range(len(self.models)))

        if self.return_alphas:
            alphas = np.asarray(sum(alphas_list[i] for i in range(len(self.models))))
        else:
            alphas = None
        return probs

    def predictBeamSearchNet(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * max_batch_size: size of the maximum batch loaded into memory
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'normalization_type': None,
                          'mean_substraction': False,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'pos_unk': False,
                          'state_below_index': -1,
                          'state_below_maxlen': -1,
                          'search_pruning': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2,
                          'attend_on_output': False
                          }
        params = checkParameters(self.params, default_params)
        predictions = dict()
        for s in params['predict_on_sets']:
            logger.info("\n <<< Predicting outputs of " + s + " set >>>")
            if len(params['model_inputs']) == 0:
                raise AssertionError('We need at least one input!')
            if not params['optimized_search']:  # use optimized search model if available
                if params['pos_unk']:
                    raise AssertionError('PosUnk is not supported with non-optimized beam search methods')
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            if params['n_samples'] < 1:
                n_samples = eval("self.dataset.len_" + s)
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                normalization_type=params['normalization_type'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True).generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                normalization_type=params['normalization_type'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=False,
                                                random_samples=n_samples).generator()
            if params['n_samples'] > 0:
                references = []
                sources_sampling = []
            best_samples = []
            if params['pos_unk']:
                best_alphas = []
                sources = []

            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            if self.n_best:
                n_best_list = []
            for _ in range(num_iterations):
                data = next(data_gen)
                X = dict()
                if params['n_samples'] > 0:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[0][input_id]
                        s_dict[input_id] = X[input_id]
                    sources_sampling.append(s_dict)

                    Y = dict()
                    for output_id in params['model_outputs']:
                        Y[output_id] = data[1][output_id]
                else:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[input_id]
                        if params['pos_unk']:
                            s_dict[input_id] = X[input_id]
                    if params['pos_unk']:
                        sources.append(s_dict)

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    if not hasattr(self, '_dynamic_display') or self._dynamic_display:
                        sys.stdout.write('\r')
                    else:
                        sys.stdout.write('\n')
                    sys.stdout.flush()
                    x = dict()
                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    samples, scores, alphas = beam_search(self, x, params,
                                                          null_sym=self.dataset.extra_words['<null>'],
                                                          return_alphas=self.return_alphas,
                                                          model_ensemble=True,
                                                          n_models=len(self.models))

                    if params['length_penalty'] or params['coverage_penalty']:
                        if params['length_penalty']:
                            length_penalties = [((5 + len(sample)) ** params['length_norm_factor'] / (5 + 1) ** params['length_norm_factor'])
                                                # this 5 is a magic number by Google...
                                                for sample in samples]
                        else:
                            length_penalties = [1.0 for _ in samples]

                        if params['coverage_penalty']:
                            coverage_penalties = []
                            for k, sample in list(enumerate(samples)):
                                # We assume that source sentences are at the first position of x
                                x_sentence = x[params['model_inputs'][0]][0]
                                alpha = np.asarray(alphas[k])
                                cp_penalty = 0.0
                                for cp_i in range(len(x_sentence)):
                                    att_weight = 0.0
                                    for cp_j in range(len(sample)):
                                        att_weight += alpha[cp_j, cp_i]
                                    cp_penalty += np.log(min(att_weight, 1.0))
                                coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
                        else:
                            coverage_penalties = [0.0 for _ in samples]
                        scores = [co / lp + cov_p for co, lp, cov_p in zip(scores, length_penalties, coverage_penalties)]

                    elif params['normalize_probs']:
                        counts = [len(sample) ** params['alpha_factor'] for sample in samples]
                        scores = [co / cn for co, cn in zip(scores, counts)]

                    if self.n_best:
                        n_best_indices = np.argsort(scores)
                        n_best_scores = np.asarray(scores)[n_best_indices]
                        n_best_samples = np.asarray(samples)[n_best_indices]
                        if alphas is not None:
                            n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
                        else:
                            n_best_alphas = [None] * len(n_best_indices)
                        n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])
                    best_score = np.argmin(scores)
                    best_sample = samples[best_score]
                    best_samples.append(best_sample)
                    if params['pos_unk']:
                        best_alphas.append(np.asarray(alphas[best_score]))
                    total_cost += scores[best_score]
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled
                    if params['n_samples'] > 0:
                        for output_id in params['model_outputs']:
                            references.append(Y[output_id][i])

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
            sys.stdout.write('The sampling took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()
            if self.n_best:
                if params['pos_unk']:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources), n_best_list
                else:
                    predictions[s] = np.asarray(best_samples), n_best_list
            else:
                if params['pos_unk']:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources)
                else:
                    predictions[s] = np.asarray(best_samples)

        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling

    def sample_beam_search(self, src_sentence):
        """

        :param src_sentence:
        :return:
        """
        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': 1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'state_below_maxlen': -1,
                          'output_text_index': 0,
                          'search_pruning': False,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = checkParameters(self.params, default_params)
        params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
        params['n_samples'] = 1
        if self.n_best:
            n_best_list = []
        X = dict()
        for input_id in params['model_inputs']:
            X[input_id] = src_sentence
        x = dict()
        for input_id in params['model_inputs']:
            x[input_id] = np.asarray([X[input_id]])
        samples, scores, alphas = beam_search(self, x, params,
                                              null_sym=self.dataset.extra_words['<null>'],
                                              return_alphas=self.return_alphas,
                                              model_ensemble=True,
                                              n_models=len(self.models))

        if params['length_penalty'] or params['coverage_penalty']:
            if params['length_penalty']:
                length_penalties = [((5 + len(sample)) ** params['length_norm_factor'] / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                                    for sample in samples]
            else:
                length_penalties = [1.0 for _ in samples]

            if params['coverage_penalty']:
                coverage_penalties = []
                for k, sample in list(enumerate(samples)):
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas[k])
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
            else:
                coverage_penalties = [0.0 for _ in samples]
            scores = [co / lp + cov_p for co, lp, cov_p in zip(scores, length_penalties, coverage_penalties)]

        elif params['normalize_probs']:
            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
            scores = [co / cn for co, cn in zip(scores, counts)]

        if self.n_best:
            n_best_indices = np.argsort(scores)
            n_best_scores = np.asarray(scores)[n_best_indices]
            n_best_samples = np.asarray(samples)[n_best_indices]
            if alphas is not None:
                n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
            else:
                n_best_alphas = [None] * len(n_best_indices)
            n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])

        best_score_idx = np.argmin(scores)
        best_sample = samples[best_score_idx]
        if params['pos_unk']:
            best_alphas = np.asarray(alphas[best_score_idx])
        else:
            best_alphas = None
        if self.n_best:
            return (np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)), n_best_list
        else:
            return np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)

    def score_cond_model(self, X, Y, params, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs.
        :param Y: Outputs to score.
        :param params: Search parameters
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        pad_on_batch = params['pad_on_batch']
        score = 0.0
        if self.return_alphas:
            all_alphas = []
        else:
            all_alphas = None
        if params['words_so_far']:
            state_below = np.asarray([[null_sym]]) \
                if pad_on_batch else np.asarray([np.zeros((params['maxlen'], params['maxlen']))])
        else:
            state_below = np.asarray([null_sym]) \
                if pad_on_batch else np.asarray([np.zeros(params['state_below_maxlen']) + null_sym])

        prev_outs = [None] * len(self.models)
        for ii in range(len(Y)):
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond_optimized(X, state_below, params, ii, prev_outs)
            else:
                probs = self.predict_cond(X, state_below, params, ii)
            # total score for every sample is sum of -log of word prb
            score -= cp.log(probs[0, int(Y[ii])])
            state_below = np.asarray([Y[:ii + 1]], dtype='int64')
            if self.return_alphas:
                all_alphas.append(alphas[0])
            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if pad_on_batch:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
            else:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                         np.zeros((state_below.shape[0],
                                                   max(params['maxlen'] - state_below.shape[1] - 1, 0)),
                                                  dtype='int64')))

                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
                    state_below = np.hstack((state_below,
                                             np.zeros((state_below.shape[0], params['maxlen'] - state_below.shape[1],
                                                       state_below.shape[2]))))

            if self.optimized_search and ii > 0:
                for n_model in range(len(self.models)):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_outs[n_model])):
                        prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars]
        if cupy:
            score = cp.asnumpy(score)

        return score, all_alphas

    def scoreNet(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * max_batch_size: size of the maximum batch loaded into memory
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'normalization_type': None,
                          'mean_substraction': False,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'state_below_maxlen': -1,
                          'output_text_index': 0,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = checkParameters(self.params, default_params)

        scores_dict = dict()

        for s in params['predict_on_sets']:
            logger.info("<<< Scoring outputs of " + s + " set >>>")
            if len(params['model_inputs']) == 0:
                raise AssertionError('We need at least one input!')
            if not params['optimized_search']:  # use optimized search model if available
                if params['pos_unk']:
                    raise AssertionError('PosUnk is not supported with non-optimized beam search methods')
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            n_samples = eval("self.dataset.len_" + s)
            num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

            # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
            # TODO: We prepare data as model 0... Different data preparators for each model?
            data_gen = Data_Batch_Generator(s,
                                            self.models[0],
                                            self.dataset,
                                            num_iterations,
                                            shuffle=False,
                                            batch_size=1,
                                            normalization=params['normalize'],
                                            normalization_type=params['normalization_type'],
                                            data_augmentation=False,
                                            mean_substraction=params['mean_substraction'],
                                            predict=False).generator()
            sources_sampling = []
            scores = []
            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            for _ in range(num_iterations):
                data = next(data_gen)
                X = dict()
                s_dict = {}
                for input_id in params['model_inputs']:
                    X[input_id] = data[0][input_id]
                    s_dict[input_id] = X[input_id]
                sources_sampling.append(s_dict)

                Y = dict()
                for output_id in params['model_outputs']:
                    Y[output_id] = data[1][output_id]

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1

                    if not hasattr(self, '_dynamic_display') or self._dynamic_display:
                        sys.stdout.write('\r')
                    else:
                        sys.stdout.write('\n')
                    sys.stdout.write("Scored %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()

                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    sample = one_hot_2_indices([Y[params['dataset_outputs'][params['output_text_index']]][i]],
                                               pad_sequences=True, verbose=0)[0]
                    score, alphas = self.score_cond_model(x, sample, params,
                                                          null_sym=self.dataset.extra_words['<null>'])

                    if params['length_penalty'] or params['coverage_penalty']:
                        if params['length_penalty']:
                            length_penalty = ((5 + len(sample)) ** params['length_norm_factor'] / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                        else:
                            length_penalty = 1.0

                        if params['coverage_penalty']:
                            # We assume that source sentences are at the first position of x
                            x_sentence = x[params['model_inputs'][0]][0]
                            alpha = np.asarray(alphas)
                            cp_penalty = 0.0
                            for cp_i in range(len(x_sentence)):
                                att_weight = 0.0
                                for cp_j in range(len(sample)):
                                    att_weight += alpha[cp_j, cp_i]
                                cp_penalty += np.log(min(att_weight, 1.0))
                            coverage_penalty = params['coverage_norm_factor'] * cp_penalty
                        else:
                            coverage_penalty = 0.0
                        score = score / length_penalty + coverage_penalty

                    elif params['normalize_probs']:
                        counts = float(len(sample) ** params['alpha_factor'])
                        score /= counts

                    scores.append(score)
                    total_cost += score
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
            sys.stdout.write('The scoring took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()
            scores_dict[s] = scores
        return scores_dict

    def scoreSample(self, data):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * batch_size: size of the batch
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'pad_on_batch': True,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'pos_unk': False,
                          'mapping': None,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = checkParameters(self.params, default_params)

        scores = []
        total_cost = 0
        sampled = 0
        X = dict()
        for i, input_id in list(enumerate(params['model_inputs'])):
            X[input_id] = data[0][i]
        Y = dict()
        for i, output_id in list(enumerate(params['model_outputs'])):
            Y[output_id] = data[1][i]

        for i in range(len(X[params['model_inputs'][0]])):
            sampled += 1
            x = dict()

            for input_id in params['model_inputs']:
                x[input_id] = np.asarray([X[input_id][i]])
            sample = one_hot_2_indices([Y[params['dataset_outputs'][params['output_text_index']]][i]],
                                       pad_sequences=params['pad_on_batch'], verbose=0)[0]
            score, alphas = self.score_cond_model(x, sample,
                                                  params,
                                                  null_sym=self.dataset.extra_words['<null>'])

            if params['length_penalty'] or params['coverage_penalty']:
                if params['length_penalty']:
                    length_penalty = ((5 + len(sample)) ** params['length_norm_factor'] / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                else:
                    length_penalty = 1.0

                if params['coverage_penalty']:
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas)
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalty = params['coverage_norm_factor'] * cp_penalty
                else:
                    coverage_penalty = 0.0
                score = score / length_penalty + coverage_penalty
            elif params['normalize_probs']:
                counts = float(len(sample) ** params['alpha_factor'])
                score /= counts

            scores.append(score)
            total_cost += score
        return scores

    def BeamSearchNet(self):
        """
        DEPRECATED, use predictBeamSearchNet() instead.
        """
        logger.warning("Deprecated function, use predictBeamSearchNet() instead.")
        return self.predictBeamSearchNet()


class PredictEnsemble:
    def __init__(self, models, dataset, params_prediction, postprocess_fun=None, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.postprocess_fun = postprocess_fun
        self.dataset = dataset
        self.params = params_prediction
        self.verbose = verbose

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
    # PREDICTION FUNCTIONS: Functions for making prediction on input samples

    @staticmethod
    def predict_generator(models, data_gen, val_samples=1, max_q_size=1):
        """
        Call the prediction functions of all models, according to their inputs
        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A Numpy array of predictions.
        """

        outs_list = []
        for m in list(models):
            outs_list.append(m.model.predict_on_batch(data_gen, val_samples, max_q_size))
        outs = sum(outs_list[i] for i in range(len(models))) / float(len(models))
        return outs

    @staticmethod
    def predict_on_batch(models, X, in_name=None, out_name=None, expand=False):
        """
        Applies a forward pass and returns the predicted values of all models.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A Numpy array of predictions.
        """

        outs_list = []
        for _, m in list(enumerate(models)):
            outs_list.append(m.model.predict_on_batch(X))
        outs = sum(outs_list[i] for i in range(len(models))) / float(len(models))
        return outs

    def predictNet(self):
        """
            Returns the predictions of the net on the dataset splits chosen. The input 'parameters' is a dict()
            which may contain the following parameters:

            :param batch_size: size of the batch
            :param n_parallel_loaders: number of parallel data batch loaders
            :param normalize: apply data normalization on images/features or not
                              (only if using images/features as input)
            :param mean_substraction: apply mean data normalization on images or not (only if using images as input)
            :param predict_on_sets: list of set splits for which we want to extract
                                    the predictions ['train', 'val', 'test']

            Additional parameters:

            :param postprocess_fun : post-processing function applied to all predictions before returning the result.
                                     The output of the function must be a list of results, one per sample.
                                     If postprocess_fun is a list, the second element will be used as an extra
                                     input to the function.

            :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50,
                          'n_parallel_loaders': 8,
                          'normalize': False,
                          'normalization_type': None,
                          'mean_substraction': False,
                          'model_inputs': ['input1'],
                          'model_outputs': ['output1'],
                          'dataset_inputs': ['input1'],
                          'dataset_outputs': ['output1'],
                          'n_samples': None,
                          'init_sample': -1,
                          'final_sample': -1,
                          'verbose': 1,
                          'predict_on_sets': ['val'],
                          'max_eval_samples': None
                          }

        params = checkParameters(self.params, default_params)
        predictions = dict()

        for s in params['predict_on_sets']:
            logger.info("\n <<< Predicting outputs of " + s + " set >>>")
            if len(params['model_inputs']) == 0:
                raise AssertionError('We need at least one input!')
            # Calculate how many interations are we going to perform
            if params['n_samples'] is None:
                if params['init_sample'] > -1 and params['final_sample'] > -1:
                    n_samples = params['final_sample'] - params['init_sample']
                else:
                    n_samples = eval("self.dataset.len_" + s)
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))
                n_samples = min(eval("self.dataset.len_" + s), num_iterations * params['batch_size'])

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                normalization_type=params['normalization_type'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                init_sample=params['init_sample'],
                                                final_sample=params['final_sample'],
                                                predict=True).generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                normalization_type=params['normalization_type'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True,
                                                random_samples=n_samples).generator()
            # Predict on model
            # if self.postprocess_fun is None:
            #
            #     out = self.predict_generator(self.models,
            #                                  data_gen,
            #                                  val_samples=n_samples,
            #                                  max_q_size=params['n_parallel_loaders'])
            #     predictions[s] = out
            processed_samples = 0
            start_time = time.time()
            while processed_samples < n_samples:
                out = self.predict_on_batch(self.models, next(data_gen))
                # Apply post-processing function
                if self.postprocess_fun is not None:
                    if isinstance(self.postprocess_fun, list):
                        last_processed = min(processed_samples + params['batch_size'], n_samples)
                        out = self.postprocess_fun[0](out, self.postprocess_fun[1][processed_samples:last_processed])
                    else:
                        out = self.postprocess_fun(out)

                if predictions.get(s) is None:
                    predictions[s] = [out]
                else:
                    predictions[s].append(out)
                # Show progress
                processed_samples += params['batch_size']
                if processed_samples > n_samples:
                    processed_samples = n_samples
                eta = (n_samples - processed_samples) * (time.time() - start_time) / processed_samples
                sys.stdout.write("Predicting %d/%d  -  ETA: %ds " % (processed_samples, n_samples, int(eta)))
                if not hasattr(self, '_dynamic_display') or self._dynamic_display:
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')
                sys.stdout.flush()
            predictions[s] = np.concatenate([pred for pred in predictions[s]])

        return predictions
