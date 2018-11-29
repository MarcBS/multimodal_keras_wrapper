# -*- coding: utf-8 -*-
import copy
import numpy as np
import logging
try:
    import cupy as cp
    cupy = True
except:
    import numpy as cp
    cupy = False


def beam_search(model, X, params, return_alphas=False, eos_sym=0, null_sym=2, model_ensemble=False, n_models=0):
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
    :param model: Model to use
    :param X: Model inputs
    :param params: Search parameters
    :param return_alphas: Whether we should return attention weights or not.
    :param eos_sym: <eos> symbol
    :param null_sym: <null> symbol
    :param model_ensemble: Whether we are using several models in an ensemble
    :param n_models; Number of models in the ensemble.
    :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
    """
    k = params['beam_size']
    samples = []
    sample_scores = []
    pad_on_batch = params['pad_on_batch']
    dead_k = 0  # samples that reached eos
    live_k = 1  # samples that did not yet reach eos
    hyp_samples = [[]] * live_k
    hyp_scores = cp.zeros(live_k, dtype='float32')
    ret_alphas = return_alphas or params['pos_unk']
    if ret_alphas:
        sample_alphas = []
        hyp_alphas = [[]] * live_k
    if pad_on_batch:
        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        minlen = int(
            len(X[params['dataset_inputs'][0]][0]) / params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0
    else:
        minlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) /
                     params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0

        maxlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) * params[
            'output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        maxlen = min(params['state_below_maxlen'] - 1, maxlen)

    # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
    if params['words_so_far']:
        if k > maxlen:
            raise NotImplementedError("BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN on the current implementation.")
        state_below = np.asarray([[null_sym]] * live_k) if pad_on_batch else np.asarray([np.zeros((maxlen, maxlen))] * live_k)
    else:
        state_below = np.asarray([null_sym] * live_k) if pad_on_batch else np.asarray([np.zeros(params['state_below_maxlen']) + null_sym] * live_k)
    prev_out = [None] * n_models if model_ensemble else None

    for ii in range(maxlen):
        # for every possible live sample calc prob for every possible label
        if params['optimized_search']:  # use optimized search model if available
            if model_ensemble:
                [probs, prev_out, alphas] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
            else:
                [probs, prev_out] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
                if ret_alphas:
                    alphas = prev_out[-1][0]  # Shape: (k, n_steps)
                    prev_out = prev_out[:-1]
        else:
            probs = model.predict_cond(X, state_below, params, ii)
        log_probs = cp.log(probs)
        if minlen > 0 and ii < minlen:
            log_probs[:, eos_sym] = -cp.inf
        # total score for every sample is sum of -log of word prb
        cand_scores = hyp_scores[:, None] - log_probs
        cand_flat = cand_scores.flatten()
        # Find the best options by calling argsort of flatten array
        ranks_flat = cp.argsort(cand_flat)[:(k - dead_k)]
        # Decypher flatten indices
        voc_size = log_probs.shape[1]
        trans_indices = ranks_flat // voc_size  # index of row
        word_indices = ranks_flat % voc_size  # index of col
        costs = cand_flat[ranks_flat]
        best_cost = costs[0]
        if cupy:
            trans_indices = cp.asnumpy(trans_indices)
            word_indices = cp.asnumpy(word_indices)
            if ret_alphas:
                alphas = cp.asnumpy(alphas)

        # Form a beam for the next iteration
        new_hyp_samples = []
        new_trans_indices = []
        new_hyp_scores = cp.zeros(k - dead_k, dtype='float32')
        if ret_alphas:
            new_hyp_alphas = []
        for idx, [ti, wi] in list(enumerate(zip(trans_indices, word_indices))):
            if params['search_pruning']:
                if costs[idx] < k * best_cost:
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if ret_alphas:
                        new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                else:
                    dead_k += 1
            else:
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_trans_indices.append(ti)
                new_hyp_scores[idx] = copy.copy(costs[idx])
                if ret_alphas:
                    new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_alphas = []
        indices_alive = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                samples.append(new_hyp_samples[idx])
                sample_scores.append(new_hyp_scores[idx])
                if ret_alphas:
                    sample_alphas.append(new_hyp_alphas[idx])
                dead_k += 1
            else:
                indices_alive.append(new_trans_indices[idx])
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                if ret_alphas:
                    hyp_alphas.append(new_hyp_alphas[idx])
        hyp_scores = cp.array(np.asarray(hyp_scores, dtype='float32'), dtype='float32')
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break
        state_below = np.asarray(hyp_samples, dtype='int64')

        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below)) \
            if pad_on_batch else \
            np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                       state_below,
                       np.zeros((state_below.shape[0],
                                 max(params['state_below_maxlen'] - state_below.shape[1] - 1, 0)), dtype='int64')))

        # we must include an additional dimension if the input for each timestep are all the generated words so far
        if params['words_so_far']:
            state_below = np.expand_dims(state_below, axis=0)

        if params['optimized_search'] and ii > 0:
            # filter next search inputs w.r.t. remaining samples
            if model_ensemble:
                for n_model in range(n_models):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_out[n_model])):
                        prev_out[n_model][idx_vars] = prev_out[n_model][idx_vars][indices_alive]
            else:
                for idx_vars in range(len(prev_out)):
                    prev_out[idx_vars] = prev_out[idx_vars][indices_alive]

    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            samples.append(hyp_samples[idx])
            sample_scores.append(hyp_scores[idx])
            if ret_alphas:
                sample_alphas.append(hyp_alphas[idx])
    if ret_alphas:
        return samples, sample_scores, np.asarray(sample_alphas)
    else:
        return samples, sample_scores, None
