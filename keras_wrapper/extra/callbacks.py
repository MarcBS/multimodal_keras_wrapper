from __future__ import print_function

"""
Extra set of callbacks.
"""

import random
import warnings
import numpy as np
import logging

from keras.callbacks import Callback as KerasCallback

import evaluation
from read_write import *


def checkDefaultParamsBeamSearch(params):

    required_params = ['model_inputs', 'model_outputs', 'dataset_inputs', 'dataset_outputs']
    default_params = {'beam_size': 5, 'maxlen': 30, 'normalize': False, 'alpha_factor': 1.0,
                      'words_so_far': False, 'n_parallel_loaders': 5, 'optimized_search': False}

    for k,v in params.iteritems():
        if k in default_params.keys() or k in required_params:
            default_params[k] = v

    for k in required_params:
        if k not in default_params:
            raise Exception('The beam search parameter ' + k + ' must be specified.')

    return default_params

###################################################
# Performance evaluation callbacks
###################################################

class PrintPerformanceMetricOnEpochEnd(KerasCallback):
    def __init__(self, model, dataset, gt_id, metric_name, set_name, batch_size, each_n_epochs=1, extra_vars=dict(),
                 is_text=False, is_3DLabel=False, index2word_y=None, sampling='max_likelihood', beam_search=False,
                 write_samples=False, save_path='logs/performance.', reload_epoch=0,
                 start_eval_on_epoch=0, write_type='list', sampling_type='max_likelihood',
                 out_pred_idx=None, early_stop=False, patience=5, stop_metric='Bleu-4', verbose=1):
        """
            DEPRECATED!

            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param batch_size: batch size used during sampling
            :param each_n_epochs: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
            :param is_3DLabel: defines if the predicted info is of type 3DLabels
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param write_samples: flag for indicating if we want to write the predicted data in a text file
            :param save_path: path to dumb the logs
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
            :param write_type: method used for writing predictions
            :param sampling_type: type of sampling used (multinomial or max_likelihood)
            :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.is_3DLabel = is_3DLabel
        self.sampling = sampling
        self.beam_search = beam_search
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.each_n_epochs = each_n_epochs
        self.extra_vars = extra_vars
        self.save_path = save_path
        # self.reload_epoch = reload_epoch
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.early_stop = early_stop
        self.patience = patience
        self.stop_metric = stop_metric
        self.best_score = -1
        self.wait = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1  # start by index 1
        # epoch += self.reload_epoch
        if epoch < self.start_eval_on_epoch:
            if self.verbose > 0:
                logging.info('Not evaluating until end of epoch ' + str(self.start_eval_on_epoch))
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            if self.verbose > 0:
                logging.info('Evaluating only every ' + str(self.each_n_epochs) + ' epochs')
            return

        # Evaluate on each set separately
        for s in self.set_name:
            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size,
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s]}

            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)[s]
            else:
                # Convert predictions
                postprocess_fun = None
                if (self.is_3DLabel):
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                predictions = \
                self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)[s]

            if (self.is_text):
                if self.out_pred_idx is not None:
                    predictions = predictions[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(predictions,
                                                                                    self.index2word_y,
                                                                                    verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(predictions, 1,  # always set temperature to 1
                                                                        self.index2word_y,
                                                                        self.sampling_type,
                                                                        verbose=self.verbose)

            # Store predictions
            if self.write_samples:
                # Store result
                filepath = self.save_path + '/' + s + '_epoch_' + str(epoch) + '.pred'  # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])
                elif self.write_type == 'listoflists':
                    listoflists2file(filepath, predictions)
                elif self.write_type == 'numpy':
                    numpy2file(filepath, predictions)
                elif self.write_type == '3DLabels':
                    # TODO
                    print("WRITE SAMPLES FUNCTION NOT IMPLEMENTED")
                else:
                    raise NotImplementedError('The store type "' + self.write_type + '" is not implemented.')

            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric ' + metric)
                filepath = self.save_path + '/' + s + '.' + metric  # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                    pred_list=predictions,
                    verbose=self.verbose,
                    extra_vars=self.extra_vars,
                    split=s)

                # Print results to file
                with open(filepath, 'a') as f:
                    header = 'epoch,'
                    line = str(epoch) + ','
                    # Store in model log
                    self.model_to_eval.log(s, 'epoch', epoch)
                    for metric_ in sorted(metrics):
                        value = metrics[metric_]
                        header += metric_ + ','
                        line += str(value) + ','
                        # Store in model log
                        self.model_to_eval.log(s, metric_, value)
                    if (epoch == 1 or epoch == self.start_eval_on_epoch):
                        f.write(header + '\n')
                    f.write(line + '\n')
                if self.verbose > 0:
                    logging.info('Done evaluating on metric ' + metric)

            """
            # Early stop check
            if self.early_stop and s in ['val', 'validation', 'dev', 'development']:
                current_score = metrics[self.stop_metric]
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    self.wait = 0
                    if self.verbose > 0:
                        logging.info('---current best %s: %.4f' % (self.stop_metric, current_score))
                else:
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            logging.info('Epoch %d: early stopping. Best %s value found at epoch %d: %.4f' %
                                         (epoch, self.stop_metric, self.best_epoch, self.best_score))
                            self.model.stop_training = True
                    self.wait += 1.
                    if self.verbose > 0:
                        logging.info('----bad counter: %d/%d' % (self.wait, self.patience))
            """

class PrintPerformanceMetricEachNUpdates(KerasCallback):
    def __init__(self, model, dataset, gt_id, metric_name, set_name, batch_size, extra_vars=dict(),
                 is_text=False, is_3DLabel=False, index2word_y=None, sampling='max_likelihood', beam_search=False,
                 write_samples=False, save_path='logs/performance.', reload_epoch=0,
                 each_n_updates=10000, start_eval_on_epoch=0, write_type='list', sampling_type='max_likelihood',
                 out_pred_idx=None, early_stop=False, patience=5, stop_metric='Bleu-4', verbose=1):
        """
            DEPRECATED!

            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param batch_size: batch size used during sampling
            :param each_n_epochs: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
            :param is_3DLabel: defines if the predicted info is of type 3DLabels
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param write_samples: flag for indicating if we want to write the predicted data in a text file
            :param save_path: path to dumb the logs
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
            :param write_type: method used for writing predictions
            :param sampling_type: type of sampling used (multinomial or max_likelihood)
            :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.is_3DLabel = is_3DLabel
        self.sampling = sampling
        self.beam_search = beam_search
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.each_n_updates = each_n_updates
        self.extra_vars = extra_vars
        self.save_path = save_path
        # self.reload_epoch = reload_epoch
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.early_stop = early_stop
        self.patience = patience
        self.stop_metric = stop_metric
        self.best_score = -1
        self.wait = 0
        self.verbose = verbose
        self.cum_update = 0
        self.epoch = reload_epoch + 1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.cum_update % self.each_n_updates != 0:
            return
        if self.epoch < self.start_eval_on_epoch:
            return
        # Evaluate on each set separately
        for s in self.set_name:
            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size,
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s]}

            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)[s]
            else:
                # Convert predictions
                postprocess_fun = None
                if (self.is_3DLabel):
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                predictions = \
                self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)[s]

            if (self.is_text):
                if self.out_pred_idx is not None:
                    predictions = predictions[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(predictions,
                                                                                    self.index2word_y,
                                                                                    verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(predictions, 1,  # always set temperature to 1
                                                                        self.index2word_y,
                                                                        self.sampling_type,
                                                                        verbose=self.verbose)

            # Store predictions
            if self.write_samples:
                logging.info('Writing samples with write type: ' + self.write_type)
                # Store result
                filepath = self.save_path + '/' + s + '_update_' + str(self.cum_update) + '.pred'  # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])
                elif self.write_type == 'listoflists':
                    listoflists2file(filepath, predictions)
                elif self.write_type == 'numpy':
                    numpy2file(filepath, predictions)
                elif self.write_type == '3DLabels':
                    # TODO:
                    print("WRITE SAMPLES FUNCTION NOT IMPLEMENTED")
                else:
                    raise NotImplementedError('The store type "' + self.write_type + '" is not implemented.')

            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric ' + metric)
                filepath = self.save_path + '/' + s + '.' + metric  # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                    pred_list=predictions,
                    verbose=self.verbose,
                    extra_vars=self.extra_vars,
                    split=s)

                # Print results to file
                with open(filepath, 'a') as f:
                    header = 'Update,'
                    line = str(self.cum_update) + ','
                    # Store in model log
                    self.model_to_eval.log(s, 'iteration', self.cum_update)
                    for metric_ in sorted(metrics):
                        value = metrics[metric_]
                        header += metric_ + ','
                        line += str(value) + ','
                        # Store in model log
                        self.model_to_eval.log(s, metric_, value)
                    if (self.cum_update == 0 or self.cum_update == self.each_n_updates):
                        f.write(header + '\n')
                    f.write(line + '\n')
                if self.verbose > 0:
                    logging.info('Done evaluating on metric ' + metric)

            """
            # Early stop check
            if self.early_stop and s in ['val', 'validation', 'dev', 'development']:
                current_score = metrics[self.stop_metric]
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_update = self.cum_update
                    self.wait = 0
                    if self.verbose > 0:
                        logging.info('---current best %s: %.4f' % (self.stop_metric, current_score))
                else:
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            logging.info('Update %d: early stopping. Best %s value found at update %d: %.4f' %
                                         (self.cum_update, self.stop_metric, self.best_update, self.best_score))
                            self.model.stop_training = True
                    self.wait += 1
                    if self.verbose > 0:
                        logging.info('----bad counter: %d/%d' % (self.wait, self.patience))
            """

class PrintPerformanceMetricOnEpochEndOrEachNUpdates(KerasCallback):

    def __init__(self, model, dataset, gt_id, metric_name, set_name, batch_size, each_n_epochs=1,
                 extra_vars=None,
                 is_text=False, index2word_y=None, input_text_id=None, index2word_x=None,
                 sampling='max_likelihood',
                 beam_search=False, write_samples=False, save_path='logs/performance.',
                 reload_epoch=0,
                 eval_on_epochs=True, start_eval_on_epoch=0, is_3DLabel=False,
                 write_type='list', sampling_type='max_likelihood',
                 out_pred_idx=None, early_stop=False, patience=5, stop_metric='Bleu-4', verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param batch_size: batch size used during sampling
            :param each_n_epochs: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
            :param is_3DLabel: defines if the predicted info is of type 3DLabels
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param write_samples: flag for indicating if we want to write the predicted data in a text file
            :param save_path: path to dumb the logs
            :param eval_on_epochs: Eval each epochs or updates
            :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
            :param write_type: method used for writing predictions
            :param sampling_type: type of sampling used (multinomial or max_likelihood)
            :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.input_text_id = input_text_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.is_3DLabel = is_3DLabel
        self.sampling = sampling
        self.beam_search = beam_search
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.each_n_epochs = each_n_epochs
        self.extra_vars = extra_vars
        self.save_path = save_path
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.early_stop = early_stop
        self.patience = patience
        self.stop_metric = stop_metric
        self.best_score = -1
        self.best_epoch = -1
        self.wait = 0
        self.verbose = verbose
        self.cum_update = 0
        self.epoch = reload_epoch
        super(PrintPerformanceMetricOnEpochEndOrEachNUpdates, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """
        On epoch end, sample and evaluate on the specified datasets.
        :param epoch: Current epoch number
        :param logs:
        :return:
        """
        epoch += 1  # start by index 1
        self.epoch = epoch
        if not self.eval_on_epochs:
            return
        if epoch < self.start_eval_on_epoch:
            if self.verbose > 0:
                logging.info('Not evaluating until end of epoch ' + str(self.start_eval_on_epoch))
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            if self.verbose > 0:
                logging.info('Evaluating only every ' + str(self.each_n_epochs) + ' epochs')
            return
        self.evaluate(epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        if self.epoch < self.start_eval_on_epoch:
            return
        self.evaluate(self.cum_update, counter_name='iteration')

    def evaluate(self, epoch, counter_name='epoch'):
        # Evaluate on each set separately
        for s in self.set_name:
            # Apply model predictions
            if self.beam_search:
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s],
                                     'pos_unk': False, 'heuristic': 0, 'mapping': None}
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)[s]
            else:
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s]}
                # Convert predictions
                postprocess_fun = None
                if (self.is_3DLabel):
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                predictions = \
                    self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)[s]

            if (self.is_text):
                if params_prediction.get('pos_unk', False):
                    samples = predictions[0]
                    alphas = predictions[1]

                    if eval('self.ds.loaded_raw_' + s + '[0]'):
                        sources = predictions[2]
                    else:
                        sources = []
                        for preds in predictions[2]:
                            for src in preds[self.input_text_id]:
                                sources.append(src)
                        sources = self.model_to_eval.decode_predictions_beam_search(sources,
                                                                                    self.index2word_x,
                                                                                    pad_sequences=True,
                                                                                    verbose=self.verbose)
                    heuristic = params_prediction['heuristic']
                else:
                    samples = predictions
                    alphas = None
                    heuristic = None
                    sources = None
                if self.out_pred_idx is not None:
                    samples = samples[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(samples,
                                                                                    self.index2word_y,
                                                                                    alphas=alphas,
                                                                                    x_text=sources,
                                                                                    heuristic=heuristic,
                                                                                    mapping=params_prediction['mapping'],
                                                                                    verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(predictions, 1,
                                                                        # always set temperature to 1
                                                                        self.index2word_y,
                                                                        self.sampling_type,
                                                                        verbose=self.verbose)

            # Store predictions
            if self.write_samples:
                # Store result
                filepath = self.save_path + '/' + s + '_' + counter_name + '_' + str(
                    epoch) + '.pred'  # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])
                elif self.write_type == 'listoflists':
                    listoflists2file(filepath, predictions)
                elif self.write_type == 'numpy':
                    numpy2file(filepath, predictions)
                elif self.write_type == '3DLabels':
                    # TODO:
                    print("WRITE SAMPLES FUNCTION NOT IMPLEMENTED")
                else:
                    raise NotImplementedError(
                        'The store type "' + self.write_type + '" is not implemented.')

            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric ' + metric)
                filepath = self.save_path + '/' + s + '.' + metric  # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                    pred_list=predictions,
                    verbose=self.verbose,
                    extra_vars=self.extra_vars,
                    split=s)

                # Print results to file and store in model log
                with open(filepath, 'a') as f:
                    header = counter_name + ','
                    line = str(epoch) + ','
                    # Store in model log
                    self.model_to_eval.log(s, counter_name, epoch)
                    for metric_ in sorted(metrics):
                        value = metrics[metric_]
                        header += metric_ + ', '
                        line += str(value) + ', '
                        # Store in model log
                        self.model_to_eval.log(s, metric_, value)
                    if epoch == 1 or epoch == self.start_eval_on_epoch:
                        f.write(header + '\n')
                    f.write(line + '\n')
                if self.verbose > 0:
                    logging.info('Done evaluating on metric ' + metric)

            """
            # Early stop check
            if self.early_stop and s in ['val', 'validation', 'dev', 'development']:
                current_score = metrics[self.stop_metric]
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    self.wait = 0
                    if self.verbose > 0:
                        logging.info(
                            '---current best %s: %.4f' % (self.stop_metric, current_score))
                else:
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            logging.info(
                                '%s %d: early stopping. Best %s value found at %s %d: %.4f' %
                                (str(counter_name), epoch, self.stop_metric, str(counter_name),
                                 self.best_epoch, self.best_score))
                            self.model.stop_training = True
                    self.wait += 1
                    if self.verbose > 0:
                        logging.info('----bad counter: %d/%d' % (self.wait, self.patience))
            """

###################################################
# Storing callbacks
###################################################
class StoreModelWeightsOnEpochEnd(KerasCallback):
    def __init__(self, model, fun, epochs_for_save, verbose=0):
        """
        In:
            model - model to save
            fun - function for saving the model
            epochs_for_save - number of epochs before the last save
        """
        super(KerasCallback, self).__init__()
        self.model_to_save = model
        self.store_function = fun
        self.epochs_for_save = epochs_for_save
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if(epoch%self.epochs_for_save==0):
            print('')
            self.store_function(self.model_to_save, epoch)

    #def on_batch_end(self, n_update, logs={}):
    #    n_update += 1
    #    if (n_update % self.epochs_for_save == 0):
    #        print('')
    #        self.store_function(self.model_to_save, n_update)
###

###################################################
# Sampling callbacks
###################################################

class SampleEachNUpdates(KerasCallback):

    def __init__(self, model, dataset, gt_id, set_name, n_samples, each_n_updates=10000, extra_vars=dict(),
                 is_text=False, index2word_y=None, input_text_id=None, sampling='max_likelihood',
                 beam_search=False, batch_size=50, reload_epoch=0, start_sampling_on_epoch=0,
                 write_type='list', sampling_type='max_likelihood', out_pred_idx=None, in_pred_idx=None, verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param n_samples: batch size used during sampling
            :param each_n_updates: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text
                            (in that case the data will be converted from values into a textual representation)
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param out_pred_idx: index of the output prediction used for evaluation
                            (only applicable if model has more than one output, else set to None)
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_sampling_on_epoch: only starts evaluating model if a given epoch has been reached
            :param in_pred_idx: index of the input prediction used for evaluation
                            (only applicable if model has more than one input, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_y = index2word_y
        self.input_text_id = input_text_id
        self.is_text = is_text
        self.sampling = sampling
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.set_name = set_name
        self.n_samples = n_samples
        self.each_n_updates = each_n_updates
        self.extra_vars = extra_vars
        self.reload_epoch = reload_epoch
        self.start_sampling_on_epoch = start_sampling_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.out_pred_idx = out_pred_idx
        self.in_pred_idx = in_pred_idx
        self.verbose = verbose

    def on_batch_end(self, n_update, logs={}):
        n_update += 1  # start by index 1
        n_update += self.reload_epoch
        if n_update < self.start_sampling_on_epoch:
            return
        elif n_update % self.each_n_updates != 0:
            return

        # Evaluate on each set separately
        for s in self.set_name:

            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size,
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s],
                                 'n_samples': self.n_samples}

            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions, truths, data = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)
            else:
                raise NotImplementedError()

            gt_y = eval('self.ds.Y_'+s+'["'+self.gt_id+'"]')
            predictions = predictions[s]
            if(self.is_text):
                if self.out_pred_idx is not None:
                    predictions = predictions[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(predictions,
                                                      self.index2word_y,
                                                      verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(predictions, 1, # always set temperature to 1
                                                      self.index2word_y,
                                                      self.sampling_type,
                                                      verbose=self.verbose)
                truths = self.model_to_eval.decode_predictions_one_hot(truths,
                                                      self.index2word_y,
                                                      verbose=self.verbose)
            # Write samples
            for i, (sample, truth) in enumerate(zip(predictions, truths)):
                print ("Hypothesis (%d): %s"%(i, sample))
                print ("Reference  (%d): %s"%(i, truth))


###################################################
# Learning modifiers callbacks
###################################################
class EarlyStopping(KerasCallback):
    """
    Applies early stopping if performance has not improved for some epochs.
    """
    def __init__(self, model, patience=0, check_split='val', metric_check='acc', verbose=1):
        """
        :param model: model to check performance
        :param patience: number of beginning epochs without reduction; by default 0 (disabled)
        :param check_split: data split used to check metric value improvement
        :param metric_check: name of the metric to check
        :param verbose: verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.model_to_eval = model
        self.patience = patience
        self.check_split = check_split
        self.metric_check = metric_check
        self.verbose = verbose

        # check already stored scores in case we have loaded a pre-trained model
        all_scores = self.model_to_eval.getLog(self.check_split, self.metric_check)
        if all_scores[-1] is not None:
            self.best_score = max(all_scores)
            self.best_epoch = all_scores.index(self.best_score)+1
            self.wait = len(all_scores) - self.best_epoch
        else:
            self.best_score = -1.
            self.best_epoch = -1
            self.wait = 0


    def on_epoch_end(self, epoch, logs={}):
        epoch += 1  # start by index 1
        # Get last metric value from logs
        current_score = self.model_to_eval.getLog(self.check_split, self.metric_check)[-1]
        if current_score is None:
            warnings.warn('The chosen metric'+str(self.metric_check)+' does not exist;'
                          ' this reducer works only with a valid metric.')
            return

        # Check if the best score has been outperformed in the current epoch
        if current_score > self.best_score:
            self.best_epoch = epoch
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                logging.info('---current best %s %s: %.3f' % (self.check_split, self.metric_check, current_score))

        # Stop training if performance has not improved for self.patience epochs
        elif self.patience > 0:
            self.wait += 1
            logging.info('---bad counter: %d/%d' % (self.wait, self.patience))
            if self.wait >= self.patience:
                if self.verbose > 0:
                    logging.info("---epoch %d: early stopping. Best %s found at epoch %d: %f" % (epoch, self.metric_check, self.best_epoch, self.best_score))
                self.model.stop_training = True


class LearningRateReducer(KerasCallback):
    """
    Reduces learning rate during the training.
    """

    def __init__(self, lr_decay=1, reduce_rate=0.5, reduce_nb=99999, verbose=1):
        """
        :param lr_decay: minimum number of epochs passed before the last reduction
        :param reduce_rate: multiplicative rate reducer; by default 0.5
        :param reduce_nb: maximal number of reductions performed; by default 99999
        :param verbose: verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
        self.epsilon = 0.1e-10
        self.lr_decay = lr_decay
        self.last_lr_decrease = 0

    def on_epoch_end(self, epoch, logs={}):

        # Decrease LR if self.lr_decay epochs have passed sice the last decrease
        self.last_lr_decrease += 1
        if self.last_lr_decrease >= self.lr_decay:
            self.current_reduce_nb += 1
            if self.current_reduce_nb <= self.reduce_nb:
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(np.float32(lr * self.reduce_rate))
                if self.verbose > 0:
                    logging.info("LR reduction from {0:0.6f} to {1:0.6f}". \
                                 format(float(lr), float(lr * self.reduce_rate)))
                if float(lr) <= self.epsilon:
                    if self.verbose > 0:
                        logging.info('Learning rate too small, learning stops now')
                    self.model.stop_training = True
