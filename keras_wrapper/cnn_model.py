# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import os
import shutil
import sys
import time
import logging
import numpy as np
import copy
from six import iteritems

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

import keras
import keras.backend as K
from keras.engine.training import Model
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Nadam, RMSprop, SGD, TFOptimizer

from keras_wrapper.dataset import Data_Batch_Generator, Homogeneous_Data_Batch_Generator, Parallel_Data_Batch_Generator
from keras_wrapper.extra.callbacks import LearningRateReducer, EarlyStopping, StoreModelWeightsOnEpochEnd
from keras_wrapper.extra.read_write import file2list, create_dir_if_not_exists
from keras_wrapper.utils import one_hot_2_indices, decode_predictions, decode_predictions_one_hot, \
    decode_predictions_beam_search, sampling, categorical_probas_to_classes, checkParameters, \
    print_dict
from keras_wrapper.search import beam_search

# These imports must be kept to ensure backwards compatibility
from keras_wrapper.saving import saveModel, loadModel, updateModel, transferWeights

# General setup of libraries
try:
    import cupy as cp
except:
    import numpy as cp

    logger.info('<<< Cupy not available. Using numpy. >>>')


class Model_Wrapper(object):
    """
        Wrapper for Keras' models. It provides the following utilities:
            - Training visualization module.
            - Model backups.
            - Easy to use training and test methods.
            - Seq2seq support.
    """

    def __init__(self,
                 model_type='basic_model',
                 silence=False,
                 structure_path=None,
                 weights_path=None,
                 seq_to_functional=False,
                 model_name=None,
                 plots_path=None,
                 models_path=None,
                 inheritance=False
                 ):
        """
            Model_Wrapper object constructor.

            :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                         Only valid if 'structure_path' is None.
            :param silence: set to True if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file.
                                   If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param seq_to_functional: indicates if we are loading a set of weights trained
                                      on a Sequential model to a Functional one
            :param model_name: optional name given to the network
                               (if None, then it will be assigned to current time as its name)
            :param plots_path: path to the folder where the plots will be stored during training
            :param models_path: path to the folder where the temporal model packups will be stored
            :param inheritance: indicates if we are building an instance from a child class
                                (in this case the model will not be built from this __init__,
                                it should be built from the child class).
        """
        self.silence = silence
        self.model_type = model_type
        self._dynamic_display = True

        # Dictionary for storing any additional data needed
        self.additional_data = dict()

        # Model containers
        self.model = None
        self.model_init = None
        self.model_next = None
        self.multi_gpu_model = None

        # Inputs and outputs names for models of class Model
        self.ids_inputs = list()
        self.ids_outputs = list()

        # Inputs and outputs names for models for optimized search
        self.ids_inputs_init = list()
        self.ids_outputs_init = list()
        self.ids_inputs_next = list()
        self.ids_outputs_next = list()

        # Matchings from model_init to mode_next:
        self.matchings_init_to_next = None
        self.matchings_next_to_next = None

        # Inputs and outputs names for models with temporally linked samples
        self.ids_temporally_linked_inputs = list()

        # Matchings between temporally linked samples
        self.matchings_sample_to_next_sample = None

        # Placeholders for model attributes
        self.inputsMapping = dict()
        self.outputsMapping = dict()
        self.acc_output = None
        self.name = None
        self.model_path = None
        self.plot_path = None
        self.params = None

        # Prepare logger
        self.updateLogger()

        self.default_training_params = dict()
        self.default_predict_with_beam_params = dict()
        self.default_test_params = dict()
        self.tensorboard_callback = None
        self.set_default_params()

        # Prepare model
        if not inheritance:
            # Set Network name
            self.setName(model_name,
                         plots_path,
                         models_path)

            if structure_path:
                # Load a .json model
                if not self.silence:
                    logger.info("<<< Loading model structure from file " + structure_path + " >>>")
                self.model = model_from_json(open(structure_path).read())

            else:
                # Build model from scratch
                if hasattr(self, model_type):
                    if not self.silence:
                        logger.info("<<< Building " + model_type + " Model_Wrapper >>>")
                    eval('self.' + model_type + '(nOutput, input_shape)')
                else:
                    raise Exception('Model_Wrapper model_type "' + model_type + '" is not implemented.')

            # Load weights from file
            if weights_path:
                if not self.silence:
                    logger.info("<<< Loading weights from file " + weights_path + " >>>")
                self.model.load_weights(weights_path,
                                        seq_to_functional=seq_to_functional)

    def __getstate__(self):
        """
            Behaviour applied when pickling a Model_Wrapper instance.
        """
        obj_dict = self.__dict__.copy()
        del obj_dict['model']
        # Remove also optimized search models if exist
        if 'model_init' in obj_dict:
            del obj_dict['model_init']
            del obj_dict['model_next']
        return obj_dict

    def __str__(self):
        """
        Plot basic model information.
        """
        keras.utils.layer_utils.print_summary(self.model.layers)
        return ''

    def updateLogger(self,
                     force=False):
        """
            Checks if the model contains an updated logger.
            If it doesn't then it updates it, which will store evaluation results.
        """
        compulsory_data_types = ['iteration', 'loss', 'accuracy', 'accuracy top-5']
        if '_Model_Wrapper__logger' not in self.__dict__ or force:
            self.__logger = dict()
        if '_Model_Wrapper__data_types' not in self.__dict__:
            self.__data_types = compulsory_data_types
        else:
            for d in compulsory_data_types:
                if d not in self.__data_types:
                    self.__data_types.append(d)

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)

        self.__modes = ['train',
                        'val',
                        'test']

    def set_default_params(self):
        """
        Sets the default params for training, decoding and testing a Model.

        :return:
        """
        self.default_training_params = {'n_epochs': 1,
                                        'batch_size': 50,
                                        'maxlen': 100,  # sequence learning parameters (BeamSearch)
                                        'homogeneous_batches': False,
                                        'joint_batches': 4,
                                        'epochs_for_save': 1,
                                        'num_iterations_val': None,
                                        'n_parallel_loaders': 1,
                                        'normalize': False,
                                        'normalization_type': None,
                                        'mean_substraction': False,
                                        'data_augmentation': False,
                                        'wo_da_patch_type': 'whole',  # wo_da_patch_type = 'central_crop' or 'whole'.
                                        'da_patch_type': 'resize_and_rndcrop',
                                        'da_enhance_list': [],
                                        # da_enhance_list = {brightness, color, sharpness, contrast}
                                        'verbose': 1,
                                        'eval_on_sets': 'val',
                                        'reload_epoch': 0,
                                        'extra_callbacks': [],
                                        'class_weights': None,
                                        'shuffle': True,
                                        'epoch_offset': 0,
                                        'patience': 0,
                                        'metric_check': None,
                                        'min_delta': 0.,
                                        'patience_check_split': 'val',
                                        'eval_on_epochs': True,
                                        'each_n_epochs': 1,
                                        'start_eval_on_epoch': 0,  # early stopping parameters
                                        'lr_decay': None,  # LR decay parameters
                                        'initial_lr': 1.,
                                        'reduce_each_epochs': True,
                                        'start_reduction_on_epoch': 0,
                                        'lr_gamma': 0.1,
                                        'lr_reducer_type': 'linear',
                                        'lr_reducer_exp_base': 0.5,
                                        'lr_half_life': 50000,
                                        'lr_warmup_exp': -1.5,
                                        'min_lr': 1e-9,
                                        'tensorboard': False,
                                        'n_gpus': 1,
                                        'tensorboard_params':
                                            {'log_dir': 'tensorboard_logs',
                                             'histogram_freq': 0,
                                             'batch_size': 50,
                                             'write_graph': True,
                                             'write_grads': False,
                                             'write_images': False,
                                             'embeddings_freq': 0,
                                             'embeddings_layer_names': None,
                                             'embeddings_metadata': None,
                                             'update_freq': 'epoch'
                                             }
                                        }
        self.defaut_test_params = {'batch_size': 50,
                                   'n_parallel_loaders': 1,
                                   'normalize': False,
                                   'normalization_type': None,
                                   'wo_da_patch_type': 'whole',
                                   'mean_substraction': False
                                   }

        self.default_predict_with_beam_params = {'max_batch_size': 50,
                                                 'n_parallel_loaders': 1,
                                                 'beam_size': 5,
                                                 'beam_batch_size': 50,
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
                                                 'search_pruning': False,
                                                 'pos_unk': False,
                                                 'temporally_linked': False,
                                                 'link_index_id': 'link_index',
                                                 'state_below_index': -1,
                                                 'state_below_maxlen': -1,
                                                 'max_eval_samples': None,
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
        self.default_predict_params = {'batch_size': 50,
                                       'n_parallel_loaders': 1,
                                       'normalize': False,
                                       'normalization_type': None,
                                       'wo_da_patch_type': 'whole',
                                       'mean_substraction': False,
                                       'n_samples': None,
                                       'init_sample': -1,
                                       'final_sample': -1,
                                       'verbose': 0,
                                       'predict_on_sets': ['val'],
                                       'max_eval_samples': None,
                                       'model_name': 'model',
                                       # name of the attribute where the model for prediction is stored
                                       }

    def setInputsMapping(self,
                         inputsMapping):
        """
            Sets the mapping of the inputs from the format given by the dataset to the format received by the model.

            :param inputsMapping: dictionary with the model inputs' identifiers as keys and the dataset inputs
                                  identifiers' position as values.
                                  If the current model is Sequential then keys must be ints with the desired input order
                                  (starting from 0). If it is Model then keys must be str.
        """
        self.inputsMapping = inputsMapping

    def setOutputsMapping(self,
                          outputsMapping,
                          acc_output=None):
        """
            Sets the mapping of the outputs from the format given by the dataset to the format received by the model.

            :param outputsMapping: dictionary with the model outputs'
                                   identifiers as keys and the dataset outputs identifiers' position as values.
                                   If the current model is Sequential then keys must be ints with
                                   the desired output order (in this case only one value can be provided).
                                   If it is Model then keys must be str.
            :param acc_output: name of the model's output that will be used for calculating
                              the accuracy of the model (only needed for Model models)
        """
        if isinstance(self.model,
                      Sequential) and len(list(outputsMapping)) > 1:
            raise Exception("When using Sequential models only one output can be provided in outputsMapping")
        self.outputsMapping = outputsMapping
        self.acc_output = acc_output

    def setOptimizer(self,
                     lr=None,
                     momentum=None,
                     loss='categorical_crossentropy',
                     loss_weights=None,
                     metrics=None,
                     epsilon=1e-8,
                     nesterov=True,
                     decay=0.0,
                     clipnorm=10.,
                     clipvalue=0.,
                     optimizer=None,
                     sample_weight_mode=None,
                     tf_optimizer=True):
        """
            Sets a new optimizer for the CNN model.
            :param nesterov:
            :param clipvalue:
            :param lr: learning rate of the network
            :param momentum: momentum of the network (if None, then momentum = 1-lr)
            :param loss: loss function applied for optimization
            :param loss_weights: weights given to multi-loss models
            :param metrics: list of Keras' metrics used for evaluating the model.
                            To specify different metrics for different outputs of a multi-output model,
                            you could also pass a dictionary, such as `metrics={'output_a': 'accuracy'}`.
            :param epsilon: fuzz factor
            :param decay: lr decay
            :param clipnorm: gradients' clip norm
            :param optimizer: string identifying the type of optimizer used (default: SGD)
            :param sample_weight_mode: 'temporal' or None
        """
        # Pick default parameters
        if lr is None:
            lr = self.lr
        else:
            self.lr = lr
        if momentum is None:
            momentum = self.momentum
        else:
            self.momentum = momentum
        self.loss = loss
        if metrics is None:
            metrics = []
        if tf_optimizer and K.backend() == 'tensorflow':
            import tensorflow as tf
            if optimizer is None or optimizer.lower() == 'sgd':
                if self.momentum is None:
                    optimizer = TFOptimizer(tf.train.GradientDescentOptimizer(lr))
                else:
                    optimizer = TFOptimizer(tf.train.MomentumOptimizer(lr,
                                                                       self.momentum,
                                                                       use_nesterov=nesterov))
            elif optimizer.lower() == 'adam':
                optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=lr,
                                                               epsilon=epsilon))
            elif optimizer.lower() == 'adagrad':
                optimizer = TFOptimizer(tf.train.AdagradOptimizer(lr))
            elif optimizer.lower() == 'rmsprop':
                optimizer = TFOptimizer(tf.train.RMSPropOptimizer(lr,
                                                                  decay=decay,
                                                                  momentum=momentum,
                                                                  epsilon=epsilon))
            elif optimizer.lower() == 'nadam':
                logger.warning('The Nadam optimizer is not natively implemented in Tensorflow. Using Keras optimizer.')
                optimizer = Nadam(lr=lr,
                                  clipnorm=clipnorm,
                                  clipvalue=clipvalue,
                                  decay=decay,
                                  epsilon=epsilon)
            elif optimizer.lower() == 'adamax':
                logger.warning('The Adamax optimizer is not natively implemented in Tensorflow. Using Keras optimizer.')
                optimizer = Adamax(lr=lr,
                                   clipnorm=clipnorm,
                                   clipvalue=clipvalue,
                                   decay=decay,
                                   epsilon=epsilon)
            elif optimizer.lower() == 'adadelta':
                optimizer = TFOptimizer(tf.train.AdadeltaOptimizer(learning_rate=lr,
                                                                   epsilon=epsilon))
            else:
                raise Exception('\tThe chosen optimizer is not implemented.')
        else:
            if optimizer is None or optimizer.lower() == 'sgd':
                optimizer = SGD(lr=lr,
                                clipnorm=clipnorm,
                                clipvalue=clipvalue,
                                decay=decay,
                                momentum=momentum,
                                nesterov=nesterov)
            elif optimizer.lower() == 'adam':
                optimizer = Adam(lr=lr,
                                 clipnorm=clipnorm,
                                 clipvalue=clipvalue,
                                 decay=decay,
                                 epsilon=epsilon)
            elif optimizer.lower() == 'adagrad':
                optimizer = Adagrad(lr=lr,
                                    clipnorm=clipnorm,
                                    clipvalue=clipvalue,
                                    decay=decay,
                                    epsilon=epsilon)
            elif optimizer.lower() == 'rmsprop':
                optimizer = RMSprop(lr=lr,
                                    clipnorm=clipnorm,
                                    clipvalue=clipvalue,
                                    decay=decay,
                                    epsilon=epsilon)
            elif optimizer.lower() == 'nadam':
                optimizer = Nadam(lr=lr,
                                  clipnorm=clipnorm,
                                  clipvalue=clipvalue,
                                  decay=decay,
                                  epsilon=epsilon)
            elif optimizer.lower() == 'adamax':
                optimizer = Adamax(lr=lr,
                                   clipnorm=clipnorm,
                                   clipvalue=clipvalue,
                                   decay=decay,
                                   epsilon=epsilon)
            elif optimizer.lower() == 'adadelta':
                optimizer = Adadelta(lr=lr,
                                     clipnorm=clipnorm,
                                     clipvalue=clipvalue,
                                     decay=decay,
                                     epsilon=epsilon)
            else:
                raise Exception('\tThe chosen optimizer is not implemented.')

        if not self.silence:
            logger.info("Compiling model...")

        # compile differently depending if our model is 'Sequential', 'Model'
        if isinstance(self.model, Sequential) or isinstance(self.model, Model):
            self.model.compile(optimizer=optimizer,
                               metrics=metrics,
                               loss=loss,
                               loss_weights=loss_weights,
                               sample_weight_mode=sample_weight_mode)
        else:
            raise NotImplementedError()

        if not self.silence:
            logger.info("Optimizer updated, learning rate set to " + str(lr))

    def set_tensorboard_callback(self,
                                 params):
        create_dir_if_not_exists(os.path.join(self.model_path,
                                              params['tensorboard_params']['log_dir']))
        self.tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.model_path,
                                 params['tensorboard_params']['log_dir']),
            histogram_freq=params['tensorboard_params']['histogram_freq'],
            batch_size=params['tensorboard_params']['batch_size'],
            write_graph=params['tensorboard_params']['write_graph'],
            write_grads=params['tensorboard_params']['write_grads'],
            write_images=params['tensorboard_params']['write_images'],
            embeddings_freq=params['tensorboard_params']['embeddings_freq'],
            embeddings_layer_names=params['tensorboard_params']['embeddings_layer_names'],
            embeddings_metadata=params['tensorboard_params']['embeddings_metadata'],
            update_freq=params['tensorboard_params']['update_freq'],
        )

    def compile(self,
                **kwargs):
        """
        Compile the model.
        :param kwargs:
        :return:
        """
        self.model.compile(kwargs)

    def setName(self,
                model_name,
                plots_path=None,
                models_path=None,
                create_plots=False,
                clear_dirs=True):
        """
                    Changes the name (identifier) of the Model_Wrapper instance.
        :param model_name:  New model name
        :param plots_path: Path where to store the plots
        :param models_path: Path where to store the model
        :param create_plots: Whether we'll store plots or not
        :param clear_dirs: Whether the store_path directory will be erased or not
        :return: None
        """
        if not model_name:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if create_plots:
            if not plots_path:
                self.plot_path = 'Plots/' + self.name
            else:
                self.plot_path = plots_path

        if not models_path:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = models_path

        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)
            if create_plots:
                if os.path.isdir(self.plot_path):
                    shutil.rmtree(self.plot_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)
            if create_plots:
                if not os.path.isdir(self.plot_path):
                    os.makedirs(self.plot_path)

    def setParams(self,
                  params):
        """
        Set self.params as params.
        :param params:
        :return:
        """
        self.params = params

    # ------------------------------------------------------- #
    #       MODEL MODIFICATION
    #           Methods for modifying specific layers of the network
    # ------------------------------------------------------- #

    # ------------------------------------------------------- #
    #       TRAINING/TEST
    #           Methods for train and testing on the current Model_Wrapper
    # ------------------------------------------------------- #

    def ended_training(self):
        """
            Indicates if the model has early stopped.
        """
        if hasattr(self.model, 'callback_model') and self.model.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        if hasattr(callback_model, 'stop_training') and callback_model.stop_training:
            return True
        else:
            return False

    def trainNet(self,
                 ds,
                 parameters=None,
                 out_name=None):
        """
            Trains the network on the given dataset.
            :param ds: Dataset with the training data
            :param parameters: dict() which may contain the following (optional) training parameters
            :param out_name: name of the output node that will be used to evaluate the network accuracy.
                            Only applicable to Graph models.

            The input 'parameters' is a dict() which may contain the following (optional) training parameters:
            Visualization parameters
             * report_iter: number of iterations between each loss report
             * iter_for_val: number of iterations between each validation test
             * num_iterations_val: number of iterations applied on the validation dataset for computing the
                                   average performance (if None then all the validation data will be tested)
            Learning parameters
             * n_epochs: number of epochs that will be applied during training
             * batch_size: size of the batch (number of images) applied on each iteration by the SGD optimization
             * lr_decay: number of iterations passed for decreasing the learning rate
             * lr_gamma: proportion of learning rate kept at each decrease.
                         It can also be a set of rules defined by a list, e.g.
                         lr_gamma = [[3000, 0.9], ..., [None, 0.8]] means 0.9 until iteration
                         3000, ..., 0.8 until the end.
             * patience: number of epochs waiting for a possible performance increase before stopping training
             * metric_check: name of the metric checked for early stopping and LR decrease

            Data processing parameters

             * n_parallel_loaders: number of parallel data loaders allowed to work at the same time
             * normalize: boolean indicating if we want to normalize the image pixel values
             * mean_substraction: boolean indicating if we want to substract the training mean
             * data_augmentation: boolean indicating if we want to perform data augmentation
                                  (always False on validation)
             * shuffle: apply shuffling on training data at the beginning of each epoch.

            Other parameters

        """

        # Check input parameters and recover default values if needed
        if parameters is None:
            parameters = dict()
        params = checkParameters(parameters,
                                 self.default_training_params,
                                 hard_check=True)
        # Set params['start_reduction_on_epoch'] = params['lr_decay'] by default
        if params['lr_decay'] is not None and 'start_reduction_on_epoch' not in list(parameters):
            params['start_reduction_on_epoch'] = params['lr_decay']
        save_params = copy.copy(params)
        del save_params['extra_callbacks']
        if params['verbose'] > 0:
            logger.info("<<< Training model >>>")

        self.__train(ds,
                     params)

        logger.info("<<< Finished training model >>>")

    def trainNetFromSamples(self,
                            x,
                            y,
                            parameters=None,
                            class_weight=None,
                            sample_weight=None,
                            out_name=None):
        """
            Trains the network on the given samples x, y.

            :param x:
            :param y:
            :param parameters:
            :param class_weight:
            :param sample_weight:
            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable to Models.

            The input 'parameters' is a dict() which may contain the following (optional) training parameters:
                Visualization parameters
                Learning parameters
                Data processing parameters
                Other parameters

        """

        # Check input parameters and recover default values if needed
        if parameters is None:
            parameters = dict()
        params = checkParameters(parameters,
                                 self.default_training_params,
                                 hard_check=True)
        save_params = copy.copy(params)
        del save_params['extra_callbacks']
        self.__train_from_samples(x,
                                  y,
                                  params,
                                  class_weight=class_weight,
                                  sample_weight=sample_weight)
        if params['verbose'] > 0:
            logger.info("<<< Finished training model >>>")

    def __train(self,
                ds,
                params,
                state=None):

        if state is None:
            state = dict()
        if params['verbose'] > 0:
            logger.info(print_dict(params,
                                   header="Training parameters: "))

        # initialize state
        state['samples_per_epoch'] = ds.len_train
        state['n_iterations_per_epoch'] = int(math.ceil(float(state['samples_per_epoch']) / params['batch_size']))

        # Prepare callbacks
        callbacks = []

        # Extra callbacks (e.g. evaluation)
        callbacks += params['extra_callbacks']

        # LR reducer
        if params.get('lr_decay') is not None:
            callback_lr_reducer = LearningRateReducer(initial_lr=params['initial_lr'],
                                                      reduce_rate=params['lr_gamma'],
                                                      reduce_frequency=params['lr_decay'],
                                                      reduce_each_epochs=params['reduce_each_epochs'],
                                                      start_reduction_on_epoch=params['start_reduction_on_epoch'],
                                                      exp_base=params['lr_reducer_exp_base'],
                                                      half_life=params['lr_half_life'],
                                                      warmup_exp=params['lr_warmup_exp'],
                                                      reduction_function=params['lr_reducer_type'],
                                                      min_lr=params['min_lr'],
                                                      verbose=params['verbose'])
            callbacks.append(callback_lr_reducer)
        # Early stopper
        if params.get('metric_check') is not None:
            callback_early_stop = EarlyStopping(self,
                                                patience=params['patience'],
                                                metric_check=params['metric_check'],
                                                want_to_minimize=True if 'TER' in params['metric_check'] else False,
                                                min_delta=params['min_delta'],
                                                check_split=params['patience_check_split'],
                                                eval_on_epochs=params['eval_on_epochs'],
                                                each_n_epochs=params['each_n_epochs'],
                                                start_eval_on_epoch=params['start_eval_on_epoch'])
            callbacks.append(callback_early_stop)

        # Store model
        if params['epochs_for_save'] >= 0:
            callback_store_model = StoreModelWeightsOnEpochEnd(self,
                                                               saveModel,
                                                               params['epochs_for_save'])
            callbacks.insert(0,
                             callback_store_model)

        # Tensorboard callback
        if params['tensorboard'] and K.backend() == 'tensorflow':
            self.set_tensorboard_callback(params)
            self.tensorboard_callback.set_model(self.model)
            callbacks.append(self.tensorboard_callback)

        # Prepare data generators
        if params['homogeneous_batches']:
            train_gen = Homogeneous_Data_Batch_Generator('train',
                                                         self,
                                                         ds,
                                                         state['n_iterations_per_epoch'],
                                                         batch_size=params['batch_size'],
                                                         joint_batches=params['joint_batches'],
                                                         normalization=params['normalize'],
                                                         normalization_type=params['normalization_type'],
                                                         data_augmentation=params['data_augmentation'],
                                                         wo_da_patch_type=params['wo_da_patch_type'],
                                                         da_patch_type=params['da_patch_type'],
                                                         da_enhance_list=params['da_enhance_list'],
                                                         mean_substraction=params['mean_substraction']).generator()
        elif params['n_parallel_loaders'] > 1:
            train_gen = Parallel_Data_Batch_Generator('train',
                                                      self,
                                                      ds,
                                                      state['n_iterations_per_epoch'],
                                                      batch_size=params['batch_size'],
                                                      normalization=params['normalize'],
                                                      normalization_type=params['normalization_type'],
                                                      data_augmentation=params['data_augmentation'],
                                                      wo_da_patch_type=params['wo_da_patch_type'],
                                                      da_patch_type=params['da_patch_type'],
                                                      da_enhance_list=params['da_enhance_list'],
                                                      mean_substraction=params['mean_substraction'],
                                                      shuffle=params['shuffle'],
                                                      n_parallel_loaders=params['n_parallel_loaders']).generator()
        else:
            train_gen = Data_Batch_Generator('train',
                                             self,
                                             ds,
                                             state['n_iterations_per_epoch'],
                                             batch_size=params['batch_size'],
                                             normalization=params['normalize'],
                                             normalization_type=params['normalization_type'],
                                             data_augmentation=params['data_augmentation'],
                                             wo_da_patch_type=params['wo_da_patch_type'],
                                             da_patch_type=params['da_patch_type'],
                                             da_enhance_list=params['da_enhance_list'],
                                             mean_substraction=params['mean_substraction'],
                                             shuffle=params['shuffle']).generator()

        # Are we going to validate on 'val' data?
        if params['eval_on_sets']:
            # Calculate how many validation iterations are we going to perform per test
            n_valid_samples = ds.len_val
            if params['num_iterations_val'] is None:
                params['num_iterations_val'] = int(math.ceil(float(n_valid_samples) / params['batch_size']))

            # prepare data generator
            if params['n_parallel_loaders'] > 1:
                val_gen = Parallel_Data_Batch_Generator(params['eval_on_sets'],
                                                        self,
                                                        ds,
                                                        params['num_iterations_val'],
                                                        batch_size=params['batch_size'],
                                                        normalization=params['normalize'],
                                                        normalization_type=params['normalization_type'],
                                                        data_augmentation=False,
                                                        mean_substraction=params['mean_substraction'],
                                                        shuffle=False,
                                                        n_parallel_loaders=params['n_parallel_loaders']).generator()
            else:
                val_gen = Data_Batch_Generator(params['eval_on_sets'],
                                               self,
                                               ds,
                                               params['num_iterations_val'],
                                               batch_size=params['batch_size'],
                                               normalization=params['normalize'],
                                               normalization_type=params['normalization_type'],
                                               data_augmentation=False,
                                               mean_substraction=params['mean_substraction'],
                                               shuffle=False).generator()
        else:
            val_gen = None
            n_valid_samples = None

        # Are we going to use class weights?
        class_weight = {}
        if params['class_weights'] is not None:
            class_weight = ds.extra_variables['class_weights_' + params['class_weights']]
        # Train model
        if params.get('n_gpus', 1) > 1 and self.multi_gpu_model is not None:
            model_to_train = self.multi_gpu_model
        else:
            model_to_train = self.model

        if int(keras.__version__.split('.')[0]) == 1:
            # Keras 1.x version
            model_to_train.fit_generator(train_gen,
                                         validation_data=val_gen,
                                         nb_val_samples=n_valid_samples,
                                         class_weight=class_weight,
                                         samples_per_epoch=state['samples_per_epoch'],
                                         nb_epoch=params['n_epochs'],
                                         max_q_size=params['n_parallel_loaders'],
                                         verbose=params['verbose'],
                                         callbacks=callbacks,
                                         initial_epoch=params['epoch_offset'])
        else:
            # Keras 2.x version
            model_to_train.fit_generator(train_gen,
                                         steps_per_epoch=state['n_iterations_per_epoch'],
                                         epochs=params['n_epochs'],
                                         verbose=params['verbose'],
                                         callbacks=callbacks,
                                         validation_data=val_gen,
                                         validation_steps=n_valid_samples,
                                         validation_freq=params['each_n_epochs'],
                                         class_weight=class_weight,
                                         max_queue_size=params['n_parallel_loaders'],
                                         workers=1,
                                         initial_epoch=params['epoch_offset'])

    def __train_from_samples(self,
                             x,
                             y,
                             params,
                             class_weight=None,
                             sample_weight=None):

        if params['verbose'] > 0:
            logger.info(print_dict(params,
                                   header="Training parameters: "))

        callbacks = []

        # Extra callbacks (e.g. evaluation)
        callbacks += params['extra_callbacks']

        # LR reducer
        if params.get('lr_decay') is not None:
            callback_lr_reducer = LearningRateReducer(initial_lr=params['initial_lr'],
                                                      reduce_rate=params['lr_gamma'],
                                                      reduce_frequency=params['lr_decay'],
                                                      reduce_each_epochs=params['reduce_each_epochs'],
                                                      start_reduction_on_epoch=params['start_reduction_on_epoch'],
                                                      exp_base=params['lr_reducer_exp_base'],
                                                      half_life=params['lr_half_life'],
                                                      warmup_exp=params['lr_warmup_exp'],
                                                      reduction_function=params['lr_reducer_type'],
                                                      verbose=params['verbose'])
            callbacks.append(callback_lr_reducer)

        # Early stopper
        if params.get('metric_check') is not None:
            callback_early_stop = EarlyStopping(self,
                                                patience=params['patience'],
                                                metric_check=params['metric_check'],
                                                want_to_minimize=True if 'TER' in params['metric_check'] else False,
                                                min_delta=params['min_delta'],
                                                eval_on_epochs=params['eval_on_epochs'],
                                                each_n_epochs=params['each_n_epochs'],
                                                start_eval_on_epoch=params['start_eval_on_epoch'])
            callbacks.append(callback_early_stop)

        # Store model
        if params['epochs_for_save'] >= 0:
            callback_store_model = StoreModelWeightsOnEpochEnd(self,
                                                               saveModel,
                                                               params['epochs_for_save'])
            callbacks.append(callback_store_model)

        # Tensorboard callback
        if params['tensorboard'] and K.backend() == 'tensorflow':
            self.set_tensorboard_callback(params)
            self.tensorboard_callback.set_model(self.model)
            callbacks.append(self.tensorboard_callback)

        # Train model
        if params.get('n_gpus', 1) > 1 and hasattr(self, 'model_to_train'):
            model_to_train = self.model_to_train
        else:
            model_to_train = self.model

        # Train model
        model_to_train.fit(x,
                           y,
                           batch_size=min(params['batch_size'], len(x[0])),
                           epochs=params['n_epochs'],
                           verbose=params['verbose'],
                           callbacks=callbacks,
                           validation_data=None,
                           validation_split=params.get('val_split', 0.),
                           shuffle=params['shuffle'],
                           class_weight=class_weight,
                           sample_weight=sample_weight,
                           initial_epoch=params['epoch_offset'])

    def testNet(self,
                ds,
                parameters,
                out_name=None):
        """
        Evaluate the model on a given split.
        :param ds: Dataset
        :param parameters: Parameters
        :param out_name: Deprecated.
        :return:
        """
        # Check input parameters and recover default values if needed
        params = checkParameters(parameters,
                                 self.defaut_test_params)

        logger.info("<<< Testing model >>>")

        # Calculate how many test iterations are we going to perform
        n_samples = ds.len_test
        num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))

        # Test model
        # We won't use an Homogeneous_Batch_Generator for testing
        if params['n_parallel_loaders'] > 1:
            data_gen = Parallel_Data_Batch_Generator('test',
                                                     self,
                                                     ds,
                                                     num_iterations,
                                                     batch_size=params['batch_size'],
                                                     normalization=params['normalize'],
                                                     normalization_type=params['normalization_type'],
                                                     data_augmentation=False,
                                                     wo_da_patch_type=params['wo_da_patch_type'],
                                                     mean_substraction=params['mean_substraction'],
                                                     n_parallel_loaders=params['n_parallel_loaders']).generator()
        else:
            data_gen = Data_Batch_Generator('test',
                                            self,
                                            ds,
                                            num_iterations,
                                            batch_size=params['batch_size'],
                                            normalization=params['normalize'],
                                            normalization_type=params['normalization_type'],
                                            data_augmentation=False,
                                            wo_da_patch_type=params['wo_da_patch_type'],
                                            mean_substraction=params['mean_substraction']).generator()

        out = self.model.evaluate_generator(data_gen,
                                            val_samples=n_samples,
                                            max_q_size=params['n_parallel_loaders'],
                                            nb_worker=1,
                                            # params['n_parallel_loaders'],
                                            pickle_safe=False,
                                            )

        # Display metrics results
        for name, o in zip(self.model.metrics_names, out):
            logger.info('test ' + name + ': %0.8s' % o)

    def testNetSamples(self,
                       X,
                       batch_size=50):
        """
            Applies a forward pass on the samples provided and returns the predicted classes and probabilities.
        """
        classes = self.model.predict_classes(X,
                                             batch_size=batch_size)
        probs = self.model.predict_proba(X,
                                         batch_size=batch_size)

        return [classes, probs]

    def testOnBatch(self,
                    X,
                    Y,
                    accuracy=True,
                    out_name=None):
        """
            Applies a test on the samples provided and returns the resulting loss and accuracy (if True).

            :param X:
            :param Y:
            :param accuracy:
            :param out_name: name of the output node that will be used to evaluate the network accuracy.
                            Only applicable for Graph models.
        """
        n_samples = X.shape[1]
        if isinstance(self.model, Sequential) or isinstance(self.model, Model):
            [X, Y] = self._prepareSequentialData(X, Y)
            loss = self.model.test_on_batch(X, Y)
            loss = loss[0]
            if accuracy:
                [score, top_score] = self._getSequentialAccuracy(Y, self.model.predict_on_batch(X)[0])
                return loss, score, top_score, n_samples
            return loss, n_samples
        else:
            [data, last_output] = self._prepareModelData(X, Y)
            loss = self.model.test_on_batch(data)
            loss = loss[0]
            if accuracy:
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if out_name:
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]
                return loss, score, top_score, n_samples
            return loss, n_samples

    # ------------------------------------------------------- #
    #       PREDICTION FUNCTIONS
    #           Functions for making prediction on input samples
    # ------------------------------------------------------- #
    def predict_cond(self,
                     X,
                     states_below,
                     params,
                     ii):
        """
        Returns predictions on batch given the (static) input X and the current history (states_below) at time-step ii.
        WARNING!: It's assumed that the current history (state_below) is the last input of the model!
        See Dataset class for more information
        :param X: Input context
        :param states_below: Batch of partial hypotheses
        :param params: Decoding parameters
        :param ii: Decoding time-step
        :return: Network predictions at time-step ii
        """
        in_data = {}
        n_samples = states_below.shape[0]
        # Choose model to use for sampling
        model = self.model
        for model_input in params['model_inputs']:
            if X[model_input].shape[0] == 1:
                in_data[model_input] = np.repeat(X[model_input],
                                                 n_samples,
                                                 axis=0)
            else:
                in_data[model_input] = X[model_input]

        in_data[params['model_inputs'][params['state_below_index']]] = states_below
        # Recover output identifiers

        # in any case, the first output of the models must be the next words' probabilities
        output_ids_list = params['model_outputs']
        pick_idx = ii

        # Apply prediction on current timestep
        if params['max_batch_size'] >= n_samples:  # The model inputs beam will fit into one batch in memory
            out_data = model.predict_on_batch(in_data)
        else:  # It is possible that the model inputs don't fit into one single batch: Make one-sample-sized batches
            for i in range(n_samples):
                aux_in_data = {}
                for k, v in iteritems(in_data):
                    aux_in_data[k] = np.expand_dims(v[i], axis=0)
                predicted_out = model.predict_on_batch(aux_in_data)
                if i == 0:
                    out_data = predicted_out
                else:
                    if len(output_ids_list) > 1:
                        for iout in range(len(output_ids_list)):
                            out_data[iout] = np.vstack((out_data[iout], predicted_out[iout]))
                    else:
                        out_data = np.vstack((out_data, predicted_out))

        # Get outputs
        if len(output_ids_list) > 1:
            all_data = {}
            for output_id in range(len(output_ids_list)):
                all_data[output_ids_list[output_id]] = out_data[output_id]
            all_data[output_ids_list[0]] = all_data[output_ids_list[0]][:, pick_idx, :]
        else:
            all_data = {output_ids_list[0]: out_data[:, pick_idx, :]}

        # Define returned data
        probs = all_data[output_ids_list[0]]

        return probs

    def predict_cond_optimized(self,
                               X,
                               states_below,
                               params,
                               ii,
                               prev_out):
        """
        Returns predictions on batch given the (static) input X and the current history (states_below) at time-step ii.
        WARNING!: It's assumed that the current history (state_below) is the last input of the model!
        See Dataset class for more information
        :param X: Input context
        :param states_below: Batch of partial hypotheses
        :param params: Decoding parameters
        :param ii: Decoding time-step
        :param prev_out: output from the previous timestep, which will be reused by self.model_next
        (only applicable if beam search specific models self.model_init and self.model_next models are defined)
        :return: Network predictions at time-step ii
        """
        in_data = {}
        n_samples = states_below.shape[0]

        ##########################################
        # Choose model to use for sampling
        ##########################################
        if ii == 0:
            model = self.model_init
        else:
            model = self.model_next
        ##########################################
        # Get inputs
        ##########################################
        if ii > 1:  # timestep > 1 (model_next to model_next)
            for idx, next_out_name in list(enumerate(self.ids_outputs_next)):
                if idx == 0:
                    if params.get('attend_on_output', False):
                        if params.get('pad_on_batch', True):
                            pass
                    else:
                        if params.get('pad_on_batch', True):
                            states_below = states_below[:, -1].reshape(n_samples, -1)
                    in_data[self.ids_inputs_next[0]] = states_below
                if idx > 0:  # first output must be the output probs.
                    if next_out_name in list(self.matchings_next_to_next):
                        next_in_name = self.matchings_next_to_next[next_out_name]
                        if prev_out[idx].shape[0] == 1:
                            prev_out[idx] = np.repeat(prev_out[idx], n_samples, axis=0)
                        in_data[next_in_name] = prev_out[idx]
        elif ii == 0:  # first timestep
            for model_input in params['model_inputs']:
                if X[model_input].shape[0] == 1:
                    in_data[model_input] = np.repeat(X[model_input], n_samples, axis=0)
                else:
                    in_data[model_input] = X[model_input]
                if params.get('pad_on_batch', True):
                    states_below = states_below.reshape(n_samples, -1)
            in_data[params['model_inputs'][params['state_below_index']]] = states_below

        elif ii == 1:  # timestep == 1 (model_init to model_next)
            for idx, init_out_name in list(enumerate(self.ids_outputs_init)):
                if idx == 0:
                    if params.get('attend_on_output', False):
                        if params.get('pad_on_batch', True):
                            pass
                    else:
                        if params.get('pad_on_batch', True):
                            states_below = states_below[:, -1].reshape(n_samples, -1)
                    in_data[self.ids_inputs_next[0]] = states_below

                if idx > 0:  # first output must be the output probs.
                    if init_out_name in list(self.matchings_init_to_next):
                        next_in_name = self.matchings_init_to_next[init_out_name]
                        if prev_out[idx].shape[0] == 1:
                            prev_out[idx] = np.repeat(prev_out[idx], n_samples, axis=0)
                        in_data[next_in_name] = prev_out[idx]

        ##########################################
        # Recover output identifiers
        ##########################################
        # in any case, the first output of the models must be the next words' probabilities
        pick_idx = ii if params.get('attend_on_output', False) else 0
        if ii == 0:  # optimized search model (model_init case)
            output_ids_list = self.ids_outputs_init
        else:  # optimized search model (model_next case)
            output_ids_list = self.ids_outputs_next

        ##########################################
        # Apply prediction on current timestep
        ##########################################
        if params['max_batch_size'] >= n_samples:  # The model inputs beam will fit into one batch in memory
            out_data = model.predict_on_batch(in_data)
        else:
            # It is possible that the model inputs don't fit into one single batch:
            #  Make beam_batch_size-sample-sized batches
            for i in range(0, n_samples, params['beam_batch_size']):
                aux_in_data = {}
                for k, v in iteritems(in_data):
                    max_pos = min([i + params['beam_batch_size'], n_samples, len(v)])
                    aux_in_data[k] = v[i:max_pos]
                    # aux_in_data[k] = np.expand_dims(v[i], axis=0)
                predicted_out = model.predict_on_batch(aux_in_data)
                if i == 0:
                    out_data = predicted_out
                else:
                    if len(output_ids_list) > 1:
                        for iout in range(len(output_ids_list)):
                            out_data[iout] = np.vstack((out_data[iout], predicted_out[iout]))
                    else:
                        out_data = np.vstack((out_data, predicted_out))

        ##########################################
        # Get outputs
        ##########################################
        if len(output_ids_list) > 1:
            all_data = {}
            for output_id in range(len(output_ids_list)):
                all_data[output_ids_list[output_id]] = cp.asarray(out_data[output_id])
            all_data[output_ids_list[0]] = cp.asarray(all_data[output_ids_list[0]][:, pick_idx, :])
        else:
            all_data = {output_ids_list[0]: cp.asarray(out_data[:, pick_idx, :])}
        probs = cp.asarray(all_data[output_ids_list[0]])

        ##########################################
        # Define returned data
        ##########################################
        return [probs, out_data]

    def beam_search(self,
                    X,
                    params,
                    return_alphas=False,
                    eos_sym=0,
                    null_sym=2):
        """
        DEPRECATED, use search.beam_search instead.
        """
        logger.warning("Deprecated function, use search.beam_search instead.")
        return beam_search(self,
                           X,
                           params,
                           return_alphas=return_alphas,
                           eos_sym=eos_sym,
                           null_sym=null_sym)

    def BeamSearchNet(self,
                      ds,
                      parameters):
        """
        DEPRECATED, use predictBeamSearchNet() instead.
        """
        logger.warning("Deprecated function, use predictBeamSearchNet() instead.")
        return self.predictBeamSearchNet(ds,
                                         parameters)

    def predictBeamSearchNet(self,
                             ds,
                             parameters=None):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        The following attributes must be inserted to the model when building a temporally_linked model:

            * matchings_sample_to_next_sample:
            * ids_temporally_linked_inputs:

        :param ds:
        :param parameters:
        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """
        if parameters is None:
            parameters = dict()
        # Check input parameters and recover default values if needed
        params = checkParameters(parameters,
                                 self.default_predict_with_beam_params)
        # Check if the model is ready for applying an optimized search
        if params['optimized_search']:
            if 'matchings_init_to_next' not in dir(self) or \
                    'matchings_next_to_next' not in dir(self) or \
                    'ids_inputs_init' not in dir(self) or \
                    'ids_outputs_init' not in dir(self) or \
                    'ids_inputs_next' not in dir(self) or \
                    'ids_outputs_next' not in dir(self):
                raise Exception(
                    "The following attributes must be inserted to the model when building an optimized search model:\n",
                    "- matchings_init_to_next\n",
                    "- matchings_next_to_next\n",
                    "- ids_inputs_init\n",
                    "- ids_outputs_init\n",
                    "- ids_inputs_next\n",
                    "- ids_outputs_next\n")

        # Check if the model is ready for applying a temporally_linked search
        if params['temporally_linked']:
            if 'matchings_sample_to_next_sample' not in dir(self) or \
                    'ids_temporally_linked_inputs' not in dir(self):
                raise Exception(
                    "The following attributes must be inserted to the model when building a temporally_linked model:\n",
                    "- matchings_sample_to_next_sample\n",
                    "- ids_temporally_linked_inputs\n")
        predictions = dict()
        references = []
        sources_sampling = []
        for s in params['predict_on_sets']:
            print("")
            print("",
                  file=sys.stderr)
            logger.info("<<< Predicting outputs of " + s + " set >>>")

            # TODO: enable 'train' sampling on temporally-linked models
            if params['temporally_linked'] and s == 'train':
                logger.info('Sampling is currently not implemented on the "train" set for temporally-linked models.')
                data_gen = -1
                data_gen_instance = -1
            else:
                if len(params['model_inputs']) == 0:
                    raise AssertionError('We need at least one input!')
                if not params['optimized_search']:  # use optimized search model if available
                    if params['pos_unk']:
                        raise AssertionError('PosUnk is not supported with non-optimized beam search methods')

                params['pad_on_batch'] = ds.pad_on_batch[params['dataset_inputs'][params['state_below_index']]]

                if params['temporally_linked']:
                    previous_outputs = {}  # variable for storing previous outputs if using a temporally-linked model
                    for input_id in self.ids_temporally_linked_inputs:
                        previous_outputs[input_id] = dict()
                        previous_outputs[input_id][-1] = [ds.extra_words['<null>']]

                # Calculate how many iterations are we going to perform
                if params['n_samples'] < 1:
                    if params['max_eval_samples'] is not None:
                        n_samples = min(eval("ds.len_" + s),
                                        params['max_eval_samples'])
                    else:
                        n_samples = eval("ds.len_" + s)

                    num_iterations = int(math.ceil(float(n_samples)))  # / params['max_batch_size']))
                    n_samples = min(eval("ds.len_" + s),
                                    num_iterations)  # * params['batch_size'])
                    # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                    if params['n_parallel_loaders'] > 1:
                        data_gen_instance = Parallel_Data_Batch_Generator(s,
                                                                          self,
                                                                          ds,
                                                                          num_iterations,
                                                                          batch_size=1,
                                                                          normalization=params['normalize'],
                                                                          normalization_type=params[
                                                                              'normalization_type'],
                                                                          data_augmentation=False,
                                                                          mean_substraction=params['mean_substraction'],
                                                                          predict=True,
                                                                          n_parallel_loaders=params[
                                                                              'n_parallel_loaders'])
                    else:
                        data_gen_instance = Data_Batch_Generator(s,
                                                                 self,
                                                                 ds,
                                                                 num_iterations,
                                                                 batch_size=1,
                                                                 normalization=params['normalize'],
                                                                 normalization_type=params['normalization_type'],
                                                                 data_augmentation=False,
                                                                 mean_substraction=params['mean_substraction'],
                                                                 predict=True)
                    data_gen = data_gen_instance.generator()
                else:
                    n_samples = params['n_samples']
                    num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                    # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                    if params['n_parallel_loaders'] > 1:
                        data_gen_instance = Parallel_Data_Batch_Generator(s,
                                                                          self,
                                                                          ds,
                                                                          num_iterations,
                                                                          batch_size=1,
                                                                          normalization=params['normalize'],
                                                                          normalization_type=params[
                                                                              'normalization_type'],
                                                                          data_augmentation=False,
                                                                          mean_substraction=params['mean_substraction'],
                                                                          predict=False,
                                                                          random_samples=n_samples,
                                                                          temporally_linked=params['temporally_linked'],
                                                                          n_parallel_loaders=params[
                                                                              'n_parallel_loaders'])
                    else:
                        data_gen_instance = Data_Batch_Generator(s,
                                                                 self,
                                                                 ds,
                                                                 num_iterations,
                                                                 batch_size=1,
                                                                 normalization=params['normalize'],
                                                                 normalization_type=params['normalization_type'],
                                                                 data_augmentation=False,
                                                                 mean_substraction=params['mean_substraction'],
                                                                 predict=False,
                                                                 random_samples=n_samples,
                                                                 temporally_linked=params['temporally_linked'])
                    data_gen = data_gen_instance.generator()

                if params['n_samples'] > 0:
                    references = []
                    sources_sampling = []
                best_samples = []
                best_scores = np.array([])
                best_alphas = []
                sources = []
                sampled = 0
                start_time = time.time()
                eta = -1
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
                        if params['pos_unk'] and not eval('ds.loaded_raw_' + s + '[0]'):
                            sources.append(s_dict)

                    for i in range(len(X[params['model_inputs'][0]])):  # process one sample at a time
                        sampled += 1

                        sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                        if not hasattr(self, '_dynamic_display') or self._dynamic_display:
                            sys.stdout.write('\r')
                        else:
                            sys.stdout.write('\n')

                        sys.stdout.flush()
                        x = dict()

                        for input_id in params['model_inputs']:
                            if params['temporally_linked'] and input_id in self.ids_temporally_linked_inputs:
                                link = int(X[params['link_index_id']][i])
                                if link not in list(previous_outputs[input_id]):
                                    # input to current sample was not processed yet
                                    link = -1
                                prev_x = [ds.vocabulary[input_id]['idx2words'][w] for w in
                                          previous_outputs[input_id][link]]
                                x[input_id] = ds.loadText([' '.join(prev_x)],
                                                          ds.vocabulary[input_id],
                                                          ds.max_text_len[input_id][s],
                                                          ds.text_offset[input_id],
                                                          fill=ds.fill_text[input_id],
                                                          pad_on_batch=ds.pad_on_batch[input_id],
                                                          words_so_far=ds.words_so_far[input_id],
                                                          loading_X=True)[0]
                            else:
                                x[input_id] = np.asarray([X[input_id][i]])
                        samples, scores, alphas = beam_search(self,
                                                              x,
                                                              params,
                                                              eos_sym=ds.extra_words['<pad>'],
                                                              null_sym=ds.extra_words['<null>'],
                                                              return_alphas=params['coverage_penalty'])

                        if params['length_penalty'] or params['coverage_penalty']:
                            if params['length_penalty']:
                                # this 5 is a magic number by Google...
                                length_penalties = [((5 + len(sample)) ** params['length_norm_factor'] /
                                                     (5 + 1) ** params['length_norm_factor']) for sample in samples]
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
                            scores = [co / lp + cov_p for co, lp, cov_p in
                                      zip(scores, length_penalties, coverage_penalties)]

                        elif params['normalize_probs']:
                            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
                            scores = [co / cn for co, cn in zip(scores, counts)]

                        best_score_idx = np.argmin(scores)
                        best_sample = samples[best_score_idx]
                        best_samples.append(best_sample)
                        best_scores = np.concatenate([best_scores, [scores[best_score_idx]]])
                        if params['pos_unk']:
                            best_alphas.append(np.asarray(alphas[best_score_idx]))

                        eta = (n_samples - sampled) * (time.time() - start_time) / sampled
                        if params['n_samples'] > 0:
                            for output_id in params['model_outputs']:
                                references.append(Y[output_id][i])

                        # store outputs for temporally-linked models
                        if params['temporally_linked']:
                            first_idx = max(0, data_gen_instance.first_idx)
                            # TODO: Make it more general
                            for (output_id, input_id) in iteritems(self.matchings_sample_to_next_sample):
                                # Get all words previous to the padding
                                previous_outputs[input_id][first_idx + sampled - 1] = best_sample[:sum(
                                    [int(elem > 0) for elem in best_sample])]
                sys.stdout.write('\n Total cost: %f \t'
                                 ' Average cost: %f\n' % (float(np.sum(best_scores, axis=None)),
                                                          float(np.average(best_scores, axis=None))))
                sys.stdout.write('The sampling took: %f secs '
                                 '(Speed: %f sec/sample)\n' % ((time.time() - start_time),
                                                               (time.time() - start_time) / n_samples))

                sys.stdout.flush()
                sources = None
                if params['pos_unk']:
                    if eval('ds.loaded_raw_' + s + '[0]'):
                        sources = file2list(eval('ds.X_raw_' + s + '["raw_' + params['model_inputs'][0] + '"]'),
                                            stripfile=False)
                predictions[s] = {
                    'samples': np.asarray(best_samples),
                    'alphas': best_alphas,
                    'sources': sources,
                    'costs': best_scores
                }
        del data_gen
        del data_gen_instance
        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling

    def predictNet(self,
                   ds,
                   parameters=None,
                   postprocess_fun=None):
        """
            Returns the predictions of the net on the dataset splits chosen. The input 'parameters' is a dict()
            which may contain the following parameters:

            Additional parameters:
            :param ds:
            :param parameters:
            :param postprocess_fun : post-processing function applied to all predictions before returning the result.
                                    The output of the function must be a list of results, one per sample.
                                    If postprocess_fun is a list, the second element will be used as an extra
                                     input to the function.
            :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """
        if parameters is None:
            parameters = dict()
        # Check input parameters and recover default values if needed
        params = checkParameters(parameters,
                                 self.default_predict_params)

        model_predict = getattr(self,
                                params['model_name'])  # recover model for prediction
        predictions = dict()
        for s in params['predict_on_sets']:
            predictions[s] = []
            if params['verbose'] > 0:
                print("",
                      file=sys.stderr)
                logger.info("<<< Predicting outputs of " + s + " set >>>")
                logger.info(print_dict(params,
                                       header="Prediction parameters: "))

            # Calculate how many iterations are we going to perform
            if params['n_samples'] is None:
                if params['init_sample'] > -1 and params['final_sample'] > -1:
                    n_samples = params['final_sample'] - params['init_sample']
                else:
                    n_samples = eval("ds.len_" + s)
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))
                n_samples = min(eval("ds.len_" + s),
                                num_iterations * params['batch_size'])

                # Prepare data generator
                if params['n_parallel_loaders'] > 1:
                    data_gen = Parallel_Data_Batch_Generator(s,
                                                             self,
                                                             ds,
                                                             num_iterations,
                                                             batch_size=params['batch_size'],
                                                             normalization=params['normalize'],
                                                             normalization_type=params['normalization_type'],
                                                             data_augmentation=False,
                                                             wo_da_patch_type=params['wo_da_patch_type'],
                                                             mean_substraction=params['mean_substraction'],
                                                             init_sample=params['init_sample'],
                                                             final_sample=params['final_sample'],
                                                             predict=True,
                                                             n_parallel_loaders=params[
                                                                 'n_parallel_loaders']).generator()
                else:
                    data_gen = Data_Batch_Generator(s,
                                                    self,
                                                    ds,
                                                    num_iterations,
                                                    batch_size=params['batch_size'],
                                                    normalization=params['normalize'],
                                                    normalization_type=params['normalization_type'],
                                                    data_augmentation=False,
                                                    wo_da_patch_type=params['wo_da_patch_type'],
                                                    mean_substraction=params['mean_substraction'],
                                                    init_sample=params['init_sample'],
                                                    final_sample=params['final_sample'],
                                                    predict=True).generator()

            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))
                # Prepare data generator
                if params['n_parallel_loaders'] > 1:
                    data_gen = Parallel_Data_Batch_Generator(s,
                                                             self,
                                                             ds,
                                                             num_iterations,
                                                             batch_size=params['batch_size'],
                                                             normalization=params['normalize'],
                                                             normalization_type=params['normalization_type'],
                                                             data_augmentation=False,
                                                             wo_da_patch_type=params['wo_da_patch_type'],
                                                             mean_substraction=params['mean_substraction'],
                                                             predict=True,
                                                             random_samples=n_samples,
                                                             n_parallel_loaders=params[
                                                                 'n_parallel_loaders']).generator()
                else:
                    data_gen = Data_Batch_Generator(s,
                                                    self,
                                                    ds,
                                                    num_iterations,
                                                    batch_size=params['batch_size'],
                                                    normalization=params['normalize'],
                                                    normalization_type=params['normalization_type'],
                                                    data_augmentation=False,
                                                    wo_da_patch_type=params['wo_da_patch_type'],
                                                    mean_substraction=params['mean_substraction'],
                                                    predict=True,
                                                    random_samples=n_samples).generator()
            # Predict on model
            if postprocess_fun is None:
                if int(keras.__version__.split('.')[0]) == 1:
                    # Keras version 1.x
                    out = model_predict.predict_generator(data_gen,
                                                          val_samples=n_samples,
                                                          max_q_size=params['n_parallel_loaders'],
                                                          nb_worker=1,  # params['n_parallel_loaders'],
                                                          pickle_safe=False)
                else:
                    # Keras version 2.x
                    out = model_predict.predict_generator(data_gen,
                                                          num_iterations,
                                                          max_queue_size=params['n_parallel_loaders'],
                                                          workers=1,  # params['n_parallel_loaders'],
                                                          verbose=params['verbose'])
                predictions[s] = out
            else:
                processed_samples = 0
                start_time = time.time()
                while processed_samples < n_samples:
                    out = model_predict.predict_on_batch(next(data_gen))

                    # Apply post-processing function
                    if isinstance(postprocess_fun, list):
                        last_processed = min(processed_samples + params['batch_size'],
                                             n_samples)
                        out = postprocess_fun[0](out,
                                                 postprocess_fun[1][processed_samples:last_processed])
                    else:
                        out = postprocess_fun(out)
                    predictions[s] += out

                    # Show progress
                    processed_samples += params['batch_size']
                    if processed_samples > n_samples:
                        processed_samples = n_samples

                    eta = (n_samples - processed_samples) * (time.time() - start_time) / processed_samples
                    sys.stdout.write("Predicting %d/%d  -  ETA: %ds " % (processed_samples,
                                                                         n_samples,
                                                                         int(eta)))
                    if not hasattr(self, '_dynamic_display') or self._dynamic_display:
                        sys.stdout.write('\r')
                    else:
                        sys.stdout.write('\n')
                    sys.stdout.flush()

        return predictions

    def predictOnBatch(self,
                       X,
                       in_name=None,
                       out_name=None,
                       expand=False):
        """
            Applies a forward pass and returns the predicted values.
        """
        # Get desired input
        if in_name:
            X = copy.copy(X[in_name])

        # Expand input dimensions to 4
        if expand:
            while len(X.shape) < 4:
                X = np.expand_dims(X, axis=1)

        X = self.prepareData(X, None)[0]

        # Apply forward pass for prediction
        predictions = self.model.predict_on_batch(X)

        # Select output if indicated
        if isinstance(self.model, Model):
            if out_name:
                predictions = predictions[out_name]
        elif isinstance(self.model, Sequential):
            predictions = predictions[0]

        return predictions

    # ------------------------------------------------------- #
    #       SCORING FUNCTIONS
    #           Functions for making scoring (x, y) samples
    # ------------------------------------------------------- #

    def score_cond_model(self,
                         X,
                         Y,
                         params,
                         null_sym=2):
        """
        Scoring for Cond models.
        :param X: Model inputs
        :param Y: Model outputs
        :param params: Search parameters
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        pad_on_batch = params['pad_on_batch']
        score = 0.0
        if params['words_so_far']:
            state_below = np.asarray([[null_sym]]) \
                if pad_on_batch else np.asarray([np.zeros((params['maxlen'], params['maxlen']))])
        else:
            state_below = np.asarray([null_sym]) \
                if pad_on_batch else np.asarray([np.zeros(params['maxlen'])])

        prev_out = None
        for ii in range(len(Y)):
            # for every possible live sample calc prob for every possible label
            if params['optimized_search']:  # use optimized search model if available
                [probs, prev_out, _] = self.predict_cond_optimized(X, state_below, params, ii, prev_out)
            else:
                probs = self.predict_cond(X, state_below, params, ii)
            # total score for every sample is sum of -log of word prb
            score -= np.log(probs[0, int(Y[ii])])
            state_below = np.asarray([Y[:ii]], dtype='int64')
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

            if params['optimized_search'] and ii > 0:
                # filter next search inputs w.r.t. remaining samples
                for idx_vars in range(len(prev_out)):
                    prev_out[idx_vars] = prev_out[idx_vars]

        return score

    def scoreNet(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the search process:
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
        params = checkParameters(self.params, self.default_predict_with_beam_params)
        scores_dict = dict()

        for s in params['predict_on_sets']:
            print("", file=sys.stderr)
            logger.info("<<< Scoring outputs of " + s + " set >>>")
            if len(params['model_inputs']) == 0:
                raise AssertionError('We need at least one input!')
            if not params['optimized_search']:  # use optimized search model if available
                if params['pos_unk']:
                    raise AssertionError('PosUnk is not supported with non-optimized beam search methods')

            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many iterations are we going to perform
            n_samples = eval("self.dataset.len_" + s)
            num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))

            # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
            # TODO: We prepare data as model 0... Different data preparators for each model?
            if params['n_parallel_loaders'] > 1:
                data_gen = Parallel_Data_Batch_Generator(s,
                                                         self.models[0],
                                                         self.dataset,
                                                         num_iterations,
                                                         shuffle=False,
                                                         batch_size=params['batch_size'],
                                                         normalization=params['normalize'],
                                                         normalization_type=params['normalization_type'],
                                                         data_augmentation=False,
                                                         wo_da_patch_type=params['wo_da_patch_type'],
                                                         mean_substraction=params['mean_substraction'],
                                                         predict=False,
                                                         n_parallel_loaders=params['n_parallel_loaders']).generator()
            else:
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                shuffle=False,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                normalization_type=params['normalization_type'],
                                                data_augmentation=False,
                                                wo_da_patch_type=params['wo_da_patch_type'],
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
                    sys.stdout.write('\r')
                    sys.stdout.write("Scored %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()

                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    y = self.models[0].one_hot_2_indices([Y[params['dataset_outputs'][params['output_text_index']]][i]],
                                                         pad_sequences=True,
                                                         verbose=0)[0]
                    score = self.score_cond_model(x,
                                                  y,
                                                  params,
                                                  null_sym=self.dataset.extra_words['<null>'])
                    if params['normalize']:
                        counts = float(len(y) ** params['alpha_factor'])
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

    # ------------------------------------------------------- #
    #       DECODING FUNCTIONS
    #           Functions for decoding predictions
    # ------------------------------------------------------- #

    @staticmethod
    def sampling(scores,
                 sampling_type='max_likelihood',
                 temperature=1.0):
        """
        Sampling words (each sample is drawn from a categorical distribution).
        Or picks up words that maximize the likelihood.
        :param scores: array of size #samples x #classes;
        every entry determines a score for sample i having class j
        :param sampling_type:
        :param temperature: Temperature for the predictions. The higher, the flatter probabilities.
                            Hence more random outputs.
        :return: set of indices chosen as output, a vector of size #samples
        """
        logger.warning("Deprecated function, use utils.sampling() instead")
        return sampling(scores,
                        sampling_type=sampling_type,
                        temperature=temperature)

    @staticmethod
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
        logger.warning("Deprecated function, use utils.decode_predictions() instead.")
        return decode_predictions(preds,
                                  temperature,
                                  index2word,
                                  sampling_type,
                                  verbose=verbose)

    @staticmethod
    def decode_predictions_beam_search(preds,
                                       index2word,
                                       alphas=None,
                                       heuristic=0,
                                       x_text=None,
                                       unk_symbol='<unk>',
                                       pad_sequences=False,
                                       mapping=None,
                                       verbose=0):
        """
        Decodes predictions from the BeamSearch method.
        :param alphas:
        :param heuristic:
        :param x_text:
        :param unk_symbol:
        :param mapping:
        :param preds: Predictions codified as word indices.
        :param index2word: Mapping from word indices into word characters.
        :param pad_sequences: Whether we should make a zero-pad on the input sequence.
        :param verbose: Verbosity level, by default 0.
        :return: List of decoded predictions
        """
        logger.warning("Deprecated function, use utils.decode_predictions_beam_search() instead.")
        return decode_predictions_beam_search(preds,
                                              index2word,
                                              alphas=alphas,
                                              heuristic=heuristic,
                                              x_text=x_text,
                                              unk_symbol=unk_symbol,
                                              pad_sequences=pad_sequences,
                                              mapping=mapping,
                                              verbose=verbose)

    @staticmethod
    def one_hot_2_indices(preds,
                          pad_sequences=True,
                          verbose=0):
        """
        Converts a one-hot codification into a index-based one
        :param pad_sequences:
        :param preds: Predictions codified as one-hot vectors.
        :param verbose: Verbosity level, by default 0.
        :return: List of converted predictions
        """
        logger.warning("Deprecated function, use utils.one_hot_2_indices() instead.")
        return one_hot_2_indices(preds,
                                 pad_sequences=pad_sequences,
                                 verbose=verbose)

    @staticmethod
    def decode_predictions_one_hot(preds,
                                   index2word,
                                   verbose=0):
        """
        Decodes predictions following a one-hot codification.
        :param preds: Predictions codified as one-hot vectors.
        :param index2word: Mapping from word indices into word characters.
        :param verbose: Verbosity level, by default 0.
        :return: List of decoded predictions
        """
        logger.warning("Deprecated function, use utils.decode_predictions_one_hot() instead.")
        return decode_predictions_one_hot(preds,
                                          index2word,
                                          verbose=verbose)

    def prepareData(self,
                    X_batch,
                    Y_batch=None):
        """
        Prepares the data for the model, depending on its type (Sequential, Model).
        :param X_batch: Batch of input data.
        :param Y_batch: Batch output data.
        :return: Prepared data.
        """
        if isinstance(self.model, Sequential):
            data = self._prepareSequentialData(X_batch, Y_batch)
        elif isinstance(self.model, Model):
            data = self._prepareModelData(X_batch, Y_batch)
        else:
            raise NotImplementedError
        return data

    def _prepareSequentialData(self,
                               X,
                               Y=None,
                               sample_weights=False):

        # Format input data
        if len(list(self.inputsMapping)) == 1:  # single input
            X = X[self.inputsMapping[0]]
        else:
            X_new = [0 for _ in range(len(list(self.inputsMapping)))]  # multiple inputs
            for in_model, in_ds in iteritems(self.inputsMapping):
                X_new[in_model] = X[in_ds]
            X = X_new

        # Format output data (only one output possible for Sequential models)
        Y_sample_weights = None
        if Y is not None:
            if len(list(self.outputsMapping)) == 1:  # single output
                if isinstance(Y[self.outputsMapping[0]], tuple):
                    Y = Y[self.outputsMapping[0]][0]
                    Y_sample_weights = Y[self.outputsMapping[0]][1]
                else:
                    Y = Y[self.outputsMapping[0]]
            else:
                Y_new = [0 for _ in range(len(list(self.outputsMapping)))]  # multiple outputs
                Y_sample_weights = [None for _ in range(len(list(self.outputsMapping)))]
                for out_model, out_ds in iteritems(self.outputsMapping):
                    if isinstance(Y[out_ds], tuple):
                        Y_new[out_model] = Y[out_ds][0]
                        Y_sample_weights[out_model] = Y[out_ds][1]
                    else:
                        Y_new[out_model] = Y[out_ds]
                Y = Y_new

        return [X, Y] if Y_sample_weights is None else [X, Y, Y_sample_weights]

    def _prepareModelData(self, X, Y=None):
        X_new = dict()
        Y_new = dict()
        Y_sample_weights = dict()
        # Format input data
        for in_model, in_ds in iteritems(self.inputsMapping):
            X_new[in_model] = X[in_ds]

        # Format output data
        if Y is not None:
            for out_model, out_ds in iteritems(self.outputsMapping):
                if isinstance(Y[out_ds], tuple):
                    Y_new[out_model] = Y[out_ds][0]
                    Y_sample_weights[out_model] = Y[out_ds][1]
                else:
                    Y_new[out_model] = Y[out_ds]

        return [X_new, Y_new] if Y_sample_weights == dict() else [X_new, Y_new, Y_sample_weights]

    @staticmethod
    def _getGraphAccuracy(data,
                          prediction,
                          topN=5):
        """
            Calculates the accuracy obtained from a set of samples on a Graph model.
        """

        accuracies = dict()
        top_accuracies = dict()
        for key, val in iteritems(prediction):
            pred = categorical_probas_to_classes(val)
            top_pred = np.argsort(val, axis=1)[:, ::-1][:, :np.min([topN, val.shape[1]])]
            GT = categorical_probas_to_classes(data[key])

            # Top1 accuracy
            correct = [1 if pred[i] == GT[i] else 0 for i in range(len(pred))]
            accuracies[key] = float(np.sum(correct)) / float(len(correct))

            # TopN accuracy
            top_correct = [1 if GT[i] in top_pred[i, :] else 0 for i in range(top_pred.shape[0])]
            top_accuracies[key] = float(np.sum(top_correct)) / float(len(top_correct))

        return [accuracies, top_accuracies]

    @staticmethod
    def _getSequentialAccuracy(GT,
                               pred,
                               topN=5):
        """
            Calculates the topN accuracy obtained from a set of samples on a Sequential model.
        """
        top_pred = np.argsort(pred, axis=1)[:, ::-1][:, :np.min([topN, pred.shape[1]])]
        pred = categorical_probas_to_classes(pred)
        GT = categorical_probas_to_classes(GT)

        # Top1 accuracy
        correct = [1 if pred[i] == GT[i] else 0 for i in range(len(pred))]
        accuracies = float(np.sum(correct)) / float(len(correct))

        # TopN accuracy
        top_correct = [1 if GT[i] in top_pred[i, :] else 0 for i in range(top_pred.shape[0])]
        top_accuracies = float(np.sum(top_correct)) / float(len(top_correct))

        return [accuracies, top_accuracies]

    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for train logging and visualization
    # ------------------------------------------------------- #

    def log_tensorboard(self,
                        metrics,
                        step,
                        split=None):
        """Logs scalar metrics in Tensorboard
        """
        if self.tensorboard_callback:
            if split:
                metrics = {str(split) + '/' + k: v for k, v in iteritems(metrics)}
            self.tensorboard_callback._write_logs(metrics, step)

    def log(self,
            mode,
            data_type,
            value):
        """
        Stores the train and val information for plotting the training progress.

        :param mode: 'train', 'val' or 'test'
        :param data_type: 'iteration', 'loss', 'accuracy', etc.
        :param value: numerical value taken by the data_type
        """
        if mode not in self.__modes:
            raise Exception('The provided mode "' + mode + '" is not valid.')
        if mode not in self.__logger:
            self.__logger[mode] = dict()
        if data_type not in self.__logger[mode]:
            self.__logger[mode][data_type] = list()
        self.__logger[mode][data_type].append(value)

    def getLog(self,
               mode,
               data_type):
        """
        Returns the all logged values for a given mode and a given data_type

        :param mode: 'train', 'val' or 'test'
        :param data_type: 'iteration', 'loss', 'accuracy', etc.
        :return: list of values logged
        """
        if mode not in self.__logger:
            return [None]
        elif data_type not in self.__logger[mode]:
            return [None]
        else:
            return self.__logger[mode][data_type]

    def plot(self,
             time_measure,
             metrics,
             splits,
             upperbound=None,
             colours_shapes_dict=None,
             epsilon=1e-3):
        """
        Plots the training progress information

        Example of input:
        model.plot('epoch', ['accuracy'], ['val', 'test'],
                   upperbound=1, colours_dict={'accuracy_val', 'b', 'accuracy_test', 'g'})

        :param time_measure: either 'epoch' or 'iteration'
        :param metrics: list of metrics that we want to plot
        :param splits: list of data splits that we want to plot
        :param upperbound: upper bound of the metrics about to plot (usually upperbound=1.0)
        :param colours_shapes_dict: dictionary of '<metric>_<split>' and the colour and/or shape
                that we want them to have in the plot
        """
        import matplotlib as mpl
        mpl.use('Agg')  # run matplotlib without X server (GUI)
        import matplotlib.pyplot as plt

        # Build default colours_shapes_dict if not provided
        colours_shapes_dict = colours_shapes_dict or dict()

        if not colours_shapes_dict:
            default_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            default_shapes = ['-', 'o', '.']
            m = 0
            for met in metrics:
                s = 0
                for sp in splits:
                    colours_shapes_dict[met + '_' + sp] = default_colours[m] + default_shapes[s]
                    s += 1
                    s %= len(default_shapes)
                m += 1
                m %= len(default_colours)

        plt.figure(1).add_axes([0.1, 0.1, 0.6, 0.75])

        all_iterations = []
        for sp in splits:
            if sp not in self.__logger:
                raise Exception("There is no performance data from split '" + sp + "' in the model log.")
            if time_measure not in self.__logger[sp]:
                raise Exception(
                    "There is no performance data on each '" + time_measure +
                    "' in the model log for split '" + sp + "'.")

            iterations = self.__logger[sp][time_measure]
            all_iterations = all_iterations + iterations

            for met in metrics:
                if met not in self.__logger[sp]:
                    raise Exception(
                        "There is no performance data for metric '" + met +
                        "' in the model log for split '" + sp + "'.")

                measure = self.__logger[sp][met]
                if upperbound and max(measure) > upperbound:
                    logging.warning('The value of metric %s is higher than the maximum value ploted (%.3f > %.3f)' %
                                    (str(met), float(max(measure)), float(upperbound)))
                plt.plot(iterations,
                         measure,
                         colours_shapes_dict[met + '_' + sp],
                         label=str(met),
                         marker='x')

        max_iter = np.max(all_iterations + [0])
        # Plot upperbound
        if upperbound is not None:
            plt.plot([0, max_iter + epsilon], [upperbound, upperbound], 'r-')
            plt.axis([0, max_iter + epsilon, 0, upperbound])  # limit height to upperbound

        # Fill labels
        plt.xlabel(time_measure)
        # plt.subplot(211)
        plt.title('Training progress')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Create plots dir
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Save figure
        plot_file = os.path.join(self.model_path, time_measure + '_' + str(max_iter) + '.jpg')
        plt.savefig(plot_file)
        if not self.silence:
            logger.info("\n<<< Progress plot saved in " + plot_file + ' >>>')

        # Close plot window
        plt.close()


# Backwards compatibility
CNN_Model = Model_Wrapper
