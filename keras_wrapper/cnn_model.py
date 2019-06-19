# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import shutil
import sys
import time
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

if sys.version_info.major == 3:
    import _pickle as pk
else:
    import cPickle as pk
import cloudpickle as cloudpk
import keras
from keras.engine.training import Model
from keras.layers import concatenate, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Dense, Dropout, Flatten, Input, \
    Activation, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.regularizers import l2
from keras.utils.layer_utils import print_summary
from keras_wrapper.dataset import Data_Batch_Generator, Homogeneous_Data_Batch_Generator, Parallel_Data_Batch_Generator
from keras_wrapper.extra.callbacks import *
from keras_wrapper.extra.read_write import file2list
from keras_wrapper.utils import one_hot_2_indices, decode_predictions, decode_predictions_one_hot, \
    decode_predictions_beam_search, replace_unknown_words, sampling, categorical_probas_to_classes, checkParameters, print_dict
from keras_wrapper.search import beam_search

# General setup of libraries
try:
    import cupy as cp
except:
    import numpy as cp
    logger.info('<<< Cupy not available. Using numpy. >>>')

if int(keras.__version__.split('.')[0]) == 1:
    from keras.layers import Concat as Concatenate
    from keras.layers import Convolution2D as Conv2D
    from keras.layers import Deconvolution2D as Conv2DTranspose
else:
    from keras.layers import Concatenate
    from keras.layers import Conv2D
    from keras.layers import Conv2DTranspose


# ------------------------------------------------------- #
#       SAVE/LOAD
#           External functions for saving and loading Model_Wrapper instances
# ------------------------------------------------------- #

def saveModel(model_wrapper, update_num, path=None, full_path=False, store_iter=False):
    """
    Saves a backup of the current Model_Wrapper object after being trained for 'update_num' iterations/updates/epochs.

    :param model_wrapper: object to save
    :param update_num: identifier of the number of iterations/updates/epochs elapsed
    :param path: path where the model will be saved
    :param full_path: Whether we save to the path of from path + '/epoch_' + update_num
    :param store_iter: Whether we store the current update_num
    :return: None
    """
    if not path:
        path = model_wrapper.model_path

    iteration = str(update_num)

    if full_path:
        if store_iter:
            model_name = path + '_' + iteration
        else:
            model_name = path
    else:
        if store_iter:
            model_name = path + '/update_' + iteration
        else:
            model_name = path + '/epoch_' + iteration

    if not model_wrapper.silence:
        logger.info("<<< Saving model to " + model_name + " ... >>>")

    # Create models dir
    if not os.path.isdir(path):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    try:  # Try to save model at one time
        model_wrapper.model.save(model_name + '.h5')
    except Exception as e:  # Split saving in model structure / weights
        logger.info(str(e))
        # Save model structure
        json_string = model_wrapper.model.to_json()
        open(model_name + '_structure.json', 'w').write(json_string)
        # Save model weights
        model_wrapper.model.save_weights(model_name + '_weights.h5', overwrite=True)

    # Save auxiliary models for optimized search
    if model_wrapper.model_init is not None:
        try:  # Try to save model at one time
            model_wrapper.model_init.save(model_name + '_init.h5')
        except Exception as e:  # Split saving in model structure / weights
            logger.info(str(e))
            # Save model structure
            logger.info("<<< Saving model_init to " + model_name + "_structure_init.json... >>>")
            json_string = model_wrapper.model_init.to_json()
            open(model_name + '_structure_init.json', 'w').write(json_string)
            # Save model weights
            model_wrapper.model_init.save_weights(model_name + '_weights_init.h5', overwrite=True)

    if model_wrapper.model_next is not None:
        try:  # Try to save model at one time
            model_wrapper.model_next.save(model_name + '_next.h5')
        except Exception as e:  # Split saving in model structure / weights
            logger.info(str(e))
            # Save model structure
            logger.info("<<< Saving model_next to " + model_name + "_structure_next.json... >>>")
            json_string = model_wrapper.model_next.to_json()
            open(model_name + '_structure_next.json', 'w').write(json_string)
            # Save model weights
            model_wrapper.model_next.save_weights(model_name + '_weights_next.h5', overwrite=True)

    # Save additional information
    backup_multi_gpu_model = None
    if hasattr(model_wrapper, 'multi_gpu_model'):
        backup_multi_gpu_model = model_wrapper.multi_gpu_model
        setattr(model_wrapper, 'multi_gpu_model', None)

    cloudpk.dump(model_wrapper, open(model_name + '_Model_Wrapper.pkl', 'wb'))
    setattr(model_wrapper, 'multi_gpu_model', backup_multi_gpu_model)

    if not model_wrapper.silence:
        logger.info("<<< Model saved >>>")


def loadModel(model_path, update_num, reload_epoch=True, custom_objects=None, full_path=False, compile_model=False):
    """
    Loads a previously saved Model_Wrapper object.

    :param model_path: path to the Model_Wrapper object to load
    :param update_num: identifier of the number of iterations/updates/epochs elapsed
    :param reload_epoch: Whether we should load epochs or updates
    :param custom_objects: dictionary of custom layers (i.e. input to model_from_json)
    :param full_path: Whether we should load the path from model_name or from model_path directly.
    :return: loaded Model_Wrapper
    """
    if not custom_objects:
        custom_objects = dict()

    t = time.time()
    iteration = str(update_num)

    if full_path:
        model_name = model_path
    else:
        if reload_epoch:
            model_name = model_path + "/epoch_" + iteration
        else:
            model_name = model_path + "/update_" + iteration

    logger.info("<<< Loading model from " + model_name + "_Model_Wrapper.pkl ... >>>")
    try:
        logger.info("<<< Loading model from " + model_name + ".h5 ... >>>")
        model = load_model(model_name + '.h5', custom_objects=custom_objects, compile=compile_model)
    except Exception as e:
        logger.info(str(e))
        # Load model structure
        logger.info("<<< Loading model from " + model_name + "_structure.json' ... >>>")
        model = model_from_json(open(model_name + '_structure.json').read(), custom_objects=custom_objects)
        # Load model weights
        model.load_weights(model_name + '_weights.h5')

    # Load auxiliary models for optimized search
    if os.path.exists(model_name + '_init.h5') and os.path.exists(model_name + '_next.h5'):
        loading_optimized = 1
    elif os.path.exists(model_name + '_structure_init.json') and os.path.exists(
            model_name + '_weights_init.h5') and os.path.exists(model_name + '_structure_next.json') and os.path.exists(
            model_name + '_weights_next.h5'):
        loading_optimized = 2
    else:
        loading_optimized = 0

    if loading_optimized == 1:
        logger.info("<<< Loading optimized model... >>>")
        model_init = load_model(model_name + '_init.h5', custom_objects=custom_objects, compile=False)
        model_next = load_model(model_name + '_next.h5', custom_objects=custom_objects, compile=False)

    elif loading_optimized == 2:
        # Load model structure
        logger.info("\t <<< Loading model_init from " + model_name + "_structure_init.json ... >>>")
        model_init = model_from_json(open(model_name + '_structure_init.json').read(),
                                     custom_objects=custom_objects)
        # Load model weights
        model_init.load_weights(model_name + '_weights_init.h5')
        # Load model structure
        logger.info("\t <<< Loading model_next from " + model_name + "_structure_next.json ... >>>")
        model_next = model_from_json(open(model_name + '_structure_next.json').read(), custom_objects=custom_objects)
        # Load model weights
        model_next.load_weights(model_name + '_weights_next.h5')

    # Load Model_Wrapper information
    try:
        if sys.version_info.major == 3:
            model_wrapper = pk.load(open(model_name + '_Model_Wrapper.pkl', 'rb'), encoding='latin1')
        else:
            model_wrapper = pk.load(open(model_name + '_Model_Wrapper.pkl', 'rb'))
    except Exception as e:
        # try:
        logger.info(str(e))
        if sys.version_info.major == 3:
            model_wrapper = pk.load(open(model_name + '_CNN_Model.pkl', 'rb'), encoding='latin1')
        else:
            model_wrapper = pk.load(open(model_name + '_CNN_Model.pkl', 'rb'))
        # except:
        #    raise Exception(ValueError)

    # Add logger for backwards compatibility (old pre-trained models) if it does not exist
    model_wrapper.updateLogger()

    model_wrapper.model = model
    if loading_optimized != 0:
        model_wrapper.model_init = model_init
        model_wrapper.model_next = model_next
        logger.info("<<< Optimized model loaded. >>>")
    else:
        model_wrapper.model_init = None
        model_wrapper.model_next = None
    logger.info("<<< Model loaded in %0.6s seconds. >>>" % str(time.time() - t))
    return model_wrapper


def updateModel(model, model_path, update_num, reload_epoch=True, full_path=False, compile_model=False):
    """
    Loads a the weights from files to a Model_Wrapper object.

    :param model: Model_Wrapper object to update
    :param model_path: path to the weights to load
    :param update_num: identifier of the number of iterations/updates/epochs elapsed
    :param reload_epoch: Whether we should load epochs or updates
    :param full_path: Whether we should load the path from model_name or from model_path directly.
    :return: updated Model_Wrapper
    """
    t = time.time()
    model_name = model.name
    iteration = str(update_num)

    if not full_path:
        if reload_epoch:
            model_path = model_path + "/epoch_" + iteration
        else:
            model_path = model_path + "/update_" + iteration

    logger.info("<<< Updating model " + model_name + " from " + model_path + " ... >>>")

    try:
        logger.info("<<< Updating model from " + model_path + ".h5 ... >>>")
        model.model.set_weights(load_model(model_path + '.h5', compile=False).get_weights())

    except Exception as e:
        logger.info(str(e))
        # Load model structure
        logger.info("<<< Failed -> Loading model from " + model_path + "_weights.h5' ... >>>")
        # Load model weights
        model.model.load_weights(model_path + '_weights.h5')

    # Load auxiliary models for optimized search
    if os.path.exists(model_name + '_init.h5') and os.path.exists(model_name + '_next.h5'):
        loading_optimized = 1
    elif os.path.exists(model_name + '_weights_init.h5') and os.path.exists(model_name + '_weights_next.h5'):
        loading_optimized = 2
    else:
        loading_optimized = 0

    if loading_optimized != 0:
        logger.info("<<< Updating optimized model... >>>")
        if loading_optimized == 1:
            logger.info("\t <<< Updating model_init from " + model_path + "_init.h5 ... >>>")
            model.model_init.set_weights(load_model(model_path + '_init.h5', compile=False).get_weights())
            logger.info("\t <<< Updating model_next from " + model_path + "_next.h5 ... >>>")
            model.model_next.set_weights(load_model(model_path + '_next.h5', compile=False).get_weights())
        elif loading_optimized == 2:
            logger.info("\t <<< Updating model_init from " + model_path + "_structure_init.json ... >>>")
            model.model_init.load_weights(model_path + '_weights_init.h5')
            # Load model structure
            logger.info("\t <<< Updating model_next from " + model_path + "_structure_next.json ... >>>")
            # Load model weights
            model.model_next.load_weights(model_path + '_weights_next.h5')

    logger.info("<<< Model updated in %0.6s seconds. >>>" % str(time.time() - t))
    return model


def transferWeights(old_model, new_model, layers_mapping):
    """
    Transfers all existent layers' weights from an old model to a new model.

    :param old_model: old version of the model, where the weights will be picked
    :param new_model: new version of the model, where the weights will be transferred to
    :param layers_mapping: mapping from old to new model layers
    :return: new model with weights transferred
    """

    logger.info("<<< Transferring weights from models. >>>")

    old_layer_dict = dict([(layer.name, [layer, idx]) for idx, layer in list(enumerate(old_model.model.layers))])
    new_layer_dict = dict([(layer.name, [layer, idx]) for idx, layer in list(enumerate(new_model.model.layers))])

    for lold, lnew in iteritems(layers_mapping):
        # Check if layers exist in both models
        if lold in old_layer_dict and lnew in new_layer_dict:

            # Create dictionary name --> layer
            old = old_layer_dict[lold][0].get_weights()
            new = new_layer_dict[lnew][0].get_weights()

            # Find weight sizes matchings for each layer (without repetitions)
            new_shapes = [w.shape for w in new]
            mapping_weights = dict()
            for pos_old, wo in list(enumerate(old)):
                old_shape = wo.shape
                indices = [i for i, shp in enumerate(new_shapes) if shp == old_shape]
                if indices:
                    for ind in indices:
                        if ind not in list(mapping_weights):
                            mapping_weights[ind] = pos_old
                            break

            # Alert for any weight matrix not inserted to new model
            for pos_old, wo in list(enumerate(old)):
                if pos_old not in list(mapping_weights.values()):
                    logger.info('  Pre-trained weight matrix of layer "' + lold +
                                '" with dimensions ' + str(wo.shape) + ' can not be inserted to new model.')

            # Alert for any weight matrix not modified
            for pos_new, wn in list(enumerate(new)):
                if pos_new not in list(mapping_weights):
                    logger.info('  New model weight matrix of layer "' + lnew +
                                '" with dimensions ' + str(wn.shape) + ' can not be loaded from pre-trained model.')

            # Transfer weights for each layer
            for new_idx, old_idx in iteritems(mapping_weights):
                new[new_idx] = old[old_idx]
            new_model.model.layers[new_layer_dict[lnew][1]].set_weights(new)

        else:
            logger.info('Can not apply weights transfer from "' + lold + '" to "' + lnew + '"')

    logger.info("<<< Weights transferred successfully. >>>")

    return new_model


def read_layer_names(model, starting_name=None):
    """
        Reads the existent layers' names from a model starting after a layer specified by its name

        :param model: model whose layers' names will be read
        :param starting_name: name of the layer after which the layers' names will be read
                              (if None, then all the layers' names will be read)
        :return: list of layers' names
        """

    if starting_name is None:
        read = True
    else:
        read = False

    layers_names = []
    for layer in model.layers:
        if read:
            layers_names.append(layer.name)
        elif layer.name == starting_name:
            read = True

    return layers_names


# ------------------------------------------------------- #
#       MAIN CLASS
# ------------------------------------------------------- #
class Model_Wrapper(object):
    """
        Wrapper for Keras' models. It provides the following utilities:
            - Training visualization module.
            - Set of already implemented CNNs for quick definition.
            - Easy layers re-definition for finetuning.
            - Model backups.
            - Easy to use training and test methods.
    """

    def __init__(self, nOutput=1000, model_type='basic_model', silence=False, input_shape=None,
                 structure_path=None, weights_path=None, seq_to_functional=False,
                 model_name=None, plots_path=None, models_path=None, inheritance=False):
        """
            Model_Wrapper object constructor.

            :param nOutput: number of outputs of the network. Only valid if 'structure_path' is None.
            :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                         Only valid if 'structure_path' is None.
            :param silence: set to True if you don't want the model to output informative messages
            :param input_shape: array with 3 integers which define the images' input shape [height, width, channels].
                                Only valid if 'structure_path' is None.
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
        if input_shape is None:
            input_shape = [256, 256, 3]

        self.__toprint = ['net_type', 'name', 'plot_path', 'models_path', 'lr', 'momentum',
                          'training_parameters', 'testing_parameters', 'training_state', 'loss', 'silence']

        self.silence = silence
        self.net_type = model_type
        self.lr = 0.01  # insert default learning rate
        self.momentum = 1.0 - self.lr  # insert default momentum
        self.loss = None  # default loss function
        self.training_parameters = []
        self.testing_parameters = []
        self.training_state = dict()
        self._dynamic_display = True

        # Dictionary for storing any additional data needed
        self.additional_data = dict()

        # Model containers
        self.model = None
        self.model_init = None
        self.model_next = None
        # self.model_to_train = None

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

        self.set_default_params()

        # Prepare model
        if not inheritance:
            # Set Network name
            self.setName(model_name, plots_path, models_path)

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
                self.model.load_weights(weights_path, seq_to_functional=seq_to_functional)

    def updateLogger(self, force=False):
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

        self.__modes = ['train', 'val', 'test']

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
                                        'da_enhance_list': [],  # da_enhance_list = {brightness, color, sharpness, contrast}
                                        'verbose': 1,
                                        'eval_on_sets': ['val'],
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
                                        'tensorboard_params': {'log_dir': 'tensorboard_logs',
                                                               'histogram_freq': 0,
                                                               'batch_size': 50,
                                                               'write_graph': True,
                                                               'write_grads': False,
                                                               'write_images': False,
                                                               'embeddings_freq': 0,
                                                               'embeddings_layer_names': None,
                                                               'embeddings_metadata': None
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
                                       'model_name': 'model',  # name of the attribute where the model for prediction is stored
                                       }

    def setInputsMapping(self, inputsMapping):
        """
            Sets the mapping of the inputs from the format given by the dataset to the format received by the model.

            :param inputsMapping: dictionary with the model inputs' identifiers as keys and the dataset inputs
                                  identifiers' position as values.
                                  If the current model is Sequential then keys must be ints with the desired input order
                                  (starting from 0). If it is Model then keys must be str.
        """
        self.inputsMapping = inputsMapping

    def setOutputsMapping(self, outputsMapping, acc_output=None):
        """
            Sets the mapping of the outputs from the format given by the dataset to the format received by the model.

            :param outputsMapping: dictionary with the model outputs'
                                   identifiers as keys and the dataset outputs identifiers' position as values.
                                   If the current model is Sequential then keys must be ints with
                                   the desired output order (in this case only one value can be provided).
                                   If it is Model then keys must be str.
            :param acc_output: name of the model's output that will be used for calculating
                              the accuracy of the model (only needed for Graph models)
        """
        if isinstance(self.model, Sequential) and len(list(outputsMapping)) > 1:
            raise Exception("When using Sequential models only one output can be provided in outputsMapping")
        self.outputsMapping = outputsMapping
        self.acc_output = acc_output

    def setOptimizer(self, lr=None, momentum=None, loss='categorical_crossentropy', loss_weights=None, metrics=None,
                     epsilon=1e-8,
                     nesterov=True, decay=0.0, clipnorm=10., clipvalue=0., optimizer=None, sample_weight_mode=None,
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
                    optimizer = TFOptimizer(tf.train.MomentumOptimizer(lr, self.momentum, use_nesterov=nesterov))
            elif optimizer.lower() == 'adam':
                optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon))
            elif optimizer.lower() == 'adagrad':
                optimizer = TFOptimizer(tf.train.AdagradOptimizer(lr))
            elif optimizer.lower() == 'rmsprop':
                optimizer = TFOptimizer(tf.train.RMSPropOptimizer(lr, decay=decay, momentum=momentum, epsilon=epsilon))
            elif optimizer.lower() == 'nadam':
                logger.warning('The Nadam optimizer is not natively implemented in Tensorflow. Using Keras optimizer.')
                optimizer = Nadam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'adamax':
                logger.warning('The Adamax optimizer is not natively implemented in Tensorflow. Using Keras optimizer.')
                optimizer = Adamax(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'adadelta':
                optimizer = TFOptimizer(tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=epsilon))
            else:
                raise Exception('\tThe chosen optimizer is not implemented.')
        else:
            if optimizer is None or optimizer.lower() == 'sgd':
                optimizer = SGD(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, momentum=momentum,
                                nesterov=nesterov)
            elif optimizer.lower() == 'adam':
                optimizer = Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'adagrad':
                optimizer = Adagrad(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'rmsprop':
                optimizer = RMSprop(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'nadam':
                optimizer = Nadam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'adamax':
                optimizer = Adamax(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            elif optimizer.lower() == 'adadelta':
                optimizer = Adadelta(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue, decay=decay, epsilon=epsilon)
            else:
                raise Exception('\tThe chosen optimizer is not implemented.')

        if not self.silence:
            logger.info("Compiling model...")

        # compile differently depending if our model is 'Sequential', 'Model' or 'Graph'
        if isinstance(self.model, Sequential) or isinstance(self.model, Model):
            self.model.compile(optimizer=optimizer, metrics=metrics, loss=loss, loss_weights=loss_weights,
                               sample_weight_mode=sample_weight_mode)
        else:
            raise NotImplementedError()

        if not self.silence:
            logger.info("Optimizer updated, learning rate set to " + str(lr))

    def compile(self, **kwargs):
        """
        Compile the model.
        :param kwargs:
        :return:
        """
        self.model.compile(kwargs)

    def setName(self, model_name, plots_path=None, models_path=None, create_plots=False, clear_dirs=True):
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

    def setParams(self, params):
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

    def replaceLastLayers(self, num_remove, new_layers):
        """
            Replaces the last 'num_remove' layers in the model by the newly defined in 'new_layers'.
            Function only valid for Sequential models. Use self.removeLayers(...) for Graph models.
        """
        if not self.silence:
            logger.info("Replacing layers...")

        removed_layers = []
        removed_params = []
        # If it is a Sequential model
        if isinstance(self.model, Sequential):
            # Remove old layers
            for _ in range(num_remove):
                removed_layers.append(self.model.layers.pop())
                removed_params.append(self.model.params.pop())

            # Insert new layers
            for layer in new_layers:
                self.model.add(layer)

        # If it is a Graph model
        else:
            raise NotImplementedError("Try using self.removeLayers(...) instead.")

        return [removed_layers, removed_params]

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

    def trainNet(self, ds, parameters=None, out_name=None):
        """
            Trains the network on the given dataset.
            :param ds: Dataset with the training data
            :param parameters: dict() which may contain the following (optional) training parameters
            :param out_name: name of the output node that will be used to evaluate the network accuracy.
                            Only applicable to Graph models.

            The input 'parameters' is a dict() which may contain the following (optional) training parameters:
            ####    Visualization parameters
             * report_iter: number of iterations between each loss report
             * iter_for_val: number of iterations between each validation test
             * num_iterations_val: number of iterations applied on the validation dataset for computing the
                                   average performance (if None then all the validation data will be tested)
            ####    Learning parameters
             * n_epochs: number of epochs that will be applied during training
             * batch_size: size of the batch (number of images) applied on each iteration by the SGD optimization
             * lr_decay: number of iterations passed for decreasing the learning rate
             * lr_gamma: proportion of learning rate kept at each decrease.
                         It can also be a set of rules defined by a list, e.g.
                         lr_gamma = [[3000, 0.9], ..., [None, 0.8]] means 0.9 until iteration
                         3000, ..., 0.8 until the end.
             * patience: number of epochs waiting for a possible performance increase before stopping training
             * metric_check: name of the metric checked for early stopping and LR decrease

            ####    Data processing parameters

             * n_parallel_loaders: number of parallel data loaders allowed to work at the same time
             * normalize: boolean indicating if we want to normalize the image pixel values
             * mean_substraction: boolean indicating if we want to substract the training mean
             * data_augmentation: boolean indicating if we want to perform data augmentation
                                  (always False on validation)
             * shuffle: apply shuffling on training data at the beginning of each epoch.

            ####    Other parameters

        """

        # Check input parameters and recover default values if needed
        if parameters is None:
            parameters = dict()
        params = checkParameters(parameters, self.default_training_params, hard_check=True)
        # Set params['start_reduction_on_epoch'] = params['lr_decay'] by default
        if params['lr_decay'] is not None and 'start_reduction_on_epoch' not in list(parameters):
            params['start_reduction_on_epoch'] = params['lr_decay']
        save_params = copy.copy(params)
        del save_params['extra_callbacks']
        self.training_parameters.append(save_params)
        if params['verbose'] > 0:
            logger.info("<<< Training model >>>")

        self.__train(ds, params)

        logger.info("<<< Finished training model >>>")

    def trainNetFromSamples(self, x, y, parameters=None, class_weight=None, sample_weight=None, out_name=None):
        """
            Trains the network on the given samples x, y.

            :param x:
            :param y:
            :param parameters:
            :param class_weight:
            :param sample_weight:
            :param out_name: name of the output node that will be used to evaluate the network accuracy.
                             Only applicable to Graph models.

            The input 'parameters' is a dict() which may contain the following (optional) training parameters:
            ####    Visualization parameters
            ####    Learning parameters
            ####    Data processing parameters
            ####    Other parameters

        """

        # Check input parameters and recover default values if needed
        if parameters is None:
            parameters = dict()
        params = checkParameters(parameters, self.default_training_params, hard_check=True)
        save_params = copy.copy(params)
        del save_params['extra_callbacks']
        self.training_parameters.append(save_params)
        self.__train_from_samples(x, y, params, class_weight=class_weight, sample_weight=sample_weight)
        if params['verbose'] > 0:
            logger.info("<<< Finished training model >>>")

    def __train(self, ds, params, state=None):

        if state is None:
            state = dict()
        if params['verbose'] > 0:
            logger.info(print_dict(params, header="Training parameters: "))

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
            callback_store_model = StoreModelWeightsOnEpochEnd(self, saveModel, params['epochs_for_save'])
            callbacks.insert(0, callback_store_model)

        # Tensorboard callback
        if params['tensorboard'] and K.backend() == 'tensorflow':
            # embeddings_metadata = params['tensorboard_params']['embeddings_metadata']
            create_dir_if_not_exists(self.model_path + '/' + params['tensorboard_params']['log_dir'])
            # if params['tensorboard_params']['label_word_embeddings_with_vocab'] \
            #         and params['tensorboard_params']['word_embeddings_labels'] is not None:
            #     embeddings_metadata = {}
            #     if len(params['tensorboard_params']['embeddings_layer_names']) != len(params['tensorboard_params']['word_embeddings_labels']):
            #         raise AssertionError('The number of "embeddings_layer_names" and "word_embeddings_labels" do not match. Currently, '
            #                              'we have %d "embeddings_layer_names" and %d "word_embeddings_labels"' %
            #                              (len(params['tensorboard_params']['embeddings_layer_names']),
            #                               len(params['tensorboard_params']['word_embeddings_labels'])))
            #     # Prepare word embeddings mapping
            #     for i, layer_name in list(enumerate(params['tensorboard_params']['embeddings_layer_names'])):
            #         layer_label = params['tensorboard_params']['word_embeddings_labels'][i]
            #         mapping_name = layer_label + '.tsv'
            #         dict2file(ds.vocabulary[layer_label]['words2idx'],
            #                   self.model_path + '/' + params['tensorboard_params']['log_dir'] + '/' + mapping_name,
            #                   title='Word\tIndex',
            #                   separator='\t')
            #         embeddings_metadata[layer_name] = mapping_name

            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=self.model_path + '/' + params['tensorboard_params']['log_dir'],
                histogram_freq=params['tensorboard_params']['histogram_freq'],
                batch_size=params['tensorboard_params']['batch_size'],
                write_graph=params['tensorboard_params']['write_graph'],
                write_grads=params['tensorboard_params']['write_grads'],
                write_images=params['tensorboard_params']['write_images'],
                # embeddings_freq=params['tensorboard_params']['embeddings_freq'],
                # embeddings_layer_names=params['tensorboard_params']['embeddings_layer_names'],
                # embeddings_metadata=embeddings_metadata
            )
            callback_tensorboard.set_model(self.model)
            callbacks.append(callback_tensorboard)

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
        if False:  # TODO: loss calculation on val set is deactivated
            # if 'val' in params['eval_on_sets']:
            # Calculate how many validation iterations are we going to perform per test
            n_valid_samples = ds.len_val
            if params['num_iterations_val'] is None:
                params['num_iterations_val'] = int(math.ceil(float(n_valid_samples) / params['batch_size']))

            # prepare data generator
            if params['n_parallel_loaders'] > 1:
                val_gen = Parallel_Data_Batch_Generator('val', self, ds, params['num_iterations_val'],
                                                        batch_size=params['batch_size'],
                                                        normalization=params['normalize'],
                                                        normalization_type=params['normalization_type'],
                                                        data_augmentation=False,
                                                        mean_substraction=params['mean_substraction'],
                                                        shuffle=False,
                                                        n_parallel_loaders=params['n_parallel_loaders']).generator()
            else:
                val_gen = Data_Batch_Generator('val', self, ds, params['num_iterations_val'],
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
        if params.get('n_gpus', 1) > 1 and hasattr(self, 'multi_gpu_model') and self.multi_gpu_model is not None:
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
                                         class_weight=class_weight,
                                         max_queue_size=params['n_parallel_loaders'],
                                         workers=1,
                                         initial_epoch=params['epoch_offset'])

    def __train_from_samples(self, x, y, params, class_weight=None, sample_weight=None):

        if params['verbose'] > 0:
            logger.info(print_dict(params, header="Training parameters: "))

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
            callback_store_model = StoreModelWeightsOnEpochEnd(self, saveModel, params['epochs_for_save'])
            callbacks.append(callback_store_model)

        # Tensorboard callback
        if params['tensorboard'] and K.backend() == 'tensorflow':
            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=self.model_path + '/' + params['tensorboard_params']['log_dir'],
                histogram_freq=params['tensorboard_params']['histogram_freq'],
                batch_size=params['tensorboard_params']['batch_size'],
                write_graph=params['tensorboard_params']['write_graph'],
                write_grads=params['tensorboard_params']['write_grads'],
                write_images=params['tensorboard_params']['write_images'],
                # embeddings_freq=params['tensorboard_params']['embeddings_freq'],
                # embeddings_layer_names=params['tensorboard_params']['embeddings_layer_names'],
                # embeddings_metadata=params['tensorboard_params']['embeddings_metadata']
            )
            callbacks.append(callback_tensorboard)

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

    def testNet(self, ds, parameters, out_name=None):
        """
        Evaluate the model on a given split.
        :param ds: Dataset
        :param parameters: Parameters
        :param out_name: Deprecated.
        :return:
        """
        # Check input parameters and recover default values if needed
        params = checkParameters(parameters, self.defaut_test_params)
        self.testing_parameters.append(copy.copy(params))

        logger.info("<<< Testing model >>>")

        # Calculate how many test iterations are we going to perform
        n_samples = ds.len_test
        num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))

        # Test model
        # We won't use an Homogeneous_Batch_Generator for testing
        if params['n_parallel_loaders'] > 1:
            data_gen = Parallel_Data_Batch_Generator('test', self, ds, num_iterations,
                                                     batch_size=params['batch_size'],
                                                     normalization=params['normalize'],
                                                     normalization_type=params['normalization_type'],
                                                     data_augmentation=False,
                                                     wo_da_patch_type=params['wo_da_patch_type'],
                                                     mean_substraction=params['mean_substraction'],
                                                     n_parallel_loaders=params['n_parallel_loaders']).generator()
        else:
            data_gen = Data_Batch_Generator('test', self, ds, num_iterations,
                                            batch_size=params['batch_size'],
                                            normalization=params['normalize'],
                                            normalization_type=params['normalization_type'],
                                            data_augmentation=False,
                                            wo_da_patch_type=params['wo_da_patch_type'],
                                            mean_substraction=params['mean_substraction']).generator()

        out = self.model.evaluate_generator(data_gen,
                                            val_samples=n_samples,
                                            max_q_size=params['n_parallel_loaders'],
                                            nb_worker=1,  # params['n_parallel_loaders'],
                                            pickle_safe=False,
                                            )

        # Display metrics results
        for name, o in zip(self.model.metrics_names, out):
            logger.info('test ' + name + ': %0.8s' % o)

            # loss_all = out[0]
            # loss_ecoc = out[1]
            # loss_final = out[2]
            # acc_ecoc = out[3]
            # acc_final = out[4]
            # logger.info('Test loss: %0.8s' % loss_final)
            # logger.info('Test accuracy: %0.8s' % acc_final)

    def testNetSamples(self, X, batch_size=50):
        """
            Applies a forward pass on the samples provided and returns the predicted classes and probabilities.
        """
        classes = self.model.predict_classes(X, batch_size=batch_size)
        probs = self.model.predict_proba(X, batch_size=batch_size)

        return [classes, probs]

    def testOnBatch(self, X, Y, accuracy=True, out_name=None):
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
            [data, last_output] = self._prepareGraphData(X, Y)
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
    def predict_cond(self, X, states_below, params, ii):
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
        ##########################################
        # Choose model to use for sampling
        ##########################################
        model = self.model
        for model_input in params['model_inputs']:
            if X[model_input].shape[0] == 1:
                in_data[model_input] = np.repeat(X[model_input], n_samples, axis=0)
            else:
                in_data[model_input] = X[model_input]

        in_data[params['model_inputs'][params['state_below_index']]] = states_below
        ##########################################
        # Recover output identifiers
        ##########################################
        # in any case, the first output of the models must be the next words' probabilities
        output_ids_list = params['model_outputs']
        pick_idx = ii

        ##########################################
        # Apply prediction on current timestep
        ##########################################
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

        ##########################################
        # Get outputs
        ##########################################

        if len(output_ids_list) > 1:
            all_data = {}
            for output_id in range(len(output_ids_list)):
                all_data[output_ids_list[output_id]] = out_data[output_id]
            all_data[output_ids_list[0]] = all_data[output_ids_list[0]][:, pick_idx, :]
        else:
            all_data = {output_ids_list[0]: out_data[:, pick_idx, :]}
        probs = all_data[output_ids_list[0]]

        ##########################################
        # Define returned data
        ##########################################
        return probs

    def predict_cond_optimized(self, X, states_below, params, ii, prev_out):
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

    def beam_search(self, X, params, return_alphas=False, eos_sym=0, null_sym=2):
        """
        DEPRECATED, use search.beam_search instead.
        """
        logger.warning("Deprecated function, use search.beam_search instead.")
        return beam_search(self, X, params, return_alphas=return_alphas, eos_sym=eos_sym, null_sym=null_sym)

    def BeamSearchNet(self, ds, parameters):
        """
        DEPRECATED, use predictBeamSearchNet() instead.
        """
        logger.warning("Deprecated function, use predictBeamSearchNet() instead.")
        return self.predictBeamSearchNet(ds, parameters)

    def predictBeamSearchNet(self, ds, parameters=None):
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
        params = checkParameters(parameters, self.default_predict_with_beam_params)
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
            print ("")
            print("", file=sys.stderr)
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
                        n_samples = min(eval("ds.len_" + s), params['max_eval_samples'])
                    else:
                        n_samples = eval("ds.len_" + s)

                    num_iterations = int(math.ceil(float(n_samples)))  # / params['max_batch_size']))
                    n_samples = min(eval("ds.len_" + s), num_iterations)  # * params['batch_size'])
                    # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                    if params['n_parallel_loaders'] > 1:
                        data_gen_instance = Parallel_Data_Batch_Generator(s,
                                                                          self,
                                                                          ds,
                                                                          num_iterations,
                                                                          batch_size=1,
                                                                          normalization=params['normalize'],
                                                                          normalization_type=params['normalization_type'],
                                                                          data_augmentation=False,
                                                                          mean_substraction=params['mean_substraction'],
                                                                          predict=True,
                                                                          n_parallel_loaders=params['n_parallel_loaders'])
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
                        data_gen_instance = Parallel_Data_Batch_Generator(s, self, ds, num_iterations,
                                                                          batch_size=1,
                                                                          normalization=params['normalize'],
                                                                          normalization_type=params['normalization_type'],
                                                                          data_augmentation=False,
                                                                          mean_substraction=params['mean_substraction'],
                                                                          predict=False,
                                                                          random_samples=n_samples,
                                                                          temporally_linked=params['temporally_linked'],
                                                                          n_parallel_loaders=params['n_parallel_loaders'])
                    else:
                        data_gen_instance = Data_Batch_Generator(s, self, ds, num_iterations,
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
                if params['pos_unk']:
                    best_alphas = []
                    sources = []

                total_cost = 0
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
                                x[input_id] = ds.loadText([' '.join(prev_x)], ds.vocabulary[input_id],
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
                            scores = [co / lp + cov_p for co, lp, cov_p in zip(scores, length_penalties, coverage_penalties)]

                        elif params['normalize_probs']:
                            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
                            scores = [co / cn for co, cn in zip(scores, counts)]

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

                        # store outputs for temporally-linked models
                        if params['temporally_linked']:
                            first_idx = max(0, data_gen_instance.first_idx)
                            # TODO: Make it more general
                            for (output_id, input_id) in iteritems(self.matchings_sample_to_next_sample):
                                # Get all words previous to the padding
                                previous_outputs[input_id][first_idx + sampled - 1] = best_sample[:sum(
                                    [int(elem > 0) for elem in best_sample])]

                sys.stdout.write('\n Total cost of the translations: %f \t Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
                sys.stdout.write('The sampling took: %f secs (Speed: %f sec/sample)\n' % ((time.time() - start_time), (time.time() - start_time) / n_samples))

                sys.stdout.flush()

                if params['pos_unk']:
                    if eval('ds.loaded_raw_' + s + '[0]'):
                        sources = file2list(eval('ds.X_raw_' + s + '["raw_' + params['model_inputs'][0] + '"]'),
                                            stripfile=False)
                    predictions[s] = (np.asarray(best_samples), best_alphas, sources)
                else:
                    predictions[s] = np.asarray(best_samples)
        del data_gen
        del data_gen_instance
        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling

    def predictNet(self, ds, parameters=None, postprocess_fun=None):
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
        params = checkParameters(parameters, self.default_predict_params)

        model_predict = getattr(self, params['model_name'])  # recover model for prediction
        predictions = dict()
        for s in params['predict_on_sets']:
            predictions[s] = []
            if params['verbose'] > 0:
                print("", file=sys.stderr)
                logger.info("<<< Predicting outputs of " + s + " set >>>")
                logger.info(print_dict(params, header="Prediction parameters: "))

            # Calculate how many iterations are we going to perform
            if params['n_samples'] is None:
                if params['init_sample'] > -1 and params['final_sample'] > -1:
                    n_samples = params['final_sample'] - params['init_sample']
                else:
                    n_samples = eval("ds.len_" + s)
                num_iterations = int(math.ceil(float(n_samples) / params['batch_size']))
                n_samples = min(eval("ds.len_" + s), num_iterations * params['batch_size'])

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
                                                             n_parallel_loaders=params['n_parallel_loaders']).generator()
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
                        last_processed = min(processed_samples + params['batch_size'], n_samples)
                        out = postprocess_fun[0](out, postprocess_fun[1][processed_samples:last_processed])
                    else:
                        out = postprocess_fun(out)
                    predictions[s] += out

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

        return predictions

    def predictOnBatch(self, X, in_name=None, out_name=None, expand=False):
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
        if isinstance(self.model, Model):  # Graph
            if out_name:
                predictions = predictions[out_name]
        elif isinstance(self.model, Sequential):  # Sequential
            predictions = predictions[0]

        return predictions

    # ------------------------------------------------------- #
    #       SCORING FUNCTIONS
    #           Functions for making scoring (x, y) samples
    # ------------------------------------------------------- #

    def score_cond_model(self, X, Y, params, null_sym=2):
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
                                                         pad_sequences=True, verbose=0)[0]
                    score = self.score_cond_model(x, y, params, null_sym=self.dataset.extra_words['<null>'])
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
    def sampling(scores, sampling_type='max_likelihood', temperature=1.0):
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
        return sampling(scores, sampling_type=sampling_type, temperature=temperature)

    @staticmethod
    def decode_predictions(preds, temperature, index2word, sampling_type, verbose=0):
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
        return decode_predictions(preds, temperature, index2word, sampling_type, verbose=verbose)

    @staticmethod
    def replace_unknown_words(src_word_seq, trg_word_seq, hard_alignment, unk_symbol,
                              heuristic=0, mapping=None, verbose=0):
        """
        Replaces unknown words from the target sentence according to some heuristic.
        Borrowed from: https://github.com/sebastien-j/LV_groundhog/blob/master/experiments/nmt/replace_UNK.py
        :param src_word_seq: Source sentence words
        :param trg_word_seq: Hypothesis words
        :param hard_alignment: Target-Source alignments
        :param unk_symbol: Symbol in trg_word_seq to replace
        :param heuristic: Heuristic (0, 1, 2)
        :param mapping: External alignment dictionary
        :param verbose: Verbosity level
        :return: trg_word_seq with replaced unknown words
        """
        logger.warning("Deprecated function, use utils.replace_unknown_words() instead.")
        return replace_unknown_words(src_word_seq, trg_word_seq, hard_alignment, unk_symbol,
                                     heuristic=heuristic, mapping=mapping, verbose=verbose)

    @staticmethod
    def decode_predictions_beam_search(preds, index2word, alphas=None, heuristic=0,
                                       x_text=None, unk_symbol='<unk>', pad_sequences=False,
                                       mapping=None, verbose=0):
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
        return decode_predictions_beam_search(preds, index2word, alphas=alphas, heuristic=heuristic,
                                              x_text=x_text, unk_symbol=unk_symbol, pad_sequences=pad_sequences,
                                              mapping=mapping, verbose=verbose)

    @staticmethod
    def one_hot_2_indices(preds, pad_sequences=True, verbose=0):
        """
        Converts a one-hot codification into a index-based one
        :param pad_sequences:
        :param preds: Predictions codified as one-hot vectors.
        :param verbose: Verbosity level, by default 0.
        :return: List of converted predictions
        """
        logger.warning("Deprecated function, use utils.one_hot_2_indices() instead.")
        return one_hot_2_indices(preds, pad_sequences=pad_sequences, verbose=verbose)

    @staticmethod
    def decode_predictions_one_hot(preds, index2word, verbose=0):
        """
        Decodes predictions following a one-hot codification.
        :param preds: Predictions codified as one-hot vectors.
        :param index2word: Mapping from word indices into word characters.
        :param verbose: Verbosity level, by default 0.
        :return: List of decoded predictions
        """
        logger.warning("Deprecated function, use utils.decode_predictions_one_hot() instead.")
        return decode_predictions_one_hot(preds, index2word, verbose=verbose)

    def prepareData(self, X_batch, Y_batch=None):
        """
        Prepares the data for the model, depending on its type (Sequential, Model, Graph).
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

    def _prepareSequentialData(self, X, Y=None, sample_weights=False):

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
    def _getGraphAccuracy(data, prediction, topN=5):
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
    def _getSequentialAccuracy(GT, pred, topN=5):
        """
            Calculates the topN accuracy obtained from a set of samples on a Sequential model.
        """
        top_pred = np.argsort(pred, axis=1)[:, ::-1][:, :np.min([topN, pred.shape[1]])]
        pred = categorical_probas_to_classes(pred)
        GT = np_utils.categorical_probas_to_classes(GT)

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

    def __str__(self):
        """
        Plot basic model information.
        """

        # if(isinstance(self.model, Model)):
        print_summary(self.model.layers)
        return ''

    def log(self, mode, data_type, value):
        """
        Stores the train and val information for plotting the training progress.

        :param mode: 'train', 'val' or 'test'
        :param data_type: 'iteration', 'loss', 'accuracy', etc.
        :param value: numerical value taken by the data_type
        """
        if mode not in self.__modes:
            raise Exception('The provided mode "' + mode + '" is not valid.')
        # if data_type not in self.__data_types:
        #    raise Exception('The provided data_type "'+ data_type +'" is not valid.')

        if mode not in self.__logger:
            self.__logger[mode] = dict()
        if data_type not in self.__logger[mode]:
            self.__logger[mode][data_type] = list()
        self.__logger[mode][data_type].append(value)

    def getLog(self, mode, data_type):
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

    def plot(self, time_measure, metrics, splits, upperbound=None, colours_shapes_dict=None):
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
        if colours_shapes_dict is None:
            colours_shapes_dict = dict()

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
                # plt.subplot(211)
                # plt.plot(iterations, loss, colours['train_loss']+'o')
                plt.plot(iterations, measure, colours_shapes_dict[met + '_' + sp], label=str(met))

        max_iter = np.max(all_iterations + [0])

        # Plot upperbound
        if upperbound is not None:
            # plt.subplot(211)
            plt.plot([0, max_iter], [upperbound, upperbound], 'r-')
            plt.axis([0, max_iter, 0, upperbound])  # limit height to 1

        # Fill labels
        plt.xlabel(time_measure)
        # plt.subplot(211)
        plt.title('Training progress')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Create plots dir
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Save figure
        plot_file = self.model_path + '/' + time_measure + '_' + str(max_iter) + '.jpg'
        plt.savefig(plot_file)
        if not self.silence:
            print("", file=sys.stderr)
            logger.info("<<< Progress plot saved in " + plot_file + ' >>>')

        # Close plot window
        plt.close()

    # ------------------------------------------------------- #
    #   MODELS
    #       Available definitions of CNN models (see basic_model as an example)
    #       All the models must include the following parameters:
    #           nOutput, input
    # ------------------------------------------------------- #

    def basic_model(self, nOutput, model_input):
        """
            Builds a basic CNN model.
        """

        # Define inputs and outputs IDs
        self.ids_inputs = ['input']
        self.ids_outputs = ['output']

        if len(model_input) == 3:
            input_shape = tuple([model_input[2]] + model_input[0:2])
        else:
            input_shape = tuple(model_input)

        inp = Input(shape=input_shape, name='input')

        # model_input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        x = Conv2D(32, (3, 3), padding='valid')(inp)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='valid')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        # Note: Keras does automatic shape inference.
        x = Dense(1024)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(nOutput)(x)
        out = Activation('softmax', name='output')(x)

        self.model = Model(inputs=[inp], outputs=[out])

    def basic_model_seq(self, nOutput, input_shape):
        """
            Builds a basic CNN model.
        """

        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, (3, 3), padding='valid', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        # Note: Keras does automatic shape inference.
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nOutput))
        self.model.add(Activation('softmax'))

    def One_vs_One(self, nOutput, input_shape):
        """
            Builds a simple One_vs_One network with 3 convolutional layers (useful for ECOC models).
        """
        # default lr=0.1, momentum=0.
        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=input_shape))  # default input_shape=(3,224,224)
        self.model.add(Conv2D(32, (1, 1), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(8, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(1, 1)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax'))  # default nOutput=1000

    def VGG_16(self, nOutput, input_shape):
        """
            Builds a VGG model with 16 layers.
        """
        # default lr=0.1, momentum=0.
        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=input_shape))  # default input_shape=(3,224,224)
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax'))  # default nOutput=1000

    def VGG_16_PReLU(self, nOutput, input_shape):
        """
            Builds a VGG model with 16 layers and with PReLU activations.
        """

        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=input_shape))  # default input_shape=(3,224,224)
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3)))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(PReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(PReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax'))  # default nOutput=1000

    def VGG_16_FunctionalAPI(self, nOutput, input_shape):
        """
            16-layered VGG model implemented in Keras' Functional API
        """
        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        vis_input = Input(shape=input_shape, name="vis_input")

        x = ZeroPadding2D((1, 1))(vis_input)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2),
                         name='last_max_pool')(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5, name='last_dropout')(x)
        x = Dense(nOutput, activation='softmax', name='output')(x)  # nOutput=1000 by default

        self.model = Model(inputs=[vis_input], outputs=[x])

    def VGG_19(self, nOutput, input_shape):
        """
        19-layered VGG model implemented in Keras' Functional API
        """
        # Define inputs and outputs IDs
        self.ids_inputs = ['input_1']
        self.ids_outputs = ['predictions']
        from keras.applications.vgg19 import VGG19

        # Load VGG19 model pre-trained on ImageNet
        self.model = VGG19()

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model
        out = self.model.get_layer('fc2').output
        out = Dense(nOutput, name=self.ids_outputs[0], activation='softmax')(out)

        self.model = Model(inputs=[image], outputs=[out])

    def VGG_19_ImageNet(self, nOutput, input_shape):
        """
        19-layered VGG model implemented in Keras' Functional API trained on Imagenet.
        """
        # Define inputs and outputs IDs
        self.ids_inputs = ['input_1']
        self.ids_outputs = ['predictions']
        from keras.applications.vgg19 import VGG19

        # Load VGG19 model pre-trained on ImageNet
        self.model = VGG19(weights='imagenet', layers_lr=0.001)

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model
        out = self.model.get_layer('fc2').output
        out = Dense(nOutput, name=self.ids_outputs[0], activation='softmax')(out)

        self.model = Model(inputs=[image], outputs=[out])

    ########################################
    # GoogLeNet implementation from http://dandxy89.github.io/ImageModels/googlenet/
    ########################################

    @staticmethod
    def inception_module(x, params, dim_ordering, concat_axis,
                         subsample=(1, 1), activation='relu',
                         border_mode='same', weight_decay=None):

        # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
        # file-googlenet_neon-py

        (branch1, branch2, branch3, branch4) = params

        if weight_decay:
            W_regularizer = l2(weight_decay)
            b_regularizer = l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        pathway1 = Conv2D(branch1[0], (1, 1),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(x)

        pathway2 = Conv2D(branch2[0], (1, 1),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(x)
        pathway2 = Conv2D(branch2[1], (3, 3),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(pathway2)

        pathway3 = Conv2D(branch3[0], (1, 1),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(x)
        pathway3 = Conv2D(branch3[1], (5, 5),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(pathway3)

        pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=dim_ordering)(x)
        pathway4 = Conv2D(branch4[0], (1, 1),
                          subsample=subsample,
                          activation=activation,
                          padding=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(pathway4)

        return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)

    @staticmethod
    def conv_layer(x, nb_filter, nb_row, nb_col, dim_ordering,
                   subsample=(1, 1), activation='relu',
                   border_mode='same', weight_decay=None, padding=None):

        if weight_decay:
            W_regularizer = l2(weight_decay)
            b_regularizer = l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        x = Conv2D(nb_filter, (nb_row, nb_col),
                   subsample=subsample,
                   activation=activation,
                   padding=border_mode,
                   W_regularizer=W_regularizer,
                   b_regularizer=b_regularizer,
                   bias=False,
                   dim_ordering=dim_ordering)(x)

        if padding:
            for _ in range(padding):
                x = ZeroPadding2D(padding=(1, 1), dim_ordering=dim_ordering)(x)

        return x

    def GoogLeNet_FunctionalAPI(self, nOutput, input_shape):

        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        # Define image input layer
        img_input = Input(shape=input_shape, name='input_data')
        CONCAT_AXIS = 1
        NB_CLASS = nOutput  # number of classes (default 1000)
        DROPOUT = 0.4
        # Theano - 'th' (channels, width, height)
        # Tensorflow - 'tf' (width, height, channels)
        DIM_ORDERING = 'th'
        pool_name = 'last_max_pool'  # name of the last max-pooling layer

        x = self.conv_layer(img_input, nb_col=7, nb_filter=64, subsample=(2, 2),
                            nb_row=7, dim_ordering=DIM_ORDERING, padding=1)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.conv_layer(x, nb_col=1, nb_filter=64,
                            nb_row=1, dim_ordering=DIM_ORDERING)
        x = self.conv_layer(x, nb_col=3, nb_filter=192,
                            nb_row=3, dim_ordering=DIM_ORDERING, padding=1)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.inception_module(x, params=[(64,), (96, 128), (16, 32), (32,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(128,), (128, 192), (32, 96), (64,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)

        x = ZeroPadding2D(padding=(2, 2), dim_ordering=DIM_ORDERING)(x)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.inception_module(x, params=[(192,), (96, 208), (16, 48), (64,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        # AUX 1 - Branch HERE
        x = self.inception_module(x, params=[(160,), (112, 224), (24, 64), (64,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(128,), (128, 256), (24, 64), (64,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(112,), (144, 288), (32, 64), (64,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        # AUX 2 - Branch HERE
        x = self.inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING, name=pool_name)(x)

        x = self.inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                                  dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = AveragePooling2D(strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
        x = Flatten()(x)
        x = Dropout(DROPOUT)(x)
        # x = Dense(output_dim=NB_CLASS,
        #          activation='linear')(x)
        x = Dense(output_dim=NB_CLASS,
                  activation='softmax', name='output')(x)

        self.model = Model(inputs=[img_input], outputs=[x])

    def Union_Layer(self, nOutput, input_shape):
        """
        Network with just a dropout and a softmax layers which is intended to serve as the final layer for an ECOC model
        """
        if len(input_shape) == 3:
            input_shape = tuple([input_shape[2]] + input_shape[0:2])
        else:
            input_shape = tuple(input_shape)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax'))

    def add_One_vs_One_Inception(self, input_layer, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
        Builds a simple One_vs_One_Inception network with 2 inception layers on the top of the current model
        (useful for ECOC_loss models).
        """

        # Inception Ea
        out_Ea = self.__addInception('inceptionEa_' + str(id_branch), input_layer, 4, 2, 8, 2, 2, 2)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb_' + str(id_branch), out_Ea, 2, 2, 4, 2, 1, 1)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1, 1)),
                            name='ave_pool/ECOC_' + str(id_branch), input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(),
                            name='fc_OnevsOne_' + str(id_branch) + '/flatten', input='ave_pool/ECOC_' + str(id_branch))
        self.model.add_node(Dropout(0.5),
                            name='fc_OnevsOne_' + str(id_branch) + '/drop',
                            input='fc_OnevsOne_' + str(id_branch) + '/flatten')
        output_name = 'fc_OnevsOne_' + str(id_branch)
        self.model.add_node(Dense(nOutput, activation=activation),
                            name=output_name, input='fc_OnevsOne_' + str(id_branch) + '/drop')

        return output_name

    def add_One_vs_One_Inception_Functional(self, input_layer, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
        Builds a simple One_vs_One_Inception network with 2 inception layers on the top of the current model
         (useful for ECOC_loss models).
        """

        in_node = self.model.get_layer(input_layer).output

        # Inception Ea
        [out_Ea, out_Ea_name] = self.__addInception_Functional('inceptionEa_' + str(id_branch), in_node, 4, 2, 8, 2, 2,
                                                               2)
        # Inception Eb
        [out_Eb, out_Eb_name] = self.__addInception_Functional('inceptionEb_' + str(id_branch), out_Ea, 2, 2, 4, 2, 1,
                                                               1)
        # Average Pooling    pool_size=(7,7)
        x = AveragePooling2D(pool_size=input_shape, strides=(1, 1), name='ave_pool/ECOC_' + str(id_branch))(out_Eb)

        # Softmax
        output_name = 'fc_OnevsOne_' + str(id_branch)
        x = Flatten(name='fc_OnevsOne_' + str(id_branch) + '/flatten')(x)
        x = Dropout(0.5, name='fc_OnevsOne_' + str(id_branch) + '/drop')(x)
        out_node = Dense(nOutput, activation=activation, name=output_name)(x)

        return out_node

    @staticmethod
    def add_One_vs_One_3x3_Functional(input_layer, input_shape, id_branch, nkernels, nOutput=2, activation='softmax'):

        # 3x3 convolution
        out_3x3 = Conv2D(nkernels, (3, 3), name='3x3/ecoc_' + str(id_branch), activation='relu')(input_layer)

        # Average Pooling    pool_size=(7,7)
        x = AveragePooling2D(pool_size=input_shape, strides=(1, 1), name='ave_pool/ecoc_' + str(id_branch))(out_3x3)

        # Softmax
        output_name = 'fc_OnevsOne_' + str(id_branch) + '/out'
        x = Flatten(name='fc_OnevsOne_' + str(id_branch) + '/flatten')(x)
        x = Dropout(0.5, name='fc_OnevsOne_' + str(id_branch) + '/drop')(x)
        out_node = Dense(nOutput, activation=activation, name=output_name)(x)

        return out_node

    @staticmethod
    def add_One_vs_One_3x3_double_Functional(input_layer, input_shape, id_branch, nOutput=2, activation='softmax'):

        # 3x3 convolution
        out_3x3 = Conv2D(64, (3, 3), name='3x3_1/ecoc_' + str(id_branch), activation='relu')(input_layer)

        # Max Pooling
        x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name='max_pool/ecoc_' + str(id_branch))(out_3x3)

        # 3x3 convolution
        x = Conv2D(32, (3, 3), name='3x3_2/ecoc_' + str(id_branch), activation='relu')(x)

        # Softmax
        output_name = 'fc_OnevsOne_' + str(id_branch) + '/out'
        x = Flatten(name='fc_OnevsOne_' + str(id_branch) + '/flatten')(x)
        x = Dropout(0.5, name='fc_OnevsOne_' + str(id_branch) + '/drop')(x)
        out_node = Dense(nOutput, activation=activation, name=output_name)(x)

        return out_node

    def add_One_vs_One_Inception_v2(self, input_layer, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
            Builds a simple One_vs_One_Inception_v2 network with 2 inception layers on the top of the current model
            (useful for ECOC_loss models).
        """

        # Inception Ea
        out_Ea = self.__addInception('inceptionEa_' + str(id_branch), input_layer, 16, 8, 32, 8, 8, 8)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb_' + str(id_branch), out_Ea, 8, 8, 16, 8, 4, 4)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1, 1)),
                            name='ave_pool/ECOC_' + str(id_branch), input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(),
                            name='fc_OnevsOne_' + str(id_branch) + '/flatten', input='ave_pool/ECOC_' + str(id_branch))
        self.model.add_node(Dropout(0.5),
                            name='fc_OnevsOne_' + str(id_branch) + '/drop',
                            input='fc_OnevsOne_' + str(id_branch) + '/flatten')
        output_name = 'fc_OnevsOne_' + str(id_branch)
        self.model.add_node(Dense(nOutput, activation=activation),
                            name=output_name, input='fc_OnevsOne_' + str(id_branch) + '/drop')

        return output_name

    def __addInception(self, name, input_layer, kernels_1x1, kernels_3x3_reduce, kernels_3x3, kernels_5x5_reduce,
                       kernels_5x5, kernels_pool_projection):
        """
            Adds an inception module to the model.

            :param name: string identifier of the inception layer
            :param input_layer: identifier of the layer that will serve as an input to the built inception module
            :param kernels_1x1: number of kernels of size 1x1                                      (1st branch)
            :param kernels_3x3_reduce: number of kernels of size 1x1 before the 3x3 layer          (2nd branch)
            :param kernels_3x3: number of kernels of size 3x3                                      (2nd branch)
            :param kernels_5x5_reduce: number of kernels of size 1x1 before the 5x5 layer          (3rd branch)
            :param kernels_5x5: number of kernels of size 5x5                                      (3rd branch)
            :param kernels_pool_projection: number of kernels of size 1x1 after the 3x3 pooling    (4th branch)
        """
        # Branch 1
        self.model.add_node(Conv2D(kernels_1x1, (1, 1)), name=name + '/1x1', input=input_layer)
        self.model.add_node(Activation('relu'), name=name + '/relu_1x1', input=name + '/1x1')

        # Branch 2
        self.model.add_node(Conv2D(kernels_3x3_reduce, (1, 1)), name=name + '/3x3_reduce', input=input_layer)
        self.model.add_node(Activation('relu'), name=name + '/relu_3x3_reduce', input=name + '/3x3_reduce')
        self.model.add_node(ZeroPadding2D((1, 1)), name=name + '/3x3_zeropadding', input=name + '/relu_3x3_reduce')
        self.model.add_node(Conv2D(kernels_3x3, (3, 3)), name=name + '/3x3', input=name + '/3x3_zeropadding')
        self.model.add_node(Activation('relu'), name=name + '/relu_3x3', input=name + '/3x3')

        # Branch 3
        self.model.add_node(Conv2D(kernels_5x5_reduce, (1, 1)), name=name + '/5x5_reduce', input=input_layer)
        self.model.add_node(Activation('relu'), name=name + '/relu_5x5_reduce', input=name + '/5x5_reduce')
        self.model.add_node(ZeroPadding2D((2, 2)), name=name + '/5x5_zeropadding', input=name + '/relu_5x5_reduce')
        self.model.add_node(Conv2D(kernels_5x5, (5, 5)), name=name + '/5x5', input=name + '/5x5_zeropadding')
        self.model.add_node(Activation('relu'), name=name + '/relu_5x5', input=name + '/5x5')

        # Branch 4
        self.model.add_node(ZeroPadding2D((1, 1)), name=name + '/pool_zeropadding', input=input_layer)
        self.model.add_node(MaxPooling2D((3, 3), strides=(1, 1)), name=name + '/pool', input=name + '/pool_zeropadding')
        self.model.add_node(Conv2D(kernels_pool_projection, (1, 1)), name=name + '/pool_proj', input=name + '/pool')
        self.model.add_node(Activation('relu'), name=name + '/relu_pool_proj', input=name + '/pool_proj')

        # Concatenate
        inputs_list = [name + '/relu_1x1', name + '/relu_3x3', name + '/relu_5x5', name + '/relu_pool_proj']
        out_name = name + '/concat'
        self.model.add_node(Activation('linear'), name=out_name, inputs=inputs_list, concat_axis=1)

        return out_name

    @staticmethod
    def __addInception_Functional(name, input_layer, kernels_1x1, kernels_3x3_reduce, kernels_3x3,
                                  kernels_5x5_reduce, kernels_5x5, kernels_pool_projection):
        """
            Adds an inception module to the model.

            :param name: string identifier of the inception layer
            :param input_layer: identifier of the layer that will serve as an input to the built inception module
            :param kernels_1x1: number of kernels of size 1x1                                      (1st branch)
            :param kernels_3x3_reduce: number of kernels of size 1x1 before the 3x3 layer          (2nd branch)
            :param kernels_3x3: number of kernels of size 3x3                                      (2nd branch)
            :param kernels_5x5_reduce: number of kernels of size 1x1 before the 5x5 layer          (3rd branch)
            :param kernels_5x5: number of kernels of size 5x5                                      (3rd branch)
            :param kernels_pool_projection: number of kernels of size 1x1 after the 3x3 pooling    (4th branch)
        """
        # Branch 1
        x_b1 = Conv2D(kernels_1x1, (1, 1), name=name + '/1x1', activation='relu')(input_layer)

        # Branch 2
        x_b2 = Conv2D(kernels_3x3_reduce, (1, 1), name=name + '/3x3_reduce', activation='relu')(input_layer)
        x_b2 = ZeroPadding2D((1, 1), name=name + '/3x3_zeropadding')(x_b2)
        x_b2 = Conv2D(kernels_3x3, (3, 3), name=name + '/3x3', activation='relu')(x_b2)

        # Branch 3
        x_b3 = Conv2D(kernels_5x5_reduce, (1, 1), name=name + '/5x5_reduce', activation='relu')(input_layer)
        x_b3 = ZeroPadding2D((2, 2), name=name + '/5x5_zeropadding')(x_b3)
        x_b3 = Conv2D(kernels_5x5, (5, 5), name=name + '/5x5', activation='relu')(x_b3)

        # Branch 4
        x_b4 = ZeroPadding2D((1, 1), name=name + '/pool_zeropadding')(input_layer)
        x_b4 = MaxPooling2D((3, 3), strides=(1, 1), name=name + '/pool')(x_b4)
        x_b4 = Conv2D(kernels_pool_projection, (1, 1), name=name + '/pool_proj', activation='relu')(x_b4)

        # Concatenate
        out_name = name + '/concat'
        out_node = concatenate([x_b1, x_b2, x_b3, x_b4], axis=1, name=out_name)

        return [out_node, out_name]

    def add_One_vs_One_Merge(self, inputs_list, nOutput, activation='softmax'):

        self.model.add_node(Flatten(), name='ecoc_loss', inputs=inputs_list,
                            merge_mode='concat')  # join outputs from OneVsOne classifiers
        self.model.add_node(Dropout(0.5), name='final_loss/drop', input='ecoc_loss')
        self.model.add_node(Dense(nOutput, activation=activation), name='final_loss',
                            input='final_loss/drop')  # apply final joint prediction

        # Outputs
        self.model.add_output(name='ecoc_loss/output', input='ecoc_loss')
        self.model.add_output(name='final_loss/output', input='final_loss')

        return ['ecoc_loss/output', 'final_loss/output']

    def add_One_vs_One_Merge_Functional(self, inputs_list, nOutput, activation='softmax'):

        # join outputs from OneVsOne classifiers
        ecoc_loss_name = 'ecoc_loss'
        final_loss_name = 'final_loss/out'
        ecoc_loss = concatenate(inputs_list, name=ecoc_loss_name, axis=1)
        drop = Dropout(0.5, name='final_loss/drop')(ecoc_loss)
        # apply final joint prediction
        final_loss = Dense(nOutput, activation=activation, name=final_loss_name)(drop)

        in_node = self.model.layers[0].name
        in_node = self.model.get_layer(in_node).output
        self.model = Model(inputs=[in_node], outputs=[ecoc_loss, final_loss])

        return [ecoc_loss_name, final_loss_name]

    ##############################
    #       DENSE NETS
    ##############################

    def add_dense_block(self, input_layer, nb_layers, k, drop, init_weights, name=None):
        """
        Adds a Dense Block for the transition down path.

        # References
            Jegou S, Drozdzal M, Vazquez D, Romero A, Bengio Y.
            The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
            arXiv preprint arXiv:1611.09326. 2016 Nov 28.

        :param name:
        :param input_layer: input layer to the dense block.
        :param nb_layers: number of dense layers included in the dense block (see self.add_dense_layer()
                          for information about the internal layers).
        :param k: growth rate. Number of additional feature maps learned at each layer.
        :param drop: dropout rate.
        :param init_weights: weights initialization function
        :return: output layer of the dense block
        """
        if K.image_dim_ordering() == 'th':
            axis = 1
        elif K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            raise ValueError('Invalid dim_ordering:', K.image_dim_ordering)

        list_outputs = []
        prev_layer = input_layer
        for n in range(nb_layers):
            if name is not None:
                name_dense = name + '_' + str(n)
                name_merge = 'merge' + name + '_' + str(n)
            else:
                name_dense = None
                name_merge = None

            # Insert dense layer
            new_layer = self.add_dense_layer(prev_layer, k, drop, init_weights, name=name_dense)
            list_outputs.append(new_layer)
            # Merge with previous layer
            prev_layer = concatenate([new_layer, prev_layer], axis=axis, name=name_merge)

        return concatenate(list_outputs, axis=axis, name=name_merge)

    @staticmethod
    def add_dense_layer(input_layer, k, drop, init_weights, name=None):
        """
        Adds a Dense Layer inside a Dense Block, which is composed of BN, ReLU, Conv and Dropout

        # References
            Jegou S, Drozdzal M, Vazquez D, Romero A, Bengio Y.
            The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
            arXiv preprint arXiv:1611.09326. 2016 Nov 28.

        :param name:
        :param input_layer: input layer to the dense block.
        :param k: growth rate. Number of additional feature maps learned at each layer.
        :param drop: dropout rate.
        :param init_weights: weights initialization function
        :return: output layer
        """

        if name is not None:
            name_batch = 'batchnormalization' + name
            name_activ = 'activation' + name
            name_conv = 'convolution2d' + name
            name_drop = 'dropout' + name
        else:
            name_batch = None
            name_activ = None
            name_conv = None
            name_drop = None

        out_layer = BatchNormalization(mode=2, axis=1, name=name_batch)(input_layer)
        out_layer = Activation('relu', name=name_activ)(out_layer)
        out_layer = Conv2D(k, (3, 3), kernel_initializer=init_weights, padding='same', name=name_conv)(out_layer)
        if drop > 0.0:
            out_layer = Dropout(drop, name=name_drop)(out_layer)
        return out_layer

    def add_transitiondown_block(self, input_layer,
                                 nb_filters_conv, pool_size, init_weights,
                                 nb_layers, growth, drop):
        """
        Adds a Transition Down Block. Consisting of BN, ReLU, Conv and Dropout, Pooling, Dense Block.

        # References
            Jegou S, Drozdzal M, Vazquez D, Romero A, Bengio Y.
            The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
            arXiv preprint arXiv:1611.09326. 2016 Nov 28.

        # Input layers parameters
        :param input_layer: input layer.

        # Convolutional layer parameters
        :param nb_filters_conv: number of convolutional filters to learn.
        :param pool_size: size of the max pooling operation (2 in reference paper)
        :param init_weights: weights initialization function

        # Dense Block parameters
        :param nb_layers: number of dense layers included in the dense block (see self.add_dense_layer()
                          for information about the internal layers).
        :param growth: growth rate. Number of additional feature maps learned at each layer.
        :param drop: dropout rate.

        :return: [output layer, skip connection name]
        """
        if K.image_dim_ordering() == 'th':
            axis = 1
        elif K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            raise ValueError('Invalid dim_ordering:', K.image_dim_ordering)

        # Dense Block
        x_dense = self.add_dense_block(input_layer, nb_layers, growth, drop,
                                       init_weights)  # (growth*nb_layers) feature maps added

        # Concatenate and skip connection recovery for upsampling path
        skip = concatenate([input_layer, x_dense], axis=axis)

        # Transition Down
        x_out = BatchNormalization(mode=2, axis=1)(skip)
        x_out = Activation('relu')(x_out)
        x_out = Conv2D(nb_filters_conv, (1, 1), kernel_initializer=init_weights, padding='same')(x_out)
        if drop > 0.0:
            x_out = Dropout(drop)(x_out)
        x_out = MaxPooling2D(pool_size=(pool_size, pool_size))(x_out)

        return [x_out, skip]

    def add_transitionup_block(self, input_layer, skip_conn,
                               nb_filters_deconv, init_weights,
                               nb_layers, growth, drop, name=None):
        """
        Adds a Transition Up Block. Consisting of Deconv, Skip Connection, Dense Block.

        # References
            Jegou S, Drozdzal M, Vazquez D, Romero A, Bengio Y.
            The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.
            arXiv preprint arXiv:1611.09326. 2016 Nov 28.

        # Input layers parameters
        :param name:
        :param input_layer: input layer.
        :param skip_conn: list of layers to be used as skip connections.

        # Deconvolutional layer parameters
        :param nb_filters_deconv: number of deconvolutional filters to learn.
        :param init_weights: weights initialization function

        # Dense Block parameters
        :param nb_layers: number of dense layers included in the dense block (see self.add_dense_layer()
                          for information about the internal layers).
        :param growth: growth rate. Number of additional feature maps learned at each layer.
        :param drop: dropout rate.

        :return: output layer
        """

        if K.image_dim_ordering() == 'th':
            axis = 1
        elif K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            raise ValueError('Invalid dim_ordering:', K.image_dim_ordering)

        input_layer = Conv2DTranspose(nb_filters_deconv, (3, 3),
                                      strides=(2, 2),
                                      kernel_initializer=init_weights, padding='valid')(input_layer)

        # Skip connection concatenation
        input_layer = Concatenate(axis=axis, cropping=[None, None, 'center', 'center'])([skip_conn, input_layer])

        # Dense Block
        input_layer = self.add_dense_block(input_layer, nb_layers, growth, drop, init_weights,
                                           name=name)  # (growth*nb_layers) feature maps added
        return input_layer

    @staticmethod
    def Empty(nOutput, input_layer):
        """
            Creates an empty Model_Wrapper (can be externally defined)
        """
        pass

    # ------------------------------------------------------- #
    #       SAVE/LOAD
    #           Auxiliary methods for saving and loading the model.
    # ------------------------------------------------------- #

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


# Backwards compatibility
CNN_Model = Model_Wrapper
