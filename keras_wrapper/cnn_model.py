from keras_wrapper.thread_loader import ThreadDataLoader, retrieveXY
from keras_wrapper.dataset import Dataset, Data_Batch_Generator, Homogeneous_Data_Batch_Generator
from keras_wrapper.ecoc_classifier import ECOC_Classifier
from keras_wrapper.callbacks_keras_wrapper import *

from keras.models import Sequential, Graph, model_from_json
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import merge, Dense, Dropout, Flatten, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.engine.training import Model
from keras.utils.layer_utils import print_summary

#from keras.caffe.extra_layers import LRN2D

import matplotlib as mpl
mpl.use('Agg') # run matplotlib without X server (GUI)
import matplotlib.pyplot as plt

import numpy as np
import cPickle as pk

import sys
import time
import os
import math
import copy
import logging
import shutil


# ------------------------------------------------------- #
#       SAVE/LOAD
#           External functions for saving and loading CNN_Model instances
# ------------------------------------------------------- #

def saveModel(model_wrapper, iter, path=None):
    """
        Saves a backup of the current CNN_Model object after trained for 'iter' iterations.
    """
    if(not path):
        path = model_wrapper.model_path

    if(not model_wrapper.silence):
        logging.info("<<< Saving model to "+ path +" ... >>>")

    # Create models dir
    if(not os.path.isdir(path)):
        os.makedirs(path)

    iter = str(iter)
    # Save model structure
    json_string = model_wrapper.model.to_json()
    open(path + '/epoch_'+ iter +'_structure.json', 'w').write(json_string)
    # Save model weights
    model_wrapper.model.save_weights(path + '/epoch_'+ iter +'_weights.h5', overwrite=True)
    # Save additional information
    pk.dump(model_wrapper, open(path + '/epoch_' + iter + '_CNN_Model.pkl', 'wb'))

    if(not model_wrapper.silence):
        logging.info("<<< Model saved >>>")


def loadModel(model_path, iter):
    """
        Loads a previously saved CNN_Model object.
    """
    t = time.time()
    iter = str(iter)
    logging.info("<<< Loading model from "+ model_path + "/epoch_" + iter + "_CNN_Model.pkl ... >>>")

    # Load model structure
    model = model_from_json(open(model_path + '/epoch_'+ iter +'_structure.json').read())
    # Load model weights
    model.load_weights(model_path + '/epoch_'+ iter +'_weights.h5')
    # Load additional information
    model_wrapper = pk.load(open(model_path + '/epoch_' + iter + '_CNN_Model.pkl', 'rb'))
    model_wrapper.model = model

    logging.info("<<< Model loaded in %0.6s seconds. >>>" % str(time.time()-t))
    return model_wrapper


# ------------------------------------------------------- #
#       MAIN CLASS
# ------------------------------------------------------- #
class CNN_Model(object):
    """
        Wrapper for Keras' models. It provides the following utilities:
            - Training visualization module.
            - Set of already implemented CNNs for quick definition.
            - Easy layers re-definition for finetuning.
            - Model backups.
            - Easy to use training and test methods.
    """

    def __init__(self, nOutput=1000, type='basic_model', silence=False, input_shape=[256, 256, 3],
                 structure_path=None, weights_path=None, seq_to_functional=False,
                 model_name=None, plots_path=None, models_path=None, inheritance=False):
        """
            CNN_Model object constructor.

            :param nOutput: number of outputs of the network. Only valid if 'structure_path' == None.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param silence: set to True if you don't want the model to output informative messages
            :param input_shape: array with 3 integers which define the images' input shape [height, width, channels]. Only valid if 'structure_path' == None.
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param seq_to_functional: indicates if we are loading a set of weights trained on a Sequential model to a Functional one
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param plots_path: path to the folder where the plots will be stored during training
            :param models_path: path to the folder where the temporal model packups will be stored
            :param inheritance: indicates if we are building an instance from a child class (in this case the model will not be built from this __init__, it should be built from the child class).
        """
        self.__toprint = ['net_type', 'name', 'plot_path', 'models_path', 'lr', 'momentum',
                            'training_parameters', 'testing_parameters', 'training_state', 'loss', 'silence']

        self.silence = silence
        self.net_type = type
        self.lr = 0.01 # insert default learning rate
        self.momentum = 1.0-self.lr # insert default momentum
        self.loss='categorical_crossentropy' # default loss function
        self.training_parameters = []
        self.testing_parameters = []
        self.training_state = dict()

        # Dictionary for storing any additional data needed
        self.additional_data = dict()

        # Inputs and outputs names for models of class Model
        self.ids_inputs = list()
        self.ids_outputs = list()

        # Prepare logger
        self.__logger = dict()
        self.__modes = ['train', 'val']
        self.__data_types = ['iteration', 'loss', 'accuracy', 'accuracy top-5']

        # Prepare model
        if(not inheritance):
            # Set Network name
            self.setName(model_name, plots_path, models_path)

            if(structure_path):
                # Load a .json model
                if(not self.silence):
                    logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
                self.model = model_from_json(open(structure_path).read())

            else:
                # Build model from scratch
                if(hasattr(self, type)):
                    if(not self.silence):
                        logging.info("<<< Building "+ type +" CNN >>>")
                    eval('self.'+type+'(nOutput, input_shape)')
                else:
                    raise Exception('CNN type "'+ type +'" is not implemented.')

            # Load weights from file
            if weights_path:
                if(not self.silence):
                    logging.info("<<< Loading weights from file "+ weights_path +" >>>")
                self.model.load_weights(weights_path, seq_to_functional=seq_to_functional)


    def setInputsMapping(self, inputsMapping):
        """
            Sets the mapping of the inputs from the format given by the dataset to the format received by the model.

            :param inputsMapping: dictionary with the model inputs' identifiers as keys and the dataset inputs' identifiers as values. If the current model is Sequential then keys must be ints with the desired input order (starting from 0). If it is Graph then keys must be str.
        """
        self.inputsMapping = inputsMapping


    def setOutputsMapping(self, outputsMapping, acc_output=None):
        """
            Sets the mapping of the outputs from the format given by the dataset to the format received by the model.

            :param outputsMapping: dictionary with the model outputs' identifiers as keys and the dataset outputs' identifiers as values. If the current model is Sequential then keys must be ints with the desired output order (in this case only one value can be provided). If it is Graph then keys must be str.
            :param acc_output: name of the model's output that will be used for calculating the accuracy of the model (only needed for Graph models)
        """
        if(isinstance(self.model, Sequential) and len(outputsMapping.keys()) > 1):
            raise Exception("When using Sequential models only one output can be provided in outputsMapping")
        self.outputsMapping = outputsMapping
        self.acc_output = acc_output


    def setOptimizer(self, lr=None, momentum=None, loss=None, metrics=None):
        """
            Sets a new optimizer for the CNN model.

            :param lr: learning rate of the network
            :param momentum: momentum of the network (if None, then momentum = 1-lr)
            :param loss: loss function applied for optimization
        """
        # Pick default parameters
        if(lr is None):
            lr = self.lr
        else:
            self.lr = lr
        if(momentum is None):
            momentum = self.momentum
        else:
            self.momentum = momentum
        if(loss is None):
            loss = self.loss
        else:
            self.loss = loss
        if(metrics is None):
            metrics = []

        #sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
        sgd = SGD(lr=lr, decay=0.0, momentum=momentum, nesterov=True)

        if(not self.silence):
            logging.info("Compiling model...")

        # compile differently depending if our model is 'Sequential', 'Model' or 'Graph'
        if(isinstance(self.model, Sequential) or isinstance(self.model, Model)):
            self.model.compile(loss=loss, optimizer=sgd, metrics=metrics)
        elif(isinstance(self.model, Graph)):
            if(not isinstance(loss, dict)):
                loss_dict = dict()
                for out in self.model.output_order:
                    loss_dict[out] = loss
            else:
                loss_dict = loss
            self.model.compile(loss=loss_dict, optimizer=sgd, metrics=metrics)
        else:
            raise NotImplementedError()

        if(not self.silence):
            logging.info("Optimizer updated, learning rate set to "+ str(lr))


    def setName(self, model_name, plots_path=None, models_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the CNN_Model instance.
        """
        if(not model_name):
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if(not plots_path):
            self.plot_path = 'Plots/' + self.name
        else:
            self.plot_path = plots_path

        if(not models_path):
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = models_path

        # Remove directories if existed
        if(clear_dirs):
            if(os.path.isdir(self.model_path)):
                shutil.rmtree(self.model_path)
            if(os.path.isdir(self.plot_path)):
                shutil.rmtree(self.plot_path)

        # Create new ones
        if(create_dirs):
            if(not os.path.isdir(self.model_path)):
                os.makedirs(self.model_path)
            if(not os.path.isdir(self.plot_path)):
                os.makedirs(self.plot_path)


    def checkParameters(self, input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key,val in input_params.iteritems():
            if key in valid_params:
                params[key] = val
            else:
                raise Exception("Parameter '"+ key +"' is not a valid parameter.")

        # Use default parameters if not provided
        for key,default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params


    # ------------------------------------------------------- #
    #       MODEL MODIFICATION
    #           Methods for modifying specific layers of the network
    # ------------------------------------------------------- #


    def replaceLastLayers(self, num_remove, new_layers):
        """
            Replaces the last 'num_remove' layers in the model by the newly defined in 'new_layers'.
            Function only valid for Sequential models. Use self.removeLayers(...) for Graph models.
        """
        if(not self.silence):
            logging.info("Replacing layers...")

        removed_layers = []
        removed_params = []
        # If it is a Sequential model
        if(isinstance(self.model, Sequential)):
            # Remove old layers
            for i in range(num_remove):
                removed_layers.append(self.model.layers.pop())
                removed_params.append(self.model.params.pop())

            # Insert new layers
            for layer in new_layers:
                self.model.add(layer)

        # If it is a Graph model
        else:
            raise NotImplementedError("Try using self.removeLayers(...) instead.")

        return [removed_layers, removed_params]


    def removeLayers(self, layers_names):
        """
            Removes the list of layers whose names are passed by parameter from the current network.
            Function only valid for Graph models. Use self.replaceLastLayers(...) for Sequential models.
        """
        removed_layers = []
        removed_params = []
        if(isinstance(self.model, Graph)):
            for layer in layers_names:
                removed_layers.append(self.model.nodes.pop(layer))
                self.model.namespace.remove(layer)
            detected = []
            for i, layers in enumerate(self.model.node_config):
                try:
                    pos = layers_names.index(layers['name'])
                    detected.append(i)
                    layers_names.pop(pos)
                except:
                    pass
            n_detected = len(detected)
            for i in range(n_detected):
                removed_params.append(self.model.node_config.pop(detected.pop()))

        else:
            raise NotImplementedError("Try using self.replaceLastLayers(...) instead.")

        return [removed_layers, removed_params[::-1]]


    def removeOutputs(self, outputs_names):
        """
            Removes the list of outputs whose names are passed by parameter from the current network.
            This function is only valid for Graph models.
        """
        if(isinstance(self.model, Graph)):
            new_outputs = []
            for output in self.model.output_order:
                if(output not in outputs_names):
                    new_outputs.append(output)
            self.model.output_order = new_outputs
            detected = []
            for i, outputs in enumerate(self.model.output_config):
                try:
                    pos = outputs_names.index(outputs['name'])
                    detected.append(i)
                    outputs_names.pop(pos)
                except:
                    pass
            n_detected = len(detected)
            for i in range(n_detected):
                self.model.output_config.pop(detected.pop())
            return True
        return False


    def removeInputs(self, inputs_names):
        """
            Removes the list of inputs whose names are passed by parameter from the current network.
            This function is only valid for Graph models.
        """
        if(isinstance(self.model, Graph)):
            new_inputs = []
            for input in self.model.input_order:
                if(input not in inputs_names):
                    new_inputs.append(input)
            self.model.input_order = new_inputs
            detected = []
            for i, inputs in enumerate(self.model.input_config):
                try:
                    pos = inputs_names.index(inputs['name'])
                    detected.append(i)
                    inputs_names.pop(pos)
                except:
                    pass
            n_detected = len(detected)
            for i in range(n_detected):
                inp = self.model.input_config.pop(detected.pop())
                self.model.namespace.remove(inp['name'])
            return True
        return False


    # ------------------------------------------------------- #
    #       TRAINING/TEST
    #           Methods for train and testing on the current CNN_Model
    # ------------------------------------------------------- #


    def trainNet(self, ds, parameters, out_name=None):
        """
            Trains the network on the given dataset 'ds'.
            out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable to Graph models.

            The available (optional) training parameters are the following ones:

            ####    Visualization parameters

            :param report_iter: number of iterations between each loss report
            :param iter_for_val: number of interations between each validation test
            :param num_iterations_val: number of iterations applied on the validation dataset for computing the average performance (if None then all the validation data will be tested)

            ####    Learning parameters

            :param n_epochs: number of epochs that will be applied during training
            :param batch_size: size of the batch (number of images) applied on each interation by the SGD optimization
            :param lr_decay: number of iterations passed for decreasing the learning rate
            :param lr_gamma: proportion of learning rate kept at each decrease. It can also be a set of rules defined by a list, e.g. lr_gamma = [[3000, 0.9], ..., [None, 0.8]] means 0.9 until iteration 3000, ..., 0.8 until the end.

            ####    Data processing parameters

            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
            :param data_augmentation: boolean indicating if we want to perform data augmentation (always False on validation)

            ####    Other parameters

            :param save_model: number of iterations between each model backup
        """

        # Check input parameters and recover default values if needed

        default_params = {'n_epochs': 1, 'batch_size': 50, 'lr_decay': 1, 'lr_gamma':0.1, 'maxlen':100,
                          'homogeneous_batches': False, 'epochs_for_save': 1, 'num_iterations_val': None,
                          'n_parallel_loaders': 8, 'normalize_images': False, 'mean_substraction': True,
                          'data_augmentation': True,'verbose': 1, 'eval_on_sets': ['val'],
                          'reload_epoch': 0, 'extra_callbacks': []};

        params = self.checkParameters(parameters, default_params)
        save_params = copy.copy(params)
        del save_params['extra_callbacks']
        self.training_parameters.append(save_params)

        logging.info("<<< Training model >>>")

        self.__logger = dict()
        self.__train(ds, params)

        logging.info("<<< Finished training model >>>")


    def resumeTrainNet(self, ds, parameters, out_name=None):
        """
            DEPRECATED

            Resumes the last training state of a stored model keeping also its training parameters.
            If we introduce any parameter through the argument 'parameters', it will be replaced by the old one.

            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """

        raise NotImplementedError('Deprecated')

        # Recovers the old training parameters (replacing them by the new ones if any)
        default_params = self.training_parameters[-1]
        params = self.checkParameters(parameters, default_params)
        self.training_parameters.append(copy.copy(params))

        # Recovers the last training state
        state = self.training_state

        logging.info("<<< Resuming training model >>>")

        self.__train(ds, params, state)

        logging.info("<<< Finished training CNN_Model >>>")


    def __train(self, ds, params, state=dict()):

        logging.info("Training parameters: "+ str(params))

        # initialize state
        state['samples_per_epoch'] = ds.len_train
        state['n_iterations_per_epoch'] = int(math.ceil(float(state['samples_per_epoch'])/params['batch_size']))

        # Prepare callbacks
        callbacks = []
        callback_store_model = StoreModelWeightsOnEpochEnd(self, saveModel,
                                                           params['epochs_for_save'], reload_epoch=params['reload_epoch'])
        callback_lr_reducer = LearningRateReducerWithEarlyStopping(patience=0,
                                                                   lr_decay=params['lr_decay'],
                                                                   reduce_rate=params['lr_gamma'])
        callbacks.append(callback_store_model)
        callbacks.append(callback_lr_reducer)
        callbacks += params['extra_callbacks']

        # Prepare data generators
        if params['homogeneous_batches']:
            train_gen = Homogeneous_Data_Batch_Generator('train', self, ds, state['n_iterations_per_epoch'],
                                             batch_size=params['batch_size'], maxlen=params['maxlen'],
                                             normalize_images=params['normalize_images'],
                                             data_augmentation=params['data_augmentation'],
                                             mean_substraction=params['mean_substraction']).generator()
        else:
            train_gen = Data_Batch_Generator('train', self, ds, state['n_iterations_per_epoch'],
                                             batch_size=params['batch_size'],
                                             normalize_images=params['normalize_images'],
                                             data_augmentation=params['data_augmentation'],
                                             mean_substraction=params['mean_substraction']).generator()
        # Are we going to validate on 'val' data?
        if('val' in params['eval_on_sets']):

            # Calculate how many validation interations are we going to perform per test
            n_valid_samples = ds.len_val
            if(params['num_iterations_val'] == None):
                params['num_iterations_val'] = int(math.ceil(float(n_valid_samples)/params['batch_size']))

            # prepare data generator
            val_gen = Data_Batch_Generator('val', self, ds, params['num_iterations_val'],
                                         batch_size=params['batch_size'],
                                         normalize_images=params['normalize_images'],
                                         data_augmentation=False,
                                         mean_substraction=params['mean_substraction']).generator()
        else:
            val_gen = None
            n_valid_samples = None

        # Train model
        self.model.fit_generator(train_gen,
                                 validation_data=val_gen,
                                 nb_val_samples=n_valid_samples,
                                 samples_per_epoch=state['samples_per_epoch'],
                                 nb_epoch=params['n_epochs'],
                                 max_q_size=params['n_parallel_loaders'],
                                 verbose=params['verbose'],
                                 callbacks=callbacks)


    def __train_deprecated(self, ds, params, state=dict(), out_name=None):
        """
            Main training function, which will only be called from self.trainNet(...) or self.resumeTrainNet(...)
        """
        scores_train = []
        losses_train = []
        top_scores_train = []

        logging.info("Training parameters: "+ str(params))

        # Calculate how many iterations are we going to perform
        if(not state.has_key('n_iterations_per_epoch')):
            state['n_iterations_per_epoch'] = int(math.ceil(float(ds.len_train)/params['batch_size']))
            state['count_iteration'] = 0
            state['epoch'] = 0
            state['it'] = -1
        else:
            state['count_iteration'] -= 1
            state['it'] -= 1

        # Calculate how many validation interations are we going to perform per test
        if(params['num_iterations_val'] == None):
            params['num_iterations_val'] = int(math.ceil(float(ds.len_val)/params['batch_size']))

        # Apply params['n_epochs'] for training
        for state['epoch'] in range(state['epoch'], params['n_epochs']):
            logging.info("<<< Starting epoch "+str(state['epoch']+1)+"/"+str(params['n_epochs']) +" >>>")

            # Shuffle the training samples before each epoch
            ds.shuffleTraining()

            # Initialize queue of parallel data loaders
            t_queue = []
            for t_ind in range(state['n_iterations_per_epoch']):
                t = ThreadDataLoader(retrieveXY, ds, 'train', params['batch_size'],
                                params['normalize_images'], params['mean_substraction'], params['data_augmentation'])
                if(t_ind > state['it'] and t_ind < params['n_parallel_loaders'] +state['it']+1):
                    t.start()
                t_queue.append(t)

            for state['it'] in range(state['it']+1, state['n_iterations_per_epoch']):
                state['count_iteration'] +=1

                # Recovers a pre-loaded batch of data
                time_load = time.time()*1000.0
                t = t_queue[state['it']]
                t.join()
                time_load = time.time()*1000.0-time_load
                if(params['verbose'] > 0):
                    logging.info("DEBUG: Batch loaded in %0.8s ms" % str(time_load))

                if(t.resultOK):
                    X_batch = t.X
                    Y_batch = t.Y
                else:
                    if params['verbose'] > 1:
                        logging.info("DEBUG: Exception occurred.")
                    exc_type, exc_obj, exc_trace = t.exception
                    # deal with the exception
                    print exc_type, exc_obj
                    print exc_trace
                    raise Exception('Exception occurred in ThreadLoader.')
                t_queue[state['it']] = None
                if(state['it']+params['n_parallel_loaders'] < state['n_iterations_per_epoch']):
                    if params['verbose'] > 1:
                        logging.info("DEBUG: Starting new thread loader.")
                    t = t_queue[state['it']+params['n_parallel_loaders']]
                    t.start()

                # Forward and backward passes on the current batch
                time_train = time.time()*1000.0
                if(isinstance(self.model, Sequential)):
                    [X_batch, Y_batch] = self._prepareSequentialData(X_batch, Y_batch)
                    loss = self.model.train_on_batch(X_batch, Y_batch)
                    loss = loss[0]
                    [score, top_score] = self._getSequentialAccuracy(Y_batch, self.model.predict_on_batch(X_batch)[0])
                elif(isinstance(self.model, Model)):
                    t1 = time.time()*1000.0
                    [X_batch, Y_batch] = self._prepareSequentialData(X_batch, Y_batch)
                    if params['verbose'] > 1:
                        t2 = time.time()*1000.0
                        logging.info("DEBUG: Data ready for training (%0.8s ms)." % (t2-t1))
                    loss = self.model.train_on_batch(X_batch, Y_batch)
                    if params['verbose'] > 1:
                        t3 = time.time()*1000.0
                        logging.info("DEBUG: Training forward & backward passes performed (%0.8s ms)." % (t3-t2))
                    loss = loss[0]
                    score = loss[1]
                    #[score, top_score] = self._getSequentialAccuracy(Y_batch, self.model.predict_on_batch(X_batch))
                else:
                    [data, last_output] = self._prepareGraphData(X_batch, Y_batch)
                    loss = self.model.train_on_batch(data)
                    loss = loss[0]
                    score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                    top_score = score[1]
                    score = score[0]
                    if(out_name):
                        score = score[out_name]
                        top_score = top_score[out_name]
                    else:
                        score = score[last_output]
                        top_score = top_score[last_output]
                time_train = time.time()*1000.0-time_train
                if(params['verbose'] > 0):
                    logging.info("DEBUG: Train on batch performed in %0.8s ms" % str(time_train))

                scores_train.append(float(score))
                losses_train.append(float(loss))
                top_scores_train.append(float(top_score))

                # Report train info
                if(state['count_iteration'] % params['report_iter'] == 0):
                    loss = np.mean(losses_train)
                    score = np.mean(scores_train)
                    top_score = np.mean(top_scores_train)

                    logging.info("Train - Iteration: "+ str(state['count_iteration']) + "   (" + str(state['count_iteration']*params['batch_size']) + " samples seen)")
                    logging.info("\tTrain loss: "+ str(loss))
                    logging.info("\tTrain accuracy: "+ str(score))
                    logging.info("\tTrain accuracy top-5: "+ str(top_score))

                    self.log('train', 'iteration', state['count_iteration'])
                    self.log('train', 'loss', loss)
                    self.log('train', 'accuracy', score)
                    try:
                        self.log('train', 'accuracy top-5', top_score)
                    except:
                        pass

                    scores_train = []
                    losses_train = []
                    top_scores_train = []

                # Test network on validation set
                if(state['count_iteration'] > 0 and state['count_iteration'] % params['iter_for_val'] == 0):
                    logging.info("Applying validation...")
                    scores = []
                    losses = []
                    top_scores = []

                    t_val_queue = []
                    for t_ind in range(params['num_iterations_val']):
                        t = ThreadDataLoader(retrieveXY, ds, 'val', params['batch_size'],
                                        params['normalize_images'], params['mean_substraction'], False)
                        if(t_ind < params['n_parallel_loaders']):
                            t.start()
                        t_val_queue.append(t)

                    for it_val in range(params['num_iterations_val']):

                        # Recovers a pre-loaded batch of data
                        t_val = t_val_queue[it_val]
                        t_val.join()
                        if(t_val.resultOK):
                            X_val = t_val.X
                            Y_val = t_val.Y
                        else:
                            exc_type, exc_obj, exc_trace = t.exception
                            # deal with the exception
                            print exc_type, exc_obj
                            print exc_trace
                            raise Exception('Exception occurred in ThreadLoader.')
                        t_val_queue[it_val] = None
                        if(it_val+params['n_parallel_loaders'] < params['num_iterations_val']):
                            t_val = t_val_queue[it_val+params['n_parallel_loaders']]
                            t_val.start()

                        # Forward prediction pass
                        if(isinstance(self.model, Sequential) or isinstance(self.model, Model)):
                            [X_val, Y_val] = self._prepareSequentialData(X_val, Y_val)
                            loss = self.model.test_on_batch(X_val, Y_val, accuracy=False)
                            loss = loss[0]
                            [score, top_score] = self._getSequentialAccuracy(Y_val, self.model.predict_on_batch(X_val)[0])
                        else:
                            [data, last_output] = self._prepareGraphData(X_val, Y_val)
                            loss = self.model.test_on_batch(data)
                            loss = loss[0]
                            score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                            top_score = score[1]
                            score = score[0]
                            if(out_name):
                                score = score[out_name]
                                top_score = top_score[out_name]
                            else:
                                score = score[last_output]
                                top_score = top_score[last_output]
                        losses.append(float(loss))
                        scores.append(float(score))
                        top_scores.append(float(top_score))

                    ds.resetCounters(set_name='val')
                    logging.info("Val - Iteration: "+ str(state['count_iteration']))
                    loss = np.mean(losses)
                    logging.info("\tValidation loss: "+ str(loss))
                    score = np.mean(scores)
                    logging.info("\tValidation accuracy: "+ str(score))
                    top_score = np.mean(top_scores)
                    logging.info("\tValidation accuracy top-5: "+ str(top_score))

                    self.log('val', 'iteration', state['count_iteration'])
                    self.log('val', 'loss', loss)
                    self.log('val', 'accuracy', score)
                    try:
                        self.log('val', 'accuracy top-5', top_score)
                    except:
                        pass

                    self.plot()

                # Save the model
                if(state['count_iteration'] % params['save_model'] == 0):
                    self.training_state = state
                    saveModel(self, state['count_iteration'])

                # Decrease the current learning rate
                if(state['count_iteration'] % params['lr_decay'] == 0):
                    # Check if we have a set of rules
                    if(isinstance(params['lr_gamma'], list)):
                        # Check if the current lr_gamma rule is still valid
                        if(params['lr_gamma'][0][0] == None or params['lr_gamma'][0][0] > state['count_iteration']):
                            lr_gamma = params['lr_gamma'][0][1]
                        else:
                            # Find next valid lr_gamma
                            while(params['lr_gamma'][0][0] != None and params['lr_gamma'][0][0] <= state['count_iteration']):
                                params['lr_gamma'].pop(0)
                            lr_gamma = params['lr_gamma'][0][1]
                    # Else, we have a single lr_gamma for the whole training
                    else:
                        lr_gamma = params['lr_gamma']
                    lr = self.lr * lr_gamma
                    momentum = 1-lr
                    self.setOptimizer(lr, momentum)

            self.training_state = state
            state['it'] = -1 # start again from the first iteration of the next epoch


    def testNet(self, ds, parameters, out_name=None):

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'normalize_images': False,
                          'mean_substraction': True};
        params = self.checkParameters(parameters, default_params)
        self.testing_parameters.append(copy.copy(params))

        logging.info("<<< Testing model >>>")

        # Calculate how many test interations are we going to perform
        n_samples = ds.len_test
        num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))

        # Test model
        # We won't use an Homogeneous_Batch_Generator for testing
        data_gen = Data_Batch_Generator('test', self, ds, num_iterations,
                                         batch_size=params['batch_size'],
                                         normalize_images=params['normalize_images'],
                                         data_augmentation=False,
                                         mean_substraction=params['mean_substraction']).generator()

        out = self.model.evaluate_generator(data_gen,
                                      val_samples=n_samples,
                                      max_q_size=params['n_parallel_loaders'])

        # Display metrics results
        for name, o in zip(self.model.metrics_names, out):
            logging.info('test '+name+': %0.8s' % o)

        #loss_all = out[0]
        #loss_ecoc = out[1]
        #loss_final = out[2]
        #acc_ecoc = out[3]
        #acc_final = out[4]
        #logging.info('Test loss: %0.8s' % loss_final)
        #logging.info('Test accuracy: %0.8s' % acc_final)


    def testNet_deprecated(self, ds, parameters, out_name=None):
        """
            Applies a complete round of tests using the test set in the provided Dataset instance.

            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.

            The available (optional) testing parameters are the following ones:

            :param batch_size: size of the batch (number of images) applied on each interation

            ####    Data processing parameters

            :param n_parallel_loaders: number of parallel data loaders allowed to work at the same time
            :param normalize_images: boolean indicating if we want to 0-1 normalize the image pixel values
            :param mean_substraction: boolean indicating if we want to substract the training mean
        """
        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'normalize_images': False, 'mean_substraction': True};
        params = self.checkParameters(parameters, default_params)
        self.testing_parameters.append(copy.copy(params))

        logging.info("<<< Testing model >>>")

        numIterationsTest = int(math.ceil(float(ds.len_test)/params['batch_size']))
        scores = []
        losses = []
        top_scores = []

        t_test_queue = []
        for t_ind in range(numIterationsTest):
            t = ThreadDataLoader(retrieveXY, ds, 'test', params['batch_size'],
                            params['normalize_images'], params['mean_substraction'], False)
            if(t_ind < params['n_parallel_loaders']):
                t.start()
            t_test_queue.append(t)

        for it_test in range(numIterationsTest):

            t_test = t_test_queue[it_test]
            t_test.join()
            if(t_test.resultOK):
                X_test = t_test.X
                Y_test = t_test.Y
            else:
                exc_type, exc_obj, exc_trace = t.exception
                # deal with the exception
                print exc_type, exc_obj
                print exc_trace
                raise Exception('Exception occurred in ThreadLoader.')
            t_test_queue[it_test] = None
            if(it_test+params['n_parallel_loaders'] < numIterationsTest):
                t_test = t_test_queue[it_test+params['n_parallel_loaders']]
                t_test.start()

            if(isinstance(self.model, Sequential) or isinstance(self.model, Model)):
                # (loss, score) = self.model.evaluate(X_test, Y_test, show_accuracy=True)
                [X_test, Y_test] = self._prepareSequentialData(X_test, Y_test)
                loss = self.model.test_on_batch(X_test, Y_test, accuracy=False)
                loss = loss[0]
                [score, top_score] = self._getSequentialAccuracy(Y_test, self.model.predict_on_batch(X_test)[0])
            else:
                [data, last_output] = self._prepareGraphData(X_test, Y_test)
                loss = self.model.test_on_batch(data)
                loss = loss[0]
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if(out_name):
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]
            losses.append(float(loss))
            scores.append(float(score))
            top_scores.append(float(top_score))

        ds.resetCounters(set_name='test')
        logging.info("\tTest loss: "+ str(np.mean(losses)))
        logging.info("\tTest accuracy: "+ str(np.mean(scores)))
        logging.info("\tTest accuracy top-5: "+ str(np.mean(top_scores)))



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

            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """
        n_samples = X.shape[1]
        if(isinstance(self.model, Sequential) or isinstance(self.model, Model)):
            [X, Y] = self._prepareSequentialData(X, Y)
            loss = self.model.test_on_batch(X, Y, accuracy=False)
            loss = loss[0]
            if(accuracy):
                [score, top_score] = self._getSequentialAccuracy(Y, self.model.predict_on_batch(X)[0])
                return (loss, score, top_score, n_samples)
            return (loss, n_samples)
        else:
            [data, last_output] = self._prepareGraphData(X, Y)
            loss = self.model.test_on_batch(data)
            loss = loss[0]
            if(accuracy):
                score = self._getGraphAccuracy(data, self.model.predict_on_batch(data))
                top_score = score[1]
                score = score[0]
                if(out_name):
                    score = score[out_name]
                    top_score = top_score[out_name]
                else:
                    score = score[last_output]
                    top_score = top_score[last_output]
                return (loss, score, top_score, n_samples)
            return (loss, n_samples)


    def predict_cond(self, X, states_below, params, ii):

        p = []
        for state_below in states_below:
            X['state_below'] = state_below.reshape(1,-1)
            p.append(np.array(self.model.predict_on_batch(X))[:, ii, :])  # Get probs of all words in the current timestep
        p = np.asarray(p)
        return p[:, 0, :]

    def beam_search(self, X, params):

        k = params['beam_size']
        sample = []
        sample_score = []

        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores  = np.zeros(live_k).astype('float32')
        state_below = np.asarray([np.zeros(params['maxlen'])] * live_k)
        for ii in xrange(params['maxlen']):
            # for every possible live sample calc prob for every possible label
            probs = self.predict_cond(X, state_below, params, ii)
            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(hyp_scores)[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            voc_size = probs.shape[1]
            trans_indices = ranks_flat / voc_size # index of row
            word_indices = ranks_flat % voc_size # index of col
            costs = cand_flat[ranks_flat]
            new_hyp_samples = []
            new_hyp_scores = np.zeros(k-dead_k).astype('float32')
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            state_below = np.asarray(hyp_samples, dtype='int64')
            state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                     np.zeros((state_below.shape[0], max(params['maxlen'] - state_below.shape[1]-1, 0)),
                                              dtype='int64')))
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
        return sample, sample_score

    def BeamSearchNet(self, ds, parameters):
        '''
            Returns the predictions of the net on the dataset splits chosen. The valid parameters are:

            :param batch_size: size of the batch
            :param n_parallel_loaders: number of parallel data batch loaders
            :param normalize_images: apply data normalization on images/features or not (only if using images/features as input)
            :param mean_substraction: apply mean data normalization on images or not (only if using images as input)
            :param predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']

            :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        '''

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'beam_size': 5,
                          'normalize_images': False, 'mean_substraction': True,
                          'predict_on_sets': ['val'], 'maxlen': 20, 'model_inputs': ['source_text', 'state_below']}
        params = self.checkParameters(parameters, default_params)

        predictions = dict()
        for s in params['predict_on_sets']:

            logging.info("<<< Predicting outputs of "+s+" set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'

            # Calculate how many interations are we going to perform
            n_samples = eval("ds.len_"+s)
            num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))

            # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here

            data_gen = Data_Batch_Generator(s, self, ds, num_iterations,
                                     batch_size=params['batch_size'],
                                     normalize_images=params['normalize_images'],
                                     data_augmentation=False,
                                     mean_substraction=params['mean_substraction'],
                                     predict=True).generator()
            out = []
            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            for j in range(num_iterations):
                data = data_gen.next()
                X = dict()
                for input_id in params['model_inputs']:
                    X[input_id] = data[input_id]
                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()
                    for input_id in params['model_inputs']:
                        x[input_id] = X[input_id][i].reshape(1,-1)
                    samples, scores = self.beam_search(x, params)
                    out.append(samples[0])
                    total_cost += scores[0]
                    eta = (n_samples - sampled) *  (time.time() - start_time) / sampled

            sys.stdout.write("\n")
            sys.stdout.flush()
            logging.info('\nCost of the translations: %f'%scores[0])
            predictions[s] = np.asarray(out)
        return predictions

    def predictNet(self, ds, parameters, out_name=None):
        '''
            Returns the predictions of the net on the dataset splits chosen. The valid parameters are:

            :param batch_size: size of the batch
            :param n_parallel_loaders: number of parallel data batch loaders
            :param normalize_images: apply data normalization on images/features or not (only if using images/features as input)
            :param mean_substraction: apply mean data normalization on images or not (only if using images as input)
            :param predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']

            :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        '''

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8,
                          'normalize_images': False, 'mean_substraction': True,
                          'predict_on_sets': ['val']}
        params = self.checkParameters(parameters, default_params)

        predictions = dict()
        for s in params['predict_on_sets']:

            logging.info("<<< Predicting outputs of "+s+" set >>>")

            # Calculate how many interations are we going to perform
            n_samples = eval("ds.len_"+s)
            num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))
            # Prepare data generator

            data_gen = Data_Batch_Generator(s, self, ds, num_iterations,
                                     batch_size=params['batch_size'],
                                     normalize_images=params['normalize_images'],
                                     data_augmentation=False,
                                     mean_substraction=params['mean_substraction'],
                                     predict=True).generator()

            # Predict on model
            out = self.model.predict_generator(data_gen,
                                                val_samples=n_samples,
                                                max_q_size=params['n_parallel_loaders'])

            predictions[s] = out
        return predictions


    def predictOnBatch(self, X, in_name=None, out_name=None, expand=False):
        """
            Applies a forward pass and returns the predicted values.
        """
        # Get desired input
        if(in_name):
            X = copy.copy(X[in_name])

        # Expand input dimensions to 4
        if(expand):
            while(len(X.shape) < 4):
                X = np.expand_dims(X, axis=1)

        '''
        # Prepare data if Graph model
        if(isinstance(self.model, Graph)):
            [X, last_out] = self._prepareGraphData(X)
        elif(isinstance(self.model, Sequential) or isinstance(self.model, Model)):
            [X, _] = self._prepareSequentialData(X)
        '''
        X = self.prepareData(X, None)[0]

        # Apply forward pass for prediction
        predictions = self.model.predict_on_batch(X)

        # Select output if indicated
        if(isinstance(self.model, Graph) or isinstance(self.model, Model)): # Graph
            if(out_name):
                predictions = predictions[out_name]
        elif(isinstance(self.model, Sequential)): # Sequential
            predictions = predictions[0]

        return predictions


    def prepareData(self, X_batch, Y_batch=None):
        if(isinstance(self.model, Sequential)):
            data = self._prepareSequentialData(X_batch, Y_batch)
        elif(isinstance(self.model, Model)):
            data = self._prepareModelData(X_batch, Y_batch)
        elif(isinstance(self.model, Graph)):
            [data, Y_batch] = self._prepareGraphData(X_batch, Y_batch)
        else:
            raise NotImplementedError
        return data


    def _prepareSequentialData(self, X, Y=None):

        # Format input data
        if(len(self.inputsMapping.keys()) == 1): # single input
            X = X[self.inputsMapping[0]]
        else:
            X_new = [0 for i in range(len(self.inputsMapping.keys()))] # multiple inputs
            for in_model, in_ds in self.inputsMapping.iteritems():
                X_new[in_model] = X[in_ds]
            X = X_new

        # Format output data (only one output possible for Sequential models)
        if(Y is not None):
            if(len(self.outputsMapping.keys()) == 1): # single output
                Y = Y[self.outputsMapping[0]]
            else:
                Y_new = [0 for i in range(len(self.outputsMapping.keys()))] # multiple outputs
                for out_model, out_ds in self.outputsMapping.iteritems():
                    Y_new[out_model] = Y[out_ds]
                Y = Y_new

        return [X, Y]


    def _prepareModelData(self, X, Y=None):
        X_new = dict()
        Y_new = dict()

        # Format input data
        for in_model, in_ds in self.inputsMapping.iteritems():
            X_new[in_model] = X[in_ds]

        # Format output data
        if(Y is not None):
            for out_model, out_ds in self.outputsMapping.iteritems():
                Y_new[out_model] = Y[out_ds]

        return [X_new, Y_new]


    def _prepareGraphData(self, X, Y=None):

        data = dict()
        last_out = self.acc_output

        # Format input data
        for in_model, in_ds in self.inputsMapping.iteritems():
            data[in_model] = X[in_ds]

        # Format output data
        for out_model, out_ds in self.outputsMapping.iteritems():
            if(Y is None):
                data[out_model] = None
            else:
                data[out_model] = Y[out_ds]

        # Currently all samples are assigned to all inputs and all labels to all outputs
        #data = dict()
        #last_out = ''
        #for input in self.model.input_order:
        #    data[input] = X
        #for output in self.model.output_order:
        #    data[output] = Y
        #    last_out = output

        return [data, last_out]


    def _getGraphAccuracy(self, data, prediction, topN=5):
        """
            Calculates the accuracy obtained from a set of samples on a Graph model.
        """

        accuracies = dict()
        top_accuracies = dict()
        for key, val in prediction.iteritems():
            pred = np_utils.categorical_probas_to_classes(val)
            top_pred = np.argsort(val,axis=1)[:,::-1][:,:np.min([topN, val.shape[1]])]
            GT = np_utils.categorical_probas_to_classes(data[key])

            # Top1 accuracy
            correct = [1 if pred[i]==GT[i] else 0 for i in range(len(pred))]
            accuracies[key] = float(np.sum(correct)) / float(len(correct))

            # TopN accuracy
            top_correct = [1 if GT[i] in top_pred[i,:] else 0 for i in range(top_pred.shape[0])]
            top_accuracies[key] = float(np.sum(top_correct)) / float(len(top_correct))

        return [accuracies, top_accuracies]


    def _getSequentialAccuracy(self, GT, pred, topN=5):
        """
            Calculates the topN accuracy obtained from a set of samples on a Sequential model.
        """
        top_pred = np.argsort(pred,axis=1)[:,::-1][:,:np.min([topN, pred.shape[1]])]
        pred = np_utils.categorical_probas_to_classes(pred)
        GT = np_utils.categorical_probas_to_classes(GT)

        # Top1 accuracy
        correct = [1 if pred[i]==GT[i] else 0 for i in range(len(pred))]
        accuracies = float(np.sum(correct)) / float(len(correct))

        # TopN accuracy
        top_correct = [1 if GT[i] in top_pred[i,:] else 0 for i in range(top_pred.shape[0])]
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

        #if(isinstance(self.model, Model)):
        print_summary(self.model.layers)
        return ''


        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        # Print layers structure
        obj_str += "\n::: Layers structure:\n\n"
        obj_str += 'MODEL TYPE: ' + self.model.__class__.__name__ +'\n'
        if(isinstance(self.model, Sequential)):
            obj_str += "INPUT: "+ str(tuple(self.model.layers[0].input_shape)) +"\n"
            for i, layer in enumerate(self.model.layers):
                obj_str += str(layer.name) + ' '+ str(layer.output_shape) +'\n'
            obj_str += "OUTPUT: "+ str(self.model.layers[-1].output_shape) +"\n"
        else:
            for i, inputs in enumerate(self.model.input_config):
                obj_str += "INPUT (" +str(i)+ "): "+ str(inputs['name']) + ' '+ str(tuple(inputs['input_shape'])) +"\n"
            for node in self.model.node_config:
                obj_str += str(node['name']) + ', in ['+ str(node['input']) +']' +', out_shape: ' +str(self.model.nodes[node['name']].output_shape) + '\n'
            for i, outputs in enumerate(self.model.output_config):
                obj_str += "OUTPUT (" +str(i)+ "): "+ str(outputs['name']) + ', in ['+ str(outputs['input']) +']' +', out_shape: ' +str(self.model.outputs[outputs['name']].output_shape) +"\n"

        obj_str += '-----------------------------------------------------------------------------------\n'

        print_summary(self.model.layers)

        return obj_str


    def log(self, mode, data_type, value):
        """
            Stores the train and val information for plotting the training progress.

            :param mode: 'train', or 'val'
            :param data_type: 'iteration', 'loss' or 'accuracy'
            :param value: numerical value taken by the data_type
        """
        if(mode not in self.__modes):
            raise Exception('The provided mode "'+ mode +'" is not valid.')
        if(data_type not in self.__data_types):
            raise Exception('The provided data_type "'+ data_type +'" is not valid.')

        if(mode not in self.__logger):
            self.__logger[mode] = dict()
        if(data_type not in self.__logger[mode]):
            self.__logger[mode][data_type] = list()
        self.__logger[mode][data_type].append(value)


    def plot(self):
        """
            Plots the training progress information.
        """
        colours = {'train_accuracy_top-5': 'y','train_accuracy': 'y', 'train_loss': 'k',
                   'val_accuracy_top-5': 'g', 'val_accuracy': 'g', 'val_loss': 'b',
                   'max_accuracy': 'r'}

        plt.figure(1)

        all_iterations = []
        # Plot train information
        if('train' in self.__logger):
            if('iteration' not in self.__logger['train']):
                raise Exception("The training 'iteration' must be logged into the model for plotting.")
            if('accuracy' not in self.__logger['train'] and 'loss' not in self.__logger['train']):
                raise Exception("Either train 'accuracy' and/or 'loss' must be logged into the model for plotting.")

            iterations = self.__logger['train']['iteration']
            all_iterations = all_iterations+iterations

            # Loss
            if('loss' in self.__logger['train']):
                loss = self.__logger['train']['loss']
                plt.subplot(211)
                #plt.plot(iterations, loss, colours['train_loss']+'o')
                plt.plot(iterations, loss, colours['train_loss'])
                plt.subplot(212)
                plt.plot(iterations, loss, colours['train_loss'])

            # Accuracy
            if('accuracy' in self.__logger['train']):
                accuracy = self.__logger['train']['accuracy']
                plt.subplot(211)
                plt.plot(iterations, accuracy, colours['train_accuracy']+'o')
                plt.plot(iterations, accuracy, colours['train_accuracy'])
                plt.subplot(212)
                plt.plot(iterations, accuracy, colours['train_accuracy']+'o')
                plt.plot(iterations, accuracy, colours['train_accuracy'])

            # Accuracy Top-5
            if('accuracy top-5' in self.__logger['train']):
                accuracy = self.__logger['train']['accuracy top-5']
                plt.subplot(211)
                plt.plot(iterations, accuracy, colours['train_accuracy_top-5']+'.')
                plt.plot(iterations, accuracy, colours['train_accuracy_top-5'])
                plt.subplot(212)
                plt.plot(iterations, accuracy, colours['train_accuracy_top-5']+'.')
                plt.plot(iterations, accuracy, colours['train_accuracy_top-5'])


        # Plot val information
        if('val' in self.__logger):
            if('iteration' not in self.__logger['val']):
                raise Exception("The validation 'iteration' must be logged into the model for plotting.")
            if('accuracy' not in self.__logger['val'] and 'loss' not in self.__logger['train']):
                raise Exception("Either val 'accuracy' and/or 'loss' must be logged into the model for plotting.")

            iterations = self.__logger['val']['iteration']
            all_iterations = all_iterations+iterations

            # Loss
            if('loss' in self.__logger['val']):
                loss = self.__logger['val']['loss']
                plt.subplot(211)
                #plt.plot(iterations, loss, colours['val_loss']+'o')
                plt.plot(iterations, loss, colours['val_loss'])
                plt.subplot(212)
                plt.plot(iterations, loss, colours['val_loss'])

            # Accuracy
            if('accuracy' in self.__logger['val']):
                accuracy = self.__logger['val']['accuracy']
                plt.subplot(211)
                plt.plot(iterations, accuracy, colours['val_accuracy']+'o')
                plt.plot(iterations, accuracy, colours['val_accuracy'])
                plt.subplot(212)
                plt.plot(iterations, accuracy, colours['val_accuracy']+'o')
                plt.plot(iterations, accuracy, colours['val_accuracy'])

            # Accuracy Top-5
            if('accuracy top-5' in self.__logger['val']):
                accuracy = self.__logger['val']['accuracy top-5']
                plt.subplot(211)
                plt.plot(iterations, accuracy, colours['val_accuracy_top-5']+'.')
                plt.plot(iterations, accuracy, colours['val_accuracy_top-5'])
                plt.subplot(212)
                plt.plot(iterations, accuracy, colours['val_accuracy_top-5']+'.')
                plt.plot(iterations, accuracy, colours['val_accuracy_top-5'])

        # Plot max accuracy
        max_iter = np.max(all_iterations+[0])
        plt.subplot(211)
        plt.plot([0, max_iter], [1, 1], colours['max_accuracy']+'-')
        plt.subplot(212)
        plt.plot([0, max_iter], [1, 1], colours['max_accuracy']+'-')
        plt.axis([0, max_iter, 0, 1]) # limit height to 1

        # Fill labels
        #plt.ylabel('Loss/Accuracy')
        plt.xlabel('Iteration')
        plt.subplot(211)
        plt.title('Training progress')

        # Create plots dir
        if(not os.path.isdir(self.plot_path)):
            os.makedirs(self.plot_path)

        # Save figure
        plot_file = self.plot_path+'/iter_'+str(max_iter)+'.jpg'
        plt.savefig(plot_file)
        if(not self.silence):
            logging.info("Progress plot saved in " + plot_file)

        # Close plot window
        plt.close()


    # ------------------------------------------------------- #
    #       MODELS
    #           Available definitions of CNN models (see basic_model as an example)
    #           All the models must include the following parameters:
    #               nOutput, input
    # ------------------------------------------------------- #


    def basic_model(self, nOutput, input):
        """
            Builds a basic CNN model.
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, 3, 3, border_mode='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, border_mode='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, border_mode='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
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



    def One_vs_One(self, nOutput, input):
        """
            Builds a simple One_vs_One network with 3 convolutional layers (useful for ECOC models).
        """
        # default lr=0.1, momentum=0.
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1),input_shape=input_shape)) # default input_shape=(3,224,224)
        self.model.add(Convolution2D(32, 1, 1, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(16, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(8, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(1,1)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax')) # default nOutput=1000



    def VGG_16(self, nOutput, input):
        """
            Builds a VGG model with 16 layers.
        """
        # default lr=0.1, momentum=0.
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1),input_shape=input_shape)) # default input_shape=(3,224,224)
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax')) # default nOutput=1000


    def VGG_16_PReLU(self, nOutput, input):
        """
            Builds a VGG model with 16 layers and with PReLU activations.
        """

        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1),input_shape=input_shape)) # default input_shape=(3,224,224)
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, 3, 3))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, 3, 3))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, 3, 3))
        self.model.add(PReLU())
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(PReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(PReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax')) # default nOutput=1000



    def VGG_16_FunctionalAPI(self, nOutput, input):
        """
            16-layered VGG model implemented in Keras' Functional API
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        vis_input = Input(shape=input_shape, name="vis_input")

        x = ZeroPadding2D((1,1))                           (vis_input)
        x = Convolution2D(64, 3, 3, activation='relu')     (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(64, 3, 3, activation='relu')     (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(128, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(128, 3, 3, activation='relu')    (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(256, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(256, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(256, 3, 3, activation='relu')    (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = MaxPooling2D((2,2), strides=(2,2))             (x)

        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = ZeroPadding2D((1,1))                           (x)
        x = Convolution2D(512, 3, 3, activation='relu')    (x)
        x = MaxPooling2D((2,2), strides=(2,2),
                         name='last_max_pool')             (x)

        x = Flatten()                                      (x)
        x = Dense(4096, activation='relu')                 (x)
        x = Dropout(0.5)                                   (x)
        x = Dense(4096, activation='relu')                 (x)
        x = Dropout(0.5, name='last_dropout')              (x)
        x = Dense(nOutput, activation='softmax', name='output')   (x) # nOutput=1000 by default

        self.model = Model(input=vis_input, output=x)

    ########################################
    # GoogLeNet implementation from http://dandxy89.github.io/ImageModels/googlenet/
    ########################################

    def inception_module(self, x, params, dim_ordering, concat_axis,
                         subsample=(1, 1), activation='relu',
                         border_mode='same', weight_decay=None):

        # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
        # file-googlenet_neon-py

        (branch1, branch2, branch3, branch4) = params

        if weight_decay:
            W_regularizer = regularizers.l2(weight_decay)
            b_regularizer = regularizers.l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        pathway1 = Convolution2D(branch1[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)

        pathway2 = Convolution2D(branch2[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)
        pathway2 = Convolution2D(branch2[1], 3, 3,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway2)

        pathway3 = Convolution2D(branch3[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)
        pathway3 = Convolution2D(branch3[1], 5, 5,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway3)

        pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=dim_ordering)(x)
        pathway4 = Convolution2D(branch4[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway4)

        return merge([pathway1, pathway2, pathway3, pathway4],
                     mode='concat', concat_axis=concat_axis)


    def conv_layer(self, x, nb_filter, nb_row, nb_col, dim_ordering,
                   subsample=(1, 1), activation='relu',
                   border_mode='same', weight_decay=None, padding=None):

        if weight_decay:
            W_regularizer = regularizers.l2(weight_decay)
            b_regularizer = regularizers.l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        x = Convolution2D(nb_filter, nb_row, nb_col,
                          subsample=subsample,
                          activation=activation,
                          border_mode=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          bias=False,
                          dim_ordering=dim_ordering)(x)

        if padding:
            for i in range(padding):
                x = ZeroPadding2D(padding=(1, 1), dim_ordering=dim_ordering)(x)

        return x


    def GoogLeNet_FunctionalAPI(self, nOutput, input):

        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        # Define image input layer
        img_input = Input(shape=input_shape, name='input_data')
        CONCAT_AXIS = 1
        NB_CLASS = nOutput         # number of classes (default 1000)
        DROPOUT = 0.4
        WEIGHT_DECAY = 0.0005   # L2 regularization factor
        USE_BN = True           # whether to use batch normalization
        # Theano - 'th' (channels, width, height)
        # Tensorflow - 'tf' (width, height, channels)
        DIM_ORDERING = 'th'
        pool_name = 'last_max_pool' # name of the last max-pooling layer

        x = self.conv_layer(img_input, nb_col=7, nb_filter=64, subsample=(2,2),
                       nb_row=7, dim_ordering=DIM_ORDERING, padding=1)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.conv_layer(x, nb_col=1, nb_filter=64,
                       nb_row=1, dim_ordering=DIM_ORDERING)
        x = self.conv_layer(x, nb_col=3, nb_filter=192,
                       nb_row=3, dim_ordering=DIM_ORDERING, padding=1)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                             dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                             dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)

        x = ZeroPadding2D(padding=(2, 2), dim_ordering=DIM_ORDERING)(x)
        x = MaxPooling2D(strides=(2, 2), pool_size=(3, 3), dim_ordering=DIM_ORDERING)(x)

        x = self.inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                             dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        # AUX 1 - Branch HERE
        x = self.inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                             dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                             dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
        x = self.inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
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
        #x = Dense(output_dim=NB_CLASS,
        #          activation='linear')(x)
        x = Dense(output_dim=NB_CLASS,
                  activation='softmax', name='output')(x)


        self.model = Model(input=img_input, output=[x])


    ########################################

    def Identity_Layer(self, nOutput, input):
        """
            Builds an dummy Identity_Layer, which should give as output the same as the input.
            Only used for passing the output from a previous stage to the next (see Staged_Network).
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Graph()
        # Input
        self.model.add_input(name='input', input_shape=input_shape)
        # Output
        self.model.add_output(name='output', input='input')


    def Union_Layer(self, nOutput, input):
        """
            Network with just a dropout and a softmax layers which is intended to serve as the final layer for an ECOC model
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nOutput, activation='softmax'))


    def One_vs_One_Inception(self, nOutput=2, input=[224,224,3]):
        """
            Builds a simple One_vs_One_Inception network with 2 inception layers (useful for ECOC models).
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Graph()
        # Input
        self.model.add_input(name='input', input_shape=input_shape)
        # Inception Ea
        out_Ea = self.__addInception('inceptionEa', 'input', 4, 2, 8, 2, 2, 2)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb', out_Ea, 2, 2, 4, 2, 1, 1)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1,1)), name='ave_pool/ECOC', input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(), name='loss_OnevsOne/classifier_flatten', input='ave_pool/ECOC')
        self.model.add_node(Dropout(0.5), name='loss_OnevsOne/drop', input='loss_OnevsOne/classifier_flatten')
        self.model.add_node(Dense(nOutput, activation='softmax'), name='loss_OnevsOne', input='loss_OnevsOne/drop')
        # Output
        self.model.add_output(name='loss_OnevsOne/output', input='loss_OnevsOne')
    
    
    def add_One_vs_One_Inception(self, input, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
            Builds a simple One_vs_One_Inception network with 2 inception layers on the top of the current model (useful for ECOC_loss models).
        """

        # Inception Ea
        out_Ea = self.__addInception('inceptionEa_'+str(id_branch), input, 4, 2, 8, 2, 2, 2)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb_'+str(id_branch), out_Ea, 2, 2, 4, 2, 1, 1)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1,1)), name='ave_pool/ECOC_'+str(id_branch), input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(),
                            name='fc_OnevsOne_'+str(id_branch)+'/flatten', input='ave_pool/ECOC_'+str(id_branch))
        self.model.add_node(Dropout(0.5),
                            name='fc_OnevsOne_'+str(id_branch)+'/drop', input='fc_OnevsOne_'+str(id_branch)+'/flatten')
        output_name = 'fc_OnevsOne_'+str(id_branch)
        self.model.add_node(Dense(nOutput, activation=activation), 
                            name=output_name, input='fc_OnevsOne_'+str(id_branch)+'/drop')

        return output_name
        
        
    def add_One_vs_One_Inception_Functional(self, input, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
            Builds a simple One_vs_One_Inception network with 2 inception layers on the top of the current model (useful for ECOC_loss models).
        """

        in_node = self.model.get_layer(input).output

        # Inception Ea
        [out_Ea, out_Ea_name] = self.__addInception_Functional('inceptionEa_'+str(id_branch), in_node, 4, 2, 8, 2, 2, 2)
        # Inception Eb
        [out_Eb, out_Eb_name] = self.__addInception_Functional('inceptionEb_'+str(id_branch), out_Ea, 2, 2, 4, 2, 1, 1)
        # Average Pooling    pool_size=(7,7)
        x = AveragePooling2D(pool_size=input_shape, strides=(1,1), name='ave_pool/ECOC_'+str(id_branch)) (out_Eb)

        # Softmax
        output_name = 'fc_OnevsOne_'+str(id_branch)
        x = Flatten(name='fc_OnevsOne_'+str(id_branch)+'/flatten')                (x)
        x = Dropout(0.5, name='fc_OnevsOne_'+str(id_branch)+'/drop')              (x)
        out_node = Dense(nOutput, activation=activation, name=output_name)         (x)
        
        return out_node
    
    
    
    def add_One_vs_One_3x3_Functional(self, input, input_shape, id_branch, nkernels, nOutput=2, activation='softmax'):

        # 3x3 convolution
        out_3x3 = Convolution2D(nkernels, 3, 3, name='3x3/ecoc_'+str(id_branch), activation='relu')            (input)

        # Average Pooling    pool_size=(7,7)
        x = AveragePooling2D(pool_size=input_shape, strides=(1,1), name='ave_pool/ecoc_'+str(id_branch)) (out_3x3)

        # Softmax
        output_name = 'fc_OnevsOne_'+str(id_branch)+'/out'
        x = Flatten(name='fc_OnevsOne_'+str(id_branch)+'/flatten')                (x)
        x = Dropout(0.5, name='fc_OnevsOne_'+str(id_branch)+'/drop')              (x)
        out_node = Dense(nOutput, activation=activation, name=output_name)         (x)
        
        return out_node
    
    
    def add_One_vs_One_3x3_double_Functional(self, input, input_shape, id_branch, nOutput=2, activation='softmax'):

        # 3x3 convolution
        out_3x3 = Convolution2D(64, 3, 3, name='3x3_1/ecoc_'+str(id_branch), activation='relu')          (input)

        # Max Pooling
        x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name='max_pool/ecoc_'+str(id_branch))         (out_3x3)

        # 3x3 convolution
        x = Convolution2D(32, 3, 3, name='3x3_2/ecoc_'+str(id_branch), activation='relu')                (x)

        # Softmax
        output_name = 'fc_OnevsOne_'+str(id_branch)+'/out'
        x = Flatten(name='fc_OnevsOne_'+str(id_branch)+'/flatten')                (x)
        x = Dropout(0.5, name='fc_OnevsOne_'+str(id_branch)+'/drop')              (x)
        out_node = Dense(nOutput, activation=activation, name=output_name)         (x)
        
        return out_node



    def One_vs_One_Inception_v2(self, nOutput=2, input=[224,224,3]):
        """
            Builds a simple One_vs_One_Inception_v2 network with 2 inception layers (useful for ECOC models).
        """
        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Graph()
        # Input
        self.model.add_input(name='input', input_shape=input_shape)
        # Inception Ea
        out_Ea = self.__addInception('inceptionEa', 'input', 16, 8, 32, 8, 8, 8)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb', out_Ea, 8, 8, 16, 8, 4, 4)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1,1)), name='ave_pool/ECOC', input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(), name='loss_OnevsOne/classifier_flatten', input='ave_pool/ECOC')
        self.model.add_node(Dropout(0.5), name='loss_OnevsOne/drop', input='loss_OnevsOne/classifier_flatten')
        self.model.add_node(Dense(nOutput, activation='softmax'), name='loss_OnevsOne', input='loss_OnevsOne/drop')
        # Output
        self.model.add_output(name='loss_OnevsOne/output', input='loss_OnevsOne')
    
    
    
    def add_One_vs_One_Inception_v2(self, input, input_shape, id_branch, nOutput=2, activation='softmax'):
        """
            Builds a simple One_vs_One_Inception_v2 network with 2 inception layers on the top of the current model (useful for ECOC_loss models).
        """

        # Inception Ea
        out_Ea = self.__addInception('inceptionEa_'+str(id_branch), input, 16, 8, 32, 8, 8, 8)
        # Inception Eb
        out_Eb = self.__addInception('inceptionEb_'+str(id_branch), out_Ea, 8, 8, 16, 8, 4, 4)
        # Average Pooling    pool_size=(7,7)
        self.model.add_node(AveragePooling2D(pool_size=input_shape[1:], strides=(1,1)), name='ave_pool/ECOC_'+str(id_branch), input=out_Eb)
        # Softmax
        self.model.add_node(Flatten(),
                            name='fc_OnevsOne_'+str(id_branch)+'/flatten', input='ave_pool/ECOC_'+str(id_branch))
        self.model.add_node(Dropout(0.5),
                            name='fc_OnevsOne_'+str(id_branch)+'/drop', input='fc_OnevsOne_'+str(id_branch)+'/flatten')
        output_name = 'fc_OnevsOne_'+str(id_branch)
        self.model.add_node(Dense(nOutput, activation=activation), 
                            name=output_name, input='fc_OnevsOne_'+str(id_branch)+'/drop')

        return output_name



    def __addInception(self, id, input_layer, kernels_1x1, kernels_3x3_reduce, kernels_3x3, kernels_5x5_reduce, kernels_5x5, kernels_pool_projection):
        """
            Adds an inception module to the model.

            :param id: string identifier of the inception layer
            :param input_layer: identifier of the layer that will serve as an input to the built inception module
            :param kernels_1x1: number of kernels of size 1x1                                      (1st branch)
            :param kernels_3x3_reduce: number of kernels of size 1x1 before the 3x3 layer          (2nd branch)
            :param kernels_3x3: number of kernels of size 3x3                                      (2nd branch)
            :param kernels_5x5_reduce: number of kernels of size 1x1 before the 5x5 layer          (3rd branch)
            :param kernels_5x5: number of kernels of size 5x5                                      (3rd branch)
            :param kernels_pool_projection: number of kernels of size 1x1 after the 3x3 pooling    (4th branch)
        """
        # Branch 1
        self.model.add_node(Convolution2D(kernels_1x1, 1, 1), name=id+'/1x1', input=input_layer)
        self.model.add_node(Activation('relu'), name=id+'/relu_1x1', input=id+'/1x1')

        # Branch 2
        self.model.add_node(Convolution2D(kernels_3x3_reduce, 1, 1), name=id+'/3x3_reduce', input=input_layer)
        self.model.add_node(Activation('relu'), name=id+'/relu_3x3_reduce', input=id+'/3x3_reduce')
        self.model.add_node(ZeroPadding2D((1,1)), name=id+'/3x3_zeropadding', input=id+'/relu_3x3_reduce')
        self.model.add_node(Convolution2D(kernels_3x3, 3, 3), name=id+'/3x3', input=id+'/3x3_zeropadding')
        self.model.add_node(Activation('relu'), name=id+'/relu_3x3', input=id+'/3x3')

        # Branch 3
        self.model.add_node(Convolution2D(kernels_5x5_reduce, 1, 1), name=id+'/5x5_reduce', input=input_layer)
        self.model.add_node(Activation('relu'), name=id+'/relu_5x5_reduce', input=id+'/5x5_reduce')
        self.model.add_node(ZeroPadding2D((2,2)), name=id+'/5x5_zeropadding', input=id+'/relu_5x5_reduce')
        self.model.add_node(Convolution2D(kernels_5x5, 5, 5), name=id+'/5x5', input=id+'/5x5_zeropadding')
        self.model.add_node(Activation('relu'), name=id+'/relu_5x5', input=id+'/5x5')

        # Branch 4
        self.model.add_node(ZeroPadding2D((1,1)), name=id+'/pool_zeropadding', input=input_layer)
        self.model.add_node(MaxPooling2D((3,3), strides=(1,1)), name=id+'/pool', input=id+'/pool_zeropadding')
        self.model.add_node(Convolution2D(kernels_pool_projection, 1, 1), name=id+'/pool_proj', input=id+'/pool')
        self.model.add_node(Activation('relu'), name=id+'/relu_pool_proj', input=id+'/pool_proj')

        # Concat
        inputs_list = [id+'/relu_1x1', id+'/relu_3x3', id+'/relu_5x5', id+'/relu_pool_proj']
        out_name = id+'/concat'
        self.model.add_node(Activation('linear'), name=out_name, inputs=inputs_list, concat_axis=1)

        return out_name


    def __addInception_Functional(self, id, input_layer, kernels_1x1, kernels_3x3_reduce, kernels_3x3, kernels_5x5_reduce, kernels_5x5, kernels_pool_projection):
        """
            Adds an inception module to the model.

            :param id: string identifier of the inception layer
            :param input_layer: identifier of the layer that will serve as an input to the built inception module
            :param kernels_1x1: number of kernels of size 1x1                                      (1st branch)
            :param kernels_3x3_reduce: number of kernels of size 1x1 before the 3x3 layer          (2nd branch)
            :param kernels_3x3: number of kernels of size 3x3                                      (2nd branch)
            :param kernels_5x5_reduce: number of kernels of size 1x1 before the 5x5 layer          (3rd branch)
            :param kernels_5x5: number of kernels of size 5x5                                      (3rd branch)
            :param kernels_pool_projection: number of kernels of size 1x1 after the 3x3 pooling    (4th branch)
        """
        # Branch 1
        x_b1 = Convolution2D(kernels_1x1, 1, 1, name=id+'/1x1', activation='relu')                   (input_layer)

        # Branch 2
        x_b2 = Convolution2D(kernels_3x3_reduce, 1, 1, name=id+'/3x3_reduce', activation='relu')     (input_layer)
        x_b2 = ZeroPadding2D((1,1), name=id+'/3x3_zeropadding')                                      (x_b2)
        x_b2 = Convolution2D(kernels_3x3, 3, 3, name=id+'/3x3', activation='relu')                   (x_b2)

        # Branch 3
        x_b3 = Convolution2D(kernels_5x5_reduce, 1, 1, name=id+'/5x5_reduce', activation='relu')     (input_layer)
        x_b3 = ZeroPadding2D((2,2), name=id+'/5x5_zeropadding')                                      (x_b3)
        x_b3 = Convolution2D(kernels_5x5, 5, 5, name=id+'/5x5', activation='relu')                   (x_b3)

        # Branch 4
        x_b4 = ZeroPadding2D((1,1), name=id+'/pool_zeropadding')                                     (input_layer)
        x_b4 = MaxPooling2D((3,3), strides=(1,1), name=id+'/pool')                                   (x_b4)
        x_b4 = Convolution2D(kernels_pool_projection, 1, 1, name=id+'/pool_proj', activation='relu') (x_b4)


        # Concat
        out_name = id+'/concat'
        out_node = merge([x_b1, x_b2, x_b3, x_b4], mode='concat', concat_axis=1, name=out_name)

        return [out_node, out_name]
    
    
    def add_One_vs_One_Merge(self, inputs_list, nOutput, activation='softmax'):
        
        self.model.add_node(Flatten(), name='ecoc_loss', inputs=inputs_list, merge_mode='concat') # join outputs from OneVsOne classifers
        self.model.add_node(Dropout(0.5), name='final_loss/drop', input='ecoc_loss')
        self.model.add_node(Dense(nOutput, activation=activation), name='final_loss', input='final_loss/drop') # apply final joint prediction
        
        # Outputs
        self.model.add_output(name='ecoc_loss/output', input='ecoc_loss')
        self.model.add_output(name='final_loss/output', input='final_loss')

        return ['ecoc_loss/output', 'final_loss/output']
    
    
    
    def add_One_vs_One_Merge_Functional(self, inputs_list, nOutput, activation='softmax'):
        
        # join outputs from OneVsOne classifers
        ecoc_loss_name = 'ecoc_loss'
        final_loss_name = 'final_loss/out'
        ecoc_loss = merge(inputs_list, name=ecoc_loss_name, mode='concat', concat_axis=1)
        drop = Dropout(0.5, name='final_loss/drop')                                (ecoc_loss)
        # apply final joint prediction
        final_loss = Dense(nOutput, activation=activation, name=final_loss_name)    (drop)
        
        in_node = self.model.layers[0].name
        in_node = self.model.get_layer(in_node).output
        self.model = Model(input=in_node, output=[ecoc_loss, final_loss])
        #self.model = Model(input=in_node, output=['ecoc_loss', 'final_loss'])

        return [ecoc_loss_name, final_loss_name]



    def GAP(self, nOutput, input):
        """
            Creates a GAP network for object localization as described in the paper
                Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A.
                Learning Deep Features for Discriminative Localization.
                arXiv preprint arXiv:1512.04150. 2015 Dec 14.
            Outputs:
                'GAP/softmax' output of the final softmax classification
                'GAP/conv' output of the generated convolutional maps.
        """

        if(len(input) == 3):
            input_shape = tuple([input[2]] + input[0:2])
        else:
            input_shape = tuple(input)

        self.model = Graph()

        # Input
        self.model.add_input(name='input', input_shape=input_shape)

        # Layers
        self.model.add_node(ZeroPadding2D((1,1)), name='CAM_conv/zeropadding', input='input')
        self.model.add_node(Convolution2D(1024, 3, 3), name='CAM_conv', input='CAM_conv/zeropadding')
        self.model.add_node(Activation('relu'), name='CAM_conv/relu', input='CAM_conv')
        self.model.add_node(AveragePooling2D(pool_size=(14,14)), name='GAP', input='CAM_conv/relu')
        self.model.add_node(Flatten(), name='GAP/flatten', input='GAP')
        self.model.add_node(Dense(nOutput, activation='softmax'), name='GAP/classifier_food_vs_nofood', input='GAP/flatten')

        # Output
        self.model.add_output(name='GAP/softmax', input='GAP/classifier_food_vs_nofood')



    def Empty(self, nOutput, input):
        """
            Creates an empty CNN_Model (can be externally defined)
        """
        pass

    # ------------------------------------------------------- #
    #       SAVE/LOAD
    #           Auxiliary methods for saving and loading the model.
    # ------------------------------------------------------- #

    def __getstate__(self):
        """
            Behavour applied when pickling a CNN_Model instance.
        """
        obj_dict = self.__dict__.copy()
        del obj_dict['model']
        return obj_dict


