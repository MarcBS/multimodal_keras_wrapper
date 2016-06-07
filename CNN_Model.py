from ThreadLoader import ThreadDataLoader, retrieveXY
from Dataset import Dataset
from ECOC_Classifier import ECOC_Classifier

from keras.models import Sequential, Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.caffe.extra_layers import LRN2D

import matplotlib as mpl
mpl.use('Agg') # run matplotlib without X server (GUI)
import matplotlib.pyplot as plt

import numpy as np
import cPickle as pk

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
    open(path + '/iter_'+ iter +'_structure.json', 'w').write(json_string)
    # Save model weights
    model_wrapper.model.save_weights(path + '/iter_'+ iter +'_weights.h5', overwrite=True)
    # Save additional information
    pk.dump(model_wrapper, open(path + '/iter_' + iter + '_CNN_Model.pkl', 'wb'))
    
    if(not model_wrapper.silence):
        logging.info("<<< Model saved >>>")


def loadModel(model_path, iter):
    """
        Loads a previously saved CNN_Model object.
    """
    t = time.time()
    iter = str(iter)
    logging.info("<<< Loading model from "+ model_path + "/iter_" + iter + "_CNN_Model.pkl ... >>>")
    
    # Load model structure
    model = model_from_json(open(model_path + '/iter_'+ iter +'_structure.json').read())
    # Load model weights
    model.load_weights(model_path + '/iter_'+ iter +'_weights.h5')
    # Load additional information
    model_wrapper = pk.load(open(model_path + '/iter_' + iter + '_CNN_Model.pkl', 'rb'))
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
                structure_path=None, weights_path=None, model_name=None, plots_path=None, models_path=None):
        """
            CNN_Model object constructor. 
            
            :param nOutput: number of outputs of the network. Only valid if 'structure_path' == None.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param silence: set to True if you don't want the model to output informative messages
            :param input_shape: array with 3 integers which define the images' input shape [height, width, channels]. Only valid if 'structure_path' == None.
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param plots_path: path to the folder where the plots will be stored during training
            :param models_path: path to the folder where the temporal model packups will be stored
        """
        self.__toprint = ['net_type', 'name', 'plot_path', 'model_path', 'lr', 'momentum', 
                            'training_parameters', 'testing_parameters', 'training_state', 'loss', 'silence']
        
        self.silence = silence
        self.net_type = type
        self.lr = 0.01 # insert default learning rate
        self.momentum = 1.0-self.lr # insert default momentum
        self.loss='categorical_crossentropy' # default loss function
        self.training_parameters = []
        self.testing_parameters = []
        self.training_state = dict()
        
        # Set Network name
        self.setName(model_name, plots_path, models_path)
        
        
        # Prepare logger
        self.__logger = dict()
        self.__modes = ['train', 'val']
        self.__data_types = ['iteration', 'loss', 'accuracy', 'accuracy top-5']
        
        # Prepare model
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
            self.model.load_weights(weights_path)
    
    
    def setOptimizer(self, lr=None, momentum=None, loss=None):
        """
            Sets a new optimizer for the CNN model.
            
            :param lr: learning rate of the network
            :param momentum: momentum of the network (if None, then momentum = 1-lr)
            :param loss: loss function applied for optimization
        """
        # Pick default parameters
        if(not lr):
            lr = self.lr
        else:
            self.lr = lr
        if(not momentum):
            momentum = self.momentum
        else:
            self.momentum = momentum
        if(not loss):
            loss = self.loss
        else:
            self.loss = loss
            
        #sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
        sgd = SGD(lr=lr, decay=0.0, momentum=momentum, nesterov=True)
        
        # compile differently depending if our model is 'Sequential' or 'Graph'
        if(isinstance(self.model, Sequential)):
            self.model.compile(loss=loss, optimizer=sgd)
        else:
            loss_dict = dict()
            for out in self.model.output_order:
                loss_dict[out] = loss
            self.model.compile(loss=loss_dict, optimizer=sgd)
        
        if(not self.silence):
            logging.info("Optimizer updated, learning rate set to "+ str(lr))
        
        
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
        default_params = {'n_epochs': 1, 'batch_size': 50, 'report_iter': 50, 'iter_for_val': 1000, 
                                'lr_decay': 1000, 'lr_gamma':0.1, 'save_model': 5000, 'num_iterations_val': None,
                                'n_parallel_loaders': 8, 'normalize_images': False, 'mean_substraction': True,
                                'data_augmentation': True};
        params = self.checkParameters(parameters, default_params)
        self.training_parameters.append(copy.copy(params))
        
        logging.info("<<< Training model >>>")
        
        self.__logger = dict()
        self.__train(ds, params)
        
        logging.info("<<< Finished training CNN_Model >>>")
        
        
    def resumeTrainNet(self, ds, parameters, out_name=None):
        """
            Resumes the last training state of a stored model keeping also its training parameters. 
            If we introduce any parameter through the argument 'parameters', it will be replaced by the old one.
            
            :param out_name: name of the output node that will be used to evaluate the network accuracy. Only applicable for Graph models.
        """
        # Recovers the old training parameters (replacing them by the new ones if any)
        default_params = self.training_parameters[-1]
        params = self.checkParameters(parameters, default_params)
        self.training_parameters.append(copy.copy(params))
        
        # Recovers the last training state
        state = self.training_state
    
        logging.info("<<< Resuming training model >>>")
        
        self.__train(ds, params, state)
        
        logging.info("<<< Finished training CNN_Model >>>")
        
        
    def __train(self, ds, params, state=dict(), out_name=None):
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
                t = t_queue[state['it']]
                t.join()
                if(t.resultOK):
                    X_batch = t.X 
                    Y_batch = t.Y
                else:
                    exc_type, exc_obj, exc_trace = t.exception
                    # deal with the exception
                    print exc_type, exc_obj
                    print exc_trace
                    raise Exception('Exception occurred in ThreadLoader.')
                t_queue[state['it']] = None
                if(state['it']+params['n_parallel_loaders'] < state['n_iterations_per_epoch']):
                    t = t_queue[state['it']+params['n_parallel_loaders']]
                    t.start()
                
                # Forward and backward passes on the current batch
                if(isinstance(self.model, Sequential)):
                    loss = self.model.train_on_batch(X_batch, Y_batch, accuracy=False)
                    loss = loss[0]
                    [score, top_score] = self._getSequentialAccuracy(Y_batch, self.model.predict_on_batch(X_batch)[0])
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
                        if(isinstance(self.model, Sequential)):
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
            
            if(isinstance(self.model, Sequential)):
#                (loss, score) = self.model.evaluate(X_test, Y_test, show_accuracy=True)
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
        if(isinstance(self.model, Sequential)):
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
        
        # Prepare data if Graph model
        if(isinstance(self.model, Graph)):
            [X, last_out] = self._prepareGraphData(X, np.zeros((X.shape[0],1)))
        
        # Apply forward pass for prediction
        predictions = self.model.predict_on_batch(X)
        
        # Select output if indicated
        if(isinstance(self.model, Graph)): # Graph
            if(out_name):
                predictions = predictions[out_name]
        else: # Sequential
            predictions = predictions[0]
            
        return predictions
    
    
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
    
        
    def _prepareGraphData(self, X, Y):
        # Currently all samples are assigned to all inputs and all labels to all outputs
        data = dict()
        last_out = ''
        for input in self.model.input_order:
            data[input] = X
        for output in self.model.output_order:
            data[output] = Y
            last_out = output
        return [data, last_out]
    
    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for train logging and visualization
    # ------------------------------------------------------- #
    
    def __str__(self):
        """
            Plot basic model information.
        """
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
        
    