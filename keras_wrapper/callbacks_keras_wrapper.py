from __future__ import print_function

"""
Extra set of callbacks.
"""

import random
import warnings
import numpy as np
import logging

from keras.callbacks import Callback as KerasCallback


###
# Storing callbacks
###
class StoreModelWeightsOnEpochEnd(KerasCallback):
    def __init__(self, model, fun, epochs_for_save, verbose=0):
        """
        In:
            model - model to save
            fun - function for saving the model
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
###

###
# Printing callbacks
###



                            
###

###
# Learning modifiers callbacks
###
class LearningRateReducerWithEarlyStopping(KerasCallback):
    """
    Reduces learning rate during the training.

    Original work: jiumem [https://github.com/jiumem]
    """
    def __init__(self, 
            patience=0, lr_decay=1, reduce_rate=0.5, reduce_nb=10, 
            is_early_stopping=True, verbose=1):
        """
        In:
            patience - number of beginning epochs without reduction; 
                by default 0
            lr_decay - minimum number of epochs passed before the last reduction
            reduce_rate - multiplicative rate reducer; by default 0.5
            reduce_nb - maximal number of reductions performed; by default 10
            is_early_stopping - if true then early stopping is applied when
                reduce_nb is reached; by default True
            verbose - verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.is_early_stopping = is_early_stopping
        self.verbose = verbose
        self.epsilon = 0.1e-10
        self.lr_decay = lr_decay

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score is None:
            warnings.warn('validation score is off; ' + 
                    'this reducer works only with the validation score on')
            return
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                logging.info('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience and self.wait >= self.lr_decay:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = self.model.optimizer.lr.get_value()
                    self.model.optimizer.lr.set_value(np.float32(lr*self.reduce_rate))
                    if self.verbose > 0:
                        logging.info("LR reduction from {0:0.6f} to {1:0.6f}".\
                                format(float(lr), float(lr*self.reduce_rate)))
                    if float(lr) <= self.epsilon:
                        if self.verbose > 0:
                            logging.info('Learning rate too small, learning stops now')
                        self.model.stop_training = True
                else:
                    if self.is_early_stopping:
                        if self.verbose > 0:
                            logging.info("Epoch %d: early stopping" % (epoch))
                        self.model.stop_training = True
            self.wait += 1 
