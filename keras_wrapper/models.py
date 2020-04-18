# -*- coding: utf-8 -*-
from __future__ import print_function
import copy
import itertools
import logging
import numpy as np
import time

import keras
import keras.backend as K
from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import concatenate, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Dense, Dropout, Flatten, Input, \
    Activation, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2

if int(keras.__version__.split('.')[0]) == 1:
    from keras.layers import Concat as Concatenate
    from keras.layers import Convolution2D as Conv2D
    from keras.layers import Deconvolution2D as Conv2DTranspose
else:
    from keras.layers import Concatenate
    from keras.layers import Conv2D
    from keras.layers import Conv2DTranspose

from keras_wrapper.cnn_model import Model_Wrapper

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


# ------------------------------------------------------- #
#   MODELS
#       Available definitions of CNN models (see basic_model as an example)
#       All the models must include the following parameters:
#           nOutput, input
# ------------------------------------------------------- #

class Predefined_Model(Model_Wrapper):
    """
        Wrapper for Keras' models. It provides the following utilities:
            - Set of already implemented CNNs for quick definition.
            - Easy layers re-definition for finetuning.
    """
    def __init__(self,
                 nOutput=1000,
                 model_type='basic_model',
                 silence=False,
                 input_shape=None,
                 structure_path=None,
                 weights_path=None,
                 seq_to_functional=False,
                 model_name=None,
                 plots_path=None,
                 models_path=None,
                 inheritance=False,
                 *args,
                 **kwargs):
        self.nOutput = nOutput
        self.input_shape = input_shape or [256, 256, 3]
        super(Predefined_Model, self).__init__(
            model_type=model_type,
            silence=silence,
            structure_path=structure_path,
            weights_path=weights_path,
            seq_to_functional=seq_to_functional,
            model_name=model_name,
            plots_path=plots_path,
            models_path=models_path,
            inheritance=inheritance,
        )

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


def build_OneVsOneECOC_Stage(n_classes_ecoc, input_shape, ds, stage1_lr=0.01,
                             ecoc_version=2):
    """

    :param n_classes_ecoc:
    :param input_shape:
    :param ds:
    :param stage1_lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)
    labels_list = [str(l) for l in range(n_classes)]

    combs = tuple(itertools.combinations(labels_list, n_classes_ecoc))
    stage = list()
    outputs_list = list()

    count = 0
    n_combs = len(combs)
    for c in combs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Model(nInput=n_classes, nOutput=n_classes_ecoc,
                      input_shape=input_shape, output_shape=[1, 2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Model(nInput=n_classes, nOutput=n_classes_ecoc,
                      input_shape=input_shape, output_shape=[1, 2],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            i_str = str(i)
            if i_str in c:
                input_mapping[i] = c.index(i_str)
            else:
                input_mapping[i] = None
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=stage1_lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logger.info('Built model %s/%s for classes %s in %0.5s seconds.' % (
            str(count + 1), str(n_combs), c, str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_OneVsAllECOC_Stage(n_classes_ecoc, input_shape, ds, stage1_lr):
    """

    :param n_classes_ecoc:
    :param input_shape:
    :param ds:
    :param stage1_lr:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    for c in range(n_classes):
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        s = Model(nInput=n_classes, nOutput=n_classes_ecoc, input_shape=input_shape,
                  output_shape=[1],
                  type='One_vs_One_Inception', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i == c:
                input_mapping[i] = 0
            else:
                input_mapping[i] = 1
        # Build output mask
        output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        s.defineOutputMask(output_mask)
        s.setOptimizer(lr=stage1_lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logger.info('Built model %s/%s for classes %s in %0.5s seconds.' % (
            str(count + 1), str(n_classes), '(' + str(c) + ' vs All)',
            str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneECOC_Stage(pairs, input_shape, ds, lr, ecoc_version=2):
    """

    :param pairs:
    :param input_shape:
    :param ds:
    :param lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    n_pairs = len(pairs)
    logger.info("Building " + str(n_pairs) + " classifiers...")

    for c in pairs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Model(nInput=n_classes, nOutput=2, input_shape=input_shape,
                      output_shape=[2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Model(nInput=n_classes, nOutput=2, input_shape=input_shape,
                      output_shape=[2],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i in c:
                input_mapping[i] = c.index(i)
            else:
                input_mapping[i] = None
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logger.info('Built model %s/%s for classes %s = %s in %0.5s seconds.' % (
            str(count + 1), str(n_pairs), c, (ds.classes[c[0]], ds.classes[c[1]]),
            str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneVsRestECOC_Stage(pairs, input_shape, ds, lr,
                                            ecoc_version=2):
    """

    :param pairs:
    :param input_shape:
    :param ds:
    :param lr:
    :param ecoc_version:
    :return:
    """
    n_classes = len(ds.classes)

    stage = list()
    outputs_list = list()

    count = 0
    n_pairs = len(pairs)
    for c in pairs:
        t = time.time()

        # Create each one_vs_one classifier of the intermediate stage
        if ecoc_version == 1:
            s = Model(nInput=n_classes, nOutput=3, input_shape=input_shape,
                      output_shape=[3],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Model(nInput=n_classes, nOutput=3, input_shape=input_shape,
                      output_shape=[3],
                      type='One_vs_One_Inception_v2', silence=True)
        # Build input mapping
        input_mapping = dict()
        for i in range(n_classes):
            if i in c:
                input_mapping[i] = c.index(i)
            else:
                input_mapping[i] = 2
        # Build output mask
        # output_mask = {'[0]': [0], '[1]': None}
        s.defineClassMapping(input_mapping)
        # s.defineOutputMask(output_mask)
        s.setOptimizer(lr=lr)
        s.silence = False
        stage.append(s)

        outputs_list.append('loss_OnevsOne/output')

        logger.info('Built model %s/%s for classes %s = %s in %0.5s seconds.' % (
            str(count + 1), str(n_pairs), c, (ds.classes[c[0]], ds.classes[c[1]]),
            str(time.time() - t)))
        count += 1

    return [stage, outputs_list]


def build_Specific_OneVsOneECOC_loss_Stage(net, input_net, input_shape, classes,
                                           ecoc_version=3, pairs=None,
                                           functional_api=False, activations=None):
    """

    :param net:
    :param input_net:
    :param input_shape:
    :param classes:
    :param ecoc_version:
    :param pairs:
    :param functional_api:
    :param activations:
    :return:
    """
    from keras.layers.convolutional import ZeroPadding2D
    if activations is None:
        activations = ['softmax', 'softmax']
    n_classes = len(classes)
    if pairs is None:  # generate any possible combination of two classes
        pairs = tuple(itertools.combinations(range(n_classes), 2))

    outputs_list = list()
    n_pairs = len(pairs)
    ecoc_table = np.zeros((n_classes, n_pairs, 2))

    logger.info("Building " + str(n_pairs) + " OneVsOne structures...")

    for i, c in list(enumerate(pairs)):
        # t = time.time()

        # Insert 1s in the corresponding positions of the ecoc table
        ecoc_table[c[0], i, 0] = 1
        ecoc_table[c[1], i, 1] = 1

        # Create each one_vs_one classifier of the intermediate stage
        if not functional_api:
            if ecoc_version == 1:
                output_name = net.add_One_vs_One_Inception(input_net, input_shape, i,
                                                           nOutput=2,
                                                           activation=activations[0])
            elif ecoc_version == 2:
                output_name = net.add_One_vs_One_Inception_v2(input_net, input_shape,
                                                              i, nOutput=2,
                                                              activation=activations[
                                                                  0])
            else:
                raise NotImplementedError
        else:
            if ecoc_version == 1:
                output_name = net.add_One_vs_One_Inception_Functional(input_net,
                                                                      input_shape, i,
                                                                      nOutput=2,
                                                                      activation=activations[0])
            elif ecoc_version == 2:
                raise NotImplementedError()
            elif ecoc_version == 3 or ecoc_version == 4 or ecoc_version == 5 or ecoc_version == 6:
                if ecoc_version == 3:
                    nkernels = 16
                elif ecoc_version == 4:
                    nkernels = 64
                elif ecoc_version == 5:
                    nkernels = 128
                elif ecoc_version == 6:
                    nkernels = 256
                else:
                    raise NotImplementedError()
                if i == 0:
                    in_node = net.model.get_layer(input_net).output
                    padding_node = ZeroPadding2D(padding=(1, 1),
                                                 name='3x3/ecoc_padding')(in_node)
                output_name = net.add_One_vs_One_3x3_Functional(padding_node,
                                                                input_shape, i,
                                                                nkernels, nOutput=2,
                                                                activation=activations[0])
            elif ecoc_version == 7:
                if i == 0:
                    in_node = net.model.get_layer(input_net).output
                    padding_node = ZeroPadding2D(padding=(1, 1),
                                                 name='3x3/ecoc_padding')(in_node)
                output_name = net.add_One_vs_One_3x3_double_Functional(padding_node,
                                                                       input_shape,
                                                                       i, nOutput=2,
                                                                       activation=activations[0])
            else:
                raise NotImplementedError()
        outputs_list.append(output_name)

        # logger.info('Built model %s/%s for classes %s = %s in %0.5s seconds.'%(str(i+1),
        #  str(n_pairs), c, (classes[c[0]], classes[c[1]]), str(time.time()-t)))

    ecoc_table = np.reshape(ecoc_table, [n_classes, 2 * n_pairs])

    # Build final Softmax layer
    if not functional_api:
        output_names = net.add_One_vs_One_Merge(outputs_list, n_classes,
                                                activation=activations[1])
    else:
        output_names = net.add_One_vs_One_Merge_Functional(outputs_list, n_classes,
                                                           activation=activations[1])
    logger.info('Built ECOC merge layers.')

    return [ecoc_table, output_names]


def loadGoogleNetForFood101(nClasses=101,
                            load_path='/media/HDD_2TB/CNN_MODELS/GoogleNet'):
    """

    :param nClasses:
    :param load_path:
    :return:
    """
    logger.info('Loading GoogLeNet...')

    # Build model (loading the previously converted Caffe's model)
    googLeNet = Model(nClasses, nClasses, [224, 224, 3], [nClasses],
                      type='GoogleNet',
                      model_name='GoogleNet_Food101_retrained',
                      structure_path=load_path + '/Keras_model_structure.json',
                      weights_path=load_path + '/Keras_model_weights.h5')

    return googLeNet


def prepareGoogleNet_Food101(model_wrapper):
    """
    Prepares the GoogleNet model after its conversion from Caffe
    :param model_wrapper:
    :return:
    """
    # Remove unnecessary intermediate optimizers
    layers_to_delete = ['loss2/ave_pool', 'loss2/conv', 'loss2/relu_conv',
                        'loss2/fc_flatten', 'loss2/fc',
                        'loss2/relu_fc', 'loss2/drop_fc', 'loss2/classifier',
                        'output_loss2/loss',
                        'loss1/ave_pool', 'loss1/conv', 'loss1/relu_conv',
                        'loss1/fc_flatten', 'loss1/fc',
                        'loss1/relu_fc', 'loss1/drop_fc', 'loss1/classifier',
                        'output_loss1/loss']
    model_wrapper.removeLayers(layers_to_delete)
    model_wrapper.removeOutputs(['loss1/loss', 'loss2/loss'])


def prepareGoogleNet_Food101_ECOC_loss(model_wrapper):
    """
    Prepares the GoogleNet model for inserting an ECOC structure after removing the last part of the net
    :param model_wrapper:
    :return:
    """
    # Remove all last layers (from 'inception_5a' included)
    layers_to_delete = ['inception_5a/1x1', 'inception_5a/relu_1x1',
                        'inception_5a/3x3_reduce',
                        'inception_5a/relu_3x3_reduce',
                        'inception_5a/3x3_zeropadding', 'inception_5a/3x3',
                        'inception_5a/relu_3x3',
                        'inception_5a/5x5_reduce',
                        'inception_5a/relu_5x5_reduce',
                        'inception_5a/5x5_zeropadding', 'inception_5a/5x5',
                        'inception_5a/relu_5x5',
                        'inception_5a/pool_zeropadding', 'inception_5a/pool',
                        'inception_5a/pool_proj',
                        'inception_5a/relu_pool_proj', 'inception_5a/output',
                        'inception_5b/1x1',
                        'inception_5b/relu_1x1', 'inception_5b/3x3_reduce',
                        'inception_5b/relu_3x3_reduce',
                        'inception_5b/3x3_zeropadding', 'inception_5b/3x3',
                        'inception_5b/relu_3x3',
                        'inception_5b/5x5_reduce',
                        'inception_5b/relu_5x5_reduce',
                        'inception_5b/5x5_zeropadding', 'inception_5b/5x5',
                        'inception_5b/relu_5x5',
                        'inception_5b/pool_zeropadding', 'inception_5b/pool',
                        'inception_5b/pool_proj',
                        'inception_5b/relu_pool_proj',
                        'inception_5b/output', 'pool5/7x7_s1', 'pool5/drop_7x7_s1',
                        'loss3/classifier_foodrecognition_flatten',
                        'loss3/classifier_foodrecognition']
    [layers, params] = model_wrapper.removeLayers(copy.copy(layers_to_delete))
    # Remove softmax output
    model_wrapper.removeOutputs(['loss3/loss3'])

    return ['pool4/3x3_s2',
            [832, 7, 7]]  # returns the name of the last layer and its output shape
    # Adds a new output after the layer 'pool4/3x3_s2'
    # model_wrapper.model.add_output(name='pool4', input='pool4/3x3_s2')


def prepareGoogleNet_Food101_Stage1(model_wrapper):
    """
    Prepares the GoogleNet model for serving as the first Stage of a Staged_Netork
    :param model_wrapper:
    :return:
    """
    # Adds a new output after the layer 'pool4/3x3_s2'
    model_wrapper.model.add_output(name='pool4', input='pool4/3x3_s2')


def prepareGoogleNet_Stage2(stage1, stage2):
    """
    Removes the second part of the GoogleNet for inserting it into the second stage.
    :param stage1:
    :param stage2:
    :return:
    """
    # Remove all last layers (from 'inception_5a' included)
    layers_to_delete = ['inception_5a/1x1', 'inception_5a/relu_1x1',
                        'inception_5a/3x3_reduce',
                        'inception_5a/relu_3x3_reduce',
                        'inception_5a/3x3_zeropadding', 'inception_5a/3x3',
                        'inception_5a/relu_3x3',
                        'inception_5a/5x5_reduce',
                        'inception_5a/relu_5x5_reduce',
                        'inception_5a/5x5_zeropadding', 'inception_5a/5x5',
                        'inception_5a/relu_5x5',
                        'inception_5a/pool_zeropadding', 'inception_5a/pool',
                        'inception_5a/pool_proj',
                        'inception_5a/relu_pool_proj',
                        'inception_5a/output', 'inception_5b/1x1',
                        'inception_5b/relu_1x1', 'inception_5b/3x3_reduce',
                        'inception_5b/relu_3x3_reduce',
                        'inception_5b/3x3_zeropadding', 'inception_5b/3x3',
                        'inception_5b/relu_3x3',
                        'inception_5b/5x5_reduce',
                        'inception_5b/relu_5x5_reduce',
                        'inception_5b/5x5_zeropadding', 'inception_5b/5x5',
                        'inception_5b/relu_5x5',
                        'inception_5b/pool_zeropadding', 'inception_5b/pool',
                        'inception_5b/pool_proj',
                        'inception_5b/relu_pool_proj',
                        'inception_5b/output', 'pool5/7x7_s1', 'pool5/drop_7x7_s1',
                        'loss3/classifier_foodrecognition_flatten',
                        'loss3/classifier_foodrecognition', 'output_loss3/loss3']
    [layers, params] = stage1.removeLayers(copy.copy(layers_to_delete))
    # Remove softmax output
    stage1.removeOutputs(['loss3/loss3'])

    layers_to_delete_2 = ["conv1/7x7_s2_zeropadding", "conv1/7x7_s2",
                          "conv1/relu_7x7", "pool1/3x3_s2_zeropadding",
                          "pool1/3x3_s2", "pool1/norm1", "conv2/3x3_reduce",
                          "conv2/relu_3x3_reduce",
                          "conv2/3x3_zeropadding", "conv2/3x3", "conv2/relu_3x3",
                          "conv2/norm2",
                          "pool2/3x3_s2_zeropadding", "pool2/3x3_s2",
                          "inception_3a/1x1", "inception_3a/relu_1x1",
                          "inception_3a/3x3_reduce", "inception_3a/relu_3x3_reduce",
                          "inception_3a/3x3_zeropadding",
                          "inception_3a/3x3", "inception_3a/relu_3x3",
                          "inception_3a/5x5_reduce",
                          "inception_3a/relu_5x5_reduce",
                          "inception_3a/5x5_zeropadding", "inception_3a/5x5",
                          "inception_3a/relu_5x5", "inception_3a/pool_zeropadding",
                          "inception_3a/pool",
                          "inception_3a/pool_proj", "inception_3a/relu_pool_proj",
                          "inception_3a/output",
                          "inception_3b/1x1", "inception_3b/relu_1x1",
                          "inception_3b/3x3_reduce",
                          "inception_3b/relu_3x3_reduce",
                          "inception_3b/3x3_zeropadding", "inception_3b/3x3",
                          "inception_3b/relu_3x3", "inception_3b/5x5_reduce",
                          "inception_3b/relu_5x5_reduce",
                          "inception_3b/5x5_zeropadding", "inception_3b/5x5",
                          "inception_3b/relu_5x5",
                          "inception_3b/pool_zeropadding", "inception_3b/pool",
                          "inception_3b/pool_proj",
                          "inception_3b/relu_pool_proj", "inception_3b/output",
                          "pool3/3x3_s2_zeropadding",
                          "pool3/3x3_s2", "inception_4a/1x1",
                          "inception_4a/relu_1x1", "inception_4a/3x3_reduce",
                          "inception_4a/relu_3x3_reduce",
                          "inception_4a/3x3_zeropadding", "inception_4a/3x3",
                          "inception_4a/relu_3x3", "inception_4a/5x5_reduce",
                          "inception_4a/relu_5x5_reduce",
                          "inception_4a/5x5_zeropadding", "inception_4a/5x5",
                          "inception_4a/relu_5x5",
                          "inception_4a/pool_zeropadding", "inception_4a/pool",
                          "inception_4a/pool_proj",
                          "inception_4a/relu_pool_proj", "inception_4a/output",
                          "inception_4b/1x1",
                          "inception_4b/relu_1x1", "inception_4b/3x3_reduce",
                          "inception_4b/relu_3x3_reduce",
                          "inception_4b/3x3_zeropadding", "inception_4b/3x3",
                          "inception_4b/relu_3x3",
                          "inception_4b/5x5_reduce", "inception_4b/relu_5x5_reduce",
                          "inception_4b/5x5_zeropadding",
                          "inception_4b/5x5", "inception_4b/relu_5x5",
                          "inception_4b/pool_zeropadding",
                          "inception_4b/pool", "inception_4b/pool_proj",
                          "inception_4b/relu_pool_proj",
                          "inception_4b/output", "inception_4c/1x1",
                          "inception_4c/relu_1x1", "inception_4c/3x3_reduce",
                          "inception_4c/relu_3x3_reduce",
                          "inception_4c/3x3_zeropadding", "inception_4c/3x3",
                          "inception_4c/relu_3x3", "inception_4c/5x5_reduce",
                          "inception_4c/relu_5x5_reduce",
                          "inception_4c/5x5_zeropadding", "inception_4c/5x5",
                          "inception_4c/relu_5x5",
                          "inception_4c/pool_zeropadding", "inception_4c/pool",
                          "inception_4c/pool_proj",
                          "inception_4c/relu_pool_proj", "inception_4c/output",
                          "inception_4d/1x1",
                          "inception_4d/relu_1x1", "inception_4d/3x3_reduce",
                          "inception_4d/relu_3x3_reduce",
                          "inception_4d/3x3_zeropadding", "inception_4d/3x3",
                          "inception_4d/relu_3x3",
                          "inception_4d/5x5_reduce", "inception_4d/relu_5x5_reduce",
                          "inception_4d/5x5_zeropadding",
                          "inception_4d/5x5", "inception_4d/relu_5x5",
                          "inception_4d/pool_zeropadding",
                          "inception_4d/pool", "inception_4d/pool_proj",
                          "inception_4d/relu_pool_proj",
                          "inception_4d/output", "inception_4e/1x1",
                          "inception_4e/relu_1x1", "inception_4e/3x3_reduce",
                          "inception_4e/relu_3x3_reduce",
                          "inception_4e/3x3_zeropadding", "inception_4e/3x3",
                          "inception_4e/relu_3x3", "inception_4e/5x5_reduce",
                          "inception_4e/relu_5x5_reduce",
                          "inception_4e/5x5_zeropadding", "inception_4e/5x5",
                          "inception_4e/relu_5x5",
                          "inception_4e/pool_zeropadding", "inception_4e/pool",
                          "inception_4e/pool_proj",
                          "inception_4e/relu_pool_proj", "inception_4e/output",
                          "pool4/3x3_s2_zeropadding",
                          "pool4/3x3_s2"]

    # Remove initial layers
    [layers_, params_] = stage2.removeLayers(copy.copy(layers_to_delete_2))
    # Remove previous input
    stage2.removeInputs(['input_data'])
    # Add new input
    stage2.model.add_input(name='input_data', input_shape=(832, 7, 7))
    stage2.model.nodes[layers_to_delete[0]].previous = stage2.model.inputs[
        'input_data']
    """
    # Insert layers into stage
     stage2.model = Graph()
    # Input
     stage2.model.add_input(name='input_data', input_shape=(832,7,7))
     for l_name,l,p in zip(layers_to_delete, layers, params):
        stage2.model.namespace.add(l_name)
        stage2.model.nodes[l_name] = l
        stage2.model.node_config.append(p)
    #input = stage2.model.input # keep input
    # Connect first layer with input
     stage2.model.node_config[0]['input'] = 'input_data'
     stage2.model.nodes[layers_to_delete[0]].previous = stage2.model.inputs['input_data']
     stage2.model.input_config[0]['input_shape'] = [832,7,7]

    # Output
     stage2.model.add_output(name='loss3/loss3', input=layers_to_delete[-1])
    #stage2.model.add_output(name='loss3/loss3_', input=layers_to_delete[-1])
    #stage2.model.input = input # recover input
    """
