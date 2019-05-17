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


def checkParameters(input_params, default_params, hard_check=False):
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

    def __init__(self, manager, multiprocess_type='Queue'):
        if multiprocess_type != 'Queue' and multiprocess_type != 'Pipe':
            raise NotImplementedError(
                'Not valid multiprocessing queue of type ' + multiprocess_type)

        self.type = multiprocess_type
        if multiprocess_type == 'Queue':
            self.queue = eval('manager.' + multiprocess_type + '()')
        else:
            self.queue = eval(multiprocess_type + '()')

    def put(self, elem):
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


def bbox(img, mode='max'):
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
            s = Stage(nInput=n_classes, nOutput=n_classes_ecoc,
                      input_shape=input_shape, output_shape=[1, 2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=n_classes_ecoc,
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

        logger.info('Built model %s/%s for classes %s in %0.5s seconds.' % (str(count + 1), str(n_combs), c, str(time.time() - t)))
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
        s = Stage(nInput=n_classes, nOutput=n_classes_ecoc, input_shape=input_shape,
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
            s = Stage(nInput=n_classes, nOutput=2, input_shape=input_shape,
                      output_shape=[2],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=2, input_shape=input_shape,
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
            s = Stage(nInput=n_classes, nOutput=3, input_shape=input_shape,
                      output_shape=[3],
                      type='One_vs_One_Inception', silence=True)
        elif ecoc_version == 2:
            s = Stage(nInput=n_classes, nOutput=3, input_shape=input_shape,
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


def prepareECOCLossOutputs(net, ds, ecoc_table, input_name, output_names,
                           splits=None):
    """

    :param net:
    :param ds:
    :param ecoc_table:
    :param input_name:
    :param output_names:
    :param splits:
    :return:
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    # Insert ecoc_table in net
    if 'additional_data' not in net.__dict__.keys():
        net.additional_data = dict()
    net.additional_data['ecoc_table'] = ecoc_table

    # Retrieve labels' id and images' id in dataset <- Necessary?
    # id_labels = ds.ids_outputs[ds.types_outputs.index('categorical')]
    id_labels_ecoc = 'labels_ecoc'

    # Insert ecoc-loss labels for each data split
    for s in splits:
        labels_ecoc = []
        # exec ('labels = ds.Y_' + s + '[id_labels]')
        y_split = getattr(ds, 'Y_' + s)
        labels = y_split[gt_id]
        n = len(labels)
        for i in range(n):
            labels_ecoc.append(ecoc_table[labels[i]])
        ds.setOutput(labels_ecoc, s, data_type='binary', id=id_labels_ecoc)

    # Set input and output mappings from dataset to network
    pos_images = ds.types_inputs.index('image')
    pos_labels = ds.types_outputs.index('categorical')
    pos_labels_ecoc = ds.types_outputs.index('binary')

    inputMapping = {input_name: pos_images}
    # inputMapping = {0: pos_images}
    net.setInputsMapping(inputMapping)

    outputMapping = {output_names[0]: pos_labels_ecoc, output_names[1]: pos_labels}
    # outputMapping = {0: pos_labels_ecoc, 1: pos_labels}
    net.setOutputsMapping(outputMapping, acc_output=output_names[1])


def loadGoogleNetForFood101(nClasses=101,
                            load_path='/media/HDD_2TB/CNN_MODELS/GoogleNet'):
    """

    :param nClasses:
    :param load_path:
    :return:
    """
    logger.info('Loading GoogLeNet...')

    # Build model (loading the previously converted Caffe's model)
    googLeNet = Stage(nClasses, nClasses, [224, 224, 3], [nClasses],
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


def simplifyDataset(ds, id_classes, n_classes=50):
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
                    # exec ('sample = ds.Y_' + s + '[id_out][i]')
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
        # exec ('ds.X_' + s + ' = copy.copy(kept_X)')
        # exec ('ds.Y_' + s + ' = copy.copy(kept_Y)')
        # exec ('ds.len_' + s + ' = len(kept_Y[id_labels])')
        setattr(ds, 'X_' + s, copy.copy(kept_X))
        setattr(ds, 'Y_' + s, copy.copy(kept_Y))
        setattr(ds, 'len_' + s, len(kept_Y[id_labels]))


def average_models(models, output_model, weights=None, custom_objects=None):
    from keras_wrapper.cnn_model import loadModel, saveModel
    if not isinstance(models, list):
        raise AssertionError('You must give a list of models to average.')
    if len(models) == 0:
        raise AssertionError('You provided an empty list of models to average!')

    model_weights = np.asarray([1. / len(models)] * len(models),
                               dtype=np.float32) if (weights is None) or (weights == []) else np.asarray(weights, dtype=np.float32)
    if len(model_weights) != len(models):
        raise AssertionError(
            'You must give a list of weights of the same size than the list of models.')
    loaded_models = [loadModel(m, -1, full_path=True, custom_objects=custom_objects) for m in models]

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

    if not (all([[str(loaded_models[0].model.weights[i]) == str(loaded_model.model.weights[i]) for i in range(len(loaded_models[0].model.weights))] for loaded_model in loaded_models])):
        raise AssertionError('Not all models have the same weights!')

    if hasattr(loaded_models[0], 'model_init') and getattr(loaded_models[0], 'model_init') is not None:
        if not all([[str(loaded_models[0].model.weights[i]) == str(loaded_model.model.weights[i]) for i in range(len(loaded_models[0].model_init.weights))] for loaded_model in loaded_models]):
            raise AssertionError('Not all model_inits have the same weights!')

    if hasattr(loaded_models[0], 'model_next') and getattr(loaded_models[0], 'model_next') is not None:
        if not all([[str(loaded_models[0].model_next.weights[i]) == str(loaded_model.model_next.weights[i]) for i in range(len(loaded_models[0].model_next.weights))] for loaded_model in loaded_models]):
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
            [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in range(len(current_weights))])

        # We have model_init
        if hasattr(loaded_models[0], 'model_init') and getattr(loaded_models[0], 'model_init') is not None:
            current_weights = loaded_models[m].model_init.get_weights()
            prev_weights = loaded_models[0].model_init.get_weights()
            loaded_models[0].model_init.set_weights(
                [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in range(len(current_weights))])

        # We have model_next
        if hasattr(loaded_models[0], 'model_next') and getattr(loaded_models[0], 'model_next') is not None:
            current_weights = loaded_models[m].model_next.get_weights()
            prev_weights = loaded_models[0].model_next.get_weights()
            loaded_models[0].model_next.set_weights(
                [current_weights[matrix_index] * model_weights[m] + prev_weights[matrix_index] for matrix_index in range(len(current_weights))])

    # Save averaged model
    saveModel(loaded_models[0], -1, path=output_model, full_path=True,
              store_iter=False)


# Text-related utils
def one_hot_2_indices(preds, pad_sequences=True, verbose=0):
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


def indices_2_one_hot(indices, n):
    """
    Converts a list of indices into one hot codification

    :param indices: list of indices
    :param n: integer. Size of the vocabulary
    :return: numpy array with shape (len(indices), n)
    """
    one_hot = np.zeros((len(indices), n), dtype=np.int8)
    for i in range(len(indices)):
        if indices[i] >= n:
            raise ValueError("Index out of bounds when converting to one hot")
        one_hot[i, indices[i]] = 1

    return one_hot


# From keras.utils.np_utils
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


# ------------------------------------------------------- #
#       DECODING FUNCTIONS
#           Functions for decoding predictions
# ------------------------------------------------------- #

def decode_predictions_one_hot(preds, index2word, pad_sequences=True, verbose=0):
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


def decode_categorical(preds, index2word, verbose=0):
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


def decode_multilabel(preds, index2word, min_val=0.5, get_probs=False, verbose=0):
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


def replace_unknown_words(src_word_seq, trg_word_seq, hard_alignment, unk_symbol,
                          glossary=None, heuristic=0, mapping=None, verbose=0):
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


def decode_predictions_beam_search(preds, index2word, glossary=None, alphas=None,
                                   heuristic=0,
                                   x_text=None, unk_symbol='<unk>',
                                   pad_sequences=False,
                                   mapping=None, verbose=0):
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


def sampling(scores, sampling_type='max_likelihood', temperature=1.0):
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
