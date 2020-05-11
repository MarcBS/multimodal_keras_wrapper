# -*- coding: utf-8 -*-
from __future__ import print_function

import cloudpickle as cloudpk
import logging
import os
import sys
import time
from six import iteritems

if sys.version_info.major == 3:
    import _pickle as pk
else:
    import cPickle as pk

from keras.models import model_from_json, load_model
from keras_wrapper.extra.read_write import create_dir_if_not_exists

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


# Save and load Dataset instances
def saveDataset(dataset,
                store_path):
    """
    Saves a backup of the current Dataset object.

    :param dataset: Dataset object to save
    :param store_path: Saving path
    :return: None
    """
    create_dir_if_not_exists(store_path)
    store_path = os.path.join(store_path,
                              'Dataset_' + dataset.name + '.pkl')
    if not dataset.silence:
        logger.info("<<< Saving Dataset instance to " + store_path + " ... >>>")

    pk.dump(dataset,
            open(store_path, 'wb'),
            protocol=-1)

    if not dataset.silence:
        logger.info("<<< Dataset instance saved >>>")


def loadDataset(dataset_path):
    """
    Loads a previously saved Dataset object.

    :param dataset_path: Path to the stored Dataset to load
    :return: Loaded Dataset object
    """

    logger.info("<<< Loading Dataset instance from " + dataset_path + " ... >>>")
    if sys.version_info.major == 3:
        dataset = pk.load(open(dataset_path, 'rb'),
                          encoding='utf-8')
    else:
        dataset = pk.load(open(dataset_path, 'rb'))

    if not hasattr(dataset, 'pad_symbol'):
        dataset.pad_symbol = '<pad>'
    if not hasattr(dataset, 'unk_symbol'):
        dataset.unk_symbol = '<unk>'
    if not hasattr(dataset, 'null_symbol'):
        dataset.null_symbol = '<null>'

    logger.info("<<< Dataset instance loaded >>>")
    return dataset


# Save and load Model_Wrapper instances
def saveModel(model_wrapper,
              update_num,
              path=None,
              full_path=False,
              store_iter=False):
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
            model_name = os.path.join(path,
                                      'update_' + iteration)
        else:
            model_name = os.path.join(path,
                                      'epoch_' + iteration)

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
        model_wrapper.model.save_weights(model_name + '_weights.h5',
                                         overwrite=True)

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
            model_wrapper.model_init.save_weights(model_name + '_weights_init.h5',
                                                  overwrite=True)

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
            model_wrapper.model_next.save_weights(model_name + '_weights_next.h5',
                                                  overwrite=True)

    # Save additional information
    backup_multi_gpu_model = None
    if hasattr(model_wrapper, 'multi_gpu_model'):
        backup_multi_gpu_model = model_wrapper.multi_gpu_model
        setattr(model_wrapper, 'multi_gpu_model', None)

    backup_tensorboard_callback = None
    if hasattr(model_wrapper, 'tensorboard_callback'):
        backup_tensorboard_callback = model_wrapper.tensorboard_callback
        setattr(model_wrapper, 'tensorboard_callback', None)

    cloudpk.dump(model_wrapper, open(model_name + '_Model_Wrapper.pkl', 'wb'))
    setattr(model_wrapper, 'multi_gpu_model', backup_multi_gpu_model)
    setattr(model_wrapper, 'tensorboard_callback', backup_tensorboard_callback)

    if not model_wrapper.silence:
        logger.info("<<< Model saved >>>")


def loadModel(model_path,
              update_num,
              reload_epoch=True,
              custom_objects=None,
              full_path=False,
              compile_model=False):
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
            model_name = os.path.join(model_path,
                                      "epoch_" + iteration)
        else:
            model_name = os.path.join(model_path,
                                      "update_" + iteration)

    logger.info("<<< Loading model from " + model_name + "_Model_Wrapper.pkl ... >>>")
    try:
        logger.info("<<< Loading model from " + model_name + ".h5 ... >>>")
        model = load_model(model_name + '.h5',
                           custom_objects=custom_objects,
                           compile=compile_model)
    except Exception as e:
        logger.info(str(e))
        # Load model structure
        logger.info("<<< Loading model from " + model_name + "_structure.json' ... >>>")
        model = model_from_json(open(model_name + '_structure.json').read(),
                                custom_objects=custom_objects)
        # Load model weights
        model.load_weights(model_name + '_weights.h5')

    # Load auxiliary models for optimized search
    if os.path.exists(model_name + '_init.h5') and os.path.exists(model_name + '_next.h5'):
        loading_optimized = 1
    elif os.path.exists(model_name + '_structure_init.json') and os.path.exists(
            model_name + '_weights_init.h5') and os.path.exists(model_name + '_structure_next.json') and \
            os.path.exists(model_name + '_weights_next.h5'):
        loading_optimized = 2
    else:
        loading_optimized = 0

    if loading_optimized == 1:
        logger.info("<<< Loading optimized model... >>>")
        model_init = load_model(model_name + '_init.h5',
                                custom_objects=custom_objects,
                                compile=False)
        model_next = load_model(model_name + '_next.h5',
                                custom_objects=custom_objects,
                                compile=False)

    elif loading_optimized == 2:
        # Load model structure
        logger.info("\t <<< Loading model_init from " + model_name + "_structure_init.json ... >>>")
        model_init = model_from_json(open(model_name + '_structure_init.json').read(),
                                     custom_objects=custom_objects)
        # Load model weights
        model_init.load_weights(model_name + '_weights_init.h5')
        # Load model structure
        logger.info("\t <<< Loading model_next from " + model_name + "_structure_next.json ... >>>")
        model_next = model_from_json(open(model_name + '_structure_next.json').read(),
                                     custom_objects=custom_objects)
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


def updateModel(model,
                model_path,
                update_num,
                reload_epoch=True,
                full_path=False,
                compile_model=False):
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
            model_name = os.path.join(model_path,
                                      "epoch_" + iteration)
        else:
            model_name = os.path.join(model_path,
                                      "update_" + iteration)

    logger.info("<<< Updating model " + model_name + " from " + model_path + " ... >>>")

    try:
        logger.info("<<< Updating model from " + model_name + ".h5 ... >>>")
        model.model.set_weights(load_model(model_name + '.h5',
                                           compile=False).get_weights())

    except Exception as e:
        logger.info(str(e))
        # Load model structure
        logger.info("<<< Failed -> Loading model from " + model_name + "_weights.h5' ... >>>")
        # Load model weights
        model.model.load_weights(model_name + '_weights.h5')

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
            logger.info("\t <<< Updating model_init from " + model_name + "_init.h5 ... >>>")
            model.model_init.set_weights(load_model(model_name + '_init.h5',
                                                    compile=False).get_weights())
            logger.info("\t <<< Updating model_next from " + model_name + "_next.h5 ... >>>")
            model.model_next.set_weights(load_model(model_name + '_next.h5',
                                                    compile=False).get_weights())
        elif loading_optimized == 2:
            logger.info("\t <<< Updating model_init from " + model_name + "_structure_init.json ... >>>")
            model.model_init.load_weights(model_name + '_weights_init.h5')
            # Load model structure
            logger.info("\t <<< Updating model_next from " + model_name + "_structure_next.json ... >>>")
            # Load model weights
            model.model_next.load_weights(model_name + '_weights_next.h5')

    logger.info("<<< Model updated in %0.6s seconds. >>>" % str(time.time() - t))
    return model


def transferWeights(old_model,
                    new_model,
                    layers_mapping):
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
