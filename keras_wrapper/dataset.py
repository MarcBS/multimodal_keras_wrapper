# coding=utf-8

from keras.utils import np_utils, generic_utils

import random
import math
import os
import copy
import ntpath
import fnmatch
from multiprocessing import Pool
import time
import threading
import logging
import re
from collections import Counter
from operator import add

import cPickle as pk
from scipy import misc
import numpy as np


# ------------------------------------------------------- #
#       SAVE/LOAD
#           External functions for saving and loading Dataset instances
# ------------------------------------------------------- #

def saveDataset(dataset, store_path):
    """
        Saves a backup of the current Dataset object.
    """
    store_path = store_path + '/Dataset_'+ dataset.name +'.pkl'
    
    if(not dataset.silence):
        logging.info("<<< Saving Dataset instance to "+ store_path +" ... >>>")
    
    pk.dump(dataset, open(store_path, 'wb'))
    
    if(not dataset.silence):
        logging.info("<<< Dataset instance saved >>>")


def loadDataset(dataset_path):
    """
        Loads a previously saved Dataset object.
    """
    logging.info("<<< Loading Dataset instance from "+ dataset_path +" ... >>>")
    
    dataset = pk.load(open(dataset_path, 'rb'))
    
    logging.info("<<< Dataset instance loaded >>>")
    return dataset

# ------------------------------------------------------- #
#       DATA BATCH GENERATOR CLASS
# ------------------------------------------------------- #
class Data_Batch_Generator(object):
    
    def __init__(self, set_split, net, dataset, num_iterations,
                 batch_size=50, 
                 normalize_images=False, 
                 data_augmentation=True, 
                 mean_substraction=True,
                 predict=False):
        
        self.set_split = set_split
        self.dataset = dataset
        self.net = net
        self.predict = predict
        # Several parameters
        self.params = {'batch_size': batch_size, 
                       'data_augmentation': data_augmentation,
                       'mean_substraction': mean_substraction,
                       'normalize_images': normalize_images,
                       'num_iterations': num_iterations}
    
    def generator(self):
            
        if(self.set_split == 'train' and not self.predict):
            data_augmentation = self.params['data_augmentation']
        else:
            data_augmentation = False

        it = 0
        while 1:

            if(self.set_split == 'train' and it%self.params['num_iterations']==0 and not self.predict):
                silence = self.dataset.silence
                self.dataset.silence = True
                self.dataset.shuffleTraining()
                self.dataset.silence = silence
            elif(it%self.params['num_iterations']==0 and not self.predict):
                self.dataset.resetCounters(set_name=self.set_split)
            it += 1
            
            # Checks if we are finishing processing the data split
            init_sample = (it-1)*self.params['batch_size']
            final_sample = it*self.params['batch_size']
            batch_size = self.params['batch_size']
            n_samples_split = eval("self.dataset.len_"+self.set_split)
            if final_sample >= n_samples_split:
                final_sample = n_samples_split
                batch_size = final_sample-init_sample
                it = 0
            
            # Recovers a batch of data
            if(self.predict):
                X_batch = self.dataset.getX(self.set_split, init_sample, final_sample, 
                                             normalization=self.params['normalize_images'],
                                             meanSubstraction=self.params['mean_substraction'],
                                             dataAugmentation=False)
                data = self.net.prepareData(X_batch, None)[0]
            else:
                X_batch, Y_batch = self.dataset.getXY(self.set_split, batch_size, 
                                             normalization=self.params['normalize_images'],
                                             meanSubstraction=self.params['mean_substraction'],
                                             dataAugmentation=data_augmentation)

                data = self.net.prepareData(X_batch, Y_batch)
            
            yield(data)

            
# ------------------------------------------------------- #
#       MAIN CLASS
# ------------------------------------------------------- #
class Dataset(object):
    """
        Class for defining instances of databases adapted for Keras. It includes several utility functions for easily managing
        data splits, image loading, mean calculation, etc.
    """
    
    def __init__(self, name, path, silence=False):
        
        # Dataset name
        self.name = name
        # Path to the folder where the images are stored
        self.path = path
        
        # If silence = False, some informative sentences will be printed while using the "Dataset" object instance
        self.silence = silence
        
        # Variable for storing external extra variables
        self.extra_variables = dict()
        
        ############################ Data loading parameters
        # Lock for threads synchronization
        self.__lock_read = threading.Lock()
        
        # Indicators for knowing if the data [X, Y] has been loaded for each data split
        self.loaded_train = [False, False]
        self.loaded_val = [False, False]
        self.loaded_test = [False, False]
        self.len_train = 0
        self.len_val = 0
        self.len_test = 0
        
        # Initlize dictionaries of samples
        self.X_train = dict()
        self.X_val = dict()
        self.X_test = dict()
        self.Y_train = dict()
        self.Y_val = dict()
        self.Y_test = dict()
        #################################################
        
        
        ############################ Parameters for managing all the inputs and outputs
        # List of identifiers for the inputs and outputs and their respective types 
        # (which will define the preprocessing applied)
        self.ids_inputs = []
        self.types_inputs = [] # see accepted types in self.__accepted_types_inputs
        self.optional_inputs = []

        self.ids_outputs = []
        self.types_outputs = [] # see accepted types in self.__accepted_types_outputs

        # List of implemented input and output data types
        self.__accepted_types_inputs = ['image', 'video', 'image-features', 'video-features', 'text', 'id']
        self.__accepted_types_outputs = ['categorical', 'binary', 'text', 'id']
        #    inputs/outputs with type 'id' is only used for storing external identifiers for your data 
        #    they will not be used in any way. IDs must be stored in text files with a single id per line
        
        # List of implemented input normalization functions
        self.__available_norm_im_vid = ['0-1']    # 'image' and 'video' only
        self.__available_norm_feat = ['L2']       # 'image-features' and 'video-features' only
        #################################################
        
        
        ############################ Parameters used for inputs/outputs of type 'text'
        self.extra_words = {'<pad>': 0, '<unk>': 1}    # extra words introduced in all vocabularies
        self.vocabulary = dict()     # vocabularies (words2idx and idx2words)
        self.max_text_len = dict()   # number of words accepted in a 'text' sample
        self.vocabulary_len = dict() # number of words in the vocabulary
        self.n_classes_text = dict() # only used for output text
        self.text_offset = dict()    # number of timesteps that the text is shifted (to the right)
        self.fill_text = dict()      # text padding mode

        #################################################
        
        
        ############################ Parameters used for inputs of type 'video'
        self.paths_frames = dict()
        self.max_video_len = dict() 
        #################################################
        
        ############################ Parameters used for inputs of type 'image-features' or 'video-features'
        self.features_lengths = dict()
        #################################################
        
        ############################ Parameters used for inputs of type 'image'
        # Image resize dimensions used for all the returned images
        self.img_size = dict()
        # Image crop dimensions for the returned images
        self.img_size_crop = dict()
        # Training mean image
        self.train_mean = dict()
        #################################################
        
        ############################ Parameters used for outputs of type 'categorical'
        self.classes = dict()
        self.dic_classes = dict()
        #################################################
        
        # Reset counters to start loading data in batches
        self.resetCounters()
        
    
    def shuffleTraining(self):
        """
            Applies a random shuffling to the training samples.
        """
        if(not self.silence):
            logging.info("Shuffling training samples.")
        
        # Shuffle
        num = self.len_train
        shuffled_order = random.sample([i for i in range(num)], num)
        
        # Process each input sample
        for id in self.X_train.keys():
            self.X_train[id] = [self.X_train[id][s] for s in shuffled_order]
        # Process each output sample
        for id in self.Y_train.keys():
            self.Y_train[id] = [self.Y_train[id][s] for s in shuffled_order]
            
        if(not self.silence):
            logging.info("Shuffling training done.")
    
    
    def keepTopOutputs(self, set_name, id_out, n_top):
        self.__checkSetName(set_name)
        
        if id_out not in self.ids_outputs:
            raise Exception("The parameter 'id_out' must specify a valid id for an output of the dataset.")
        
        #type_out = self.types_outputs(self.ids_outputs.index(id_out))
        #if type_out != 'text':
        #    raise Exception("This method is only applicable to outputs of type 'text'.")
        
        logging.info('Keeping top '+str(n_top)+' outputs from the '+set_name+' set and removing the rest.')
        
        # Sort outputs by number of occurrences
        exec('samples = self.Y_'+set_name)
        count = Counter(samples[id_out])
        most_frequent = sorted(count.items(), key=lambda x:x[1], reverse=True)[:n_top]
        most_frequent = [m[0] for m in most_frequent]
        
        # Select top samples
        kept = []
        for i, s in enumerate(samples[id_out]):
            if s in most_frequent:
                kept.append(i)
                
        # Remove non-top samples    
        # Inputs
        exec('ids = self.X_'+set_name+'.keys()')
        for id in ids:
            exec('self.X_'+set_name+'[id] = [self.X_'+set_name+'[id][k] for k in kept]')
        # Outputs
        exec('ids = self.Y_'+set_name+'.keys()')
        for id in ids:
            exec('self.Y_'+set_name+'[id] = [self.Y_'+set_name+'[id][k] for k in kept]')
        new_len = len(samples[id_out])
        exec('self.len_'+set_name+' = new_len')
        
        self.__checkLengthSet(set_name)
        
        logging.info(str(new_len)+' samples remaining after removal.')
    
    
    # ------------------------------------------------------- #
    #       GENERAL SETTERS
    #           classes list, train, val and test set, etc.
    # ------------------------------------------------------- #
    
    def resetCounters(self, set_name="all"):
        """
            Resets some basic counter indices for the next samples to read.
        """
        if(set_name == "all"):
            self.last_train = 0
            self.last_val = 0
            self.last_test = 0
        else:
            self.__checkSetName(set_name)
            exec('self.last_'+set_name+'=0')
            
    def setSilence(self, silence):
        """
            Changes the silence mode of the 'Dataset' instance.
        """
        self.silence = silence
        
        
    def setListGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True, type='image', id='image'):
        """
            Deprecated
        """
        logging.info("WARNING: The method setListGeneral() is deprecated, consider using setInputGeneral() instead.")
        self.setInputGeneral(path_list, split, shuffle, type, id)
    
    def setInputGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True, type='image', id='image'):
        """ 
            DEPRECATED
        
            Loads a single list of samples from which train/val/test divisions will be applied. 
            
            :param path_list: path to the text file with the list of images.
            :param split: percentage of images used for [training, validation, test].
            :param shuffle: whether we are randomly shuffling the input samples or not.
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
        """
        
        raise NotImplementedError("This function is deprecated use setInput instead.")

        if(sum(split) != 1):
            raise Exception('"split" values must sum 1.')
        if(len(split) != 3):
            raise Exception('The length of "split" must be equal to 3.')
        
        # Read list
        set = []
        with open(path_list, 'r') as list_:
            for line in list_:
                set.append(line.rstrip('\n'))
        nSamples = len(set)
        
        # Randomize list of samples
        set_num = [i for i in range(nSamples)]
        if(shuffle):
            set_num = random.sample(set_num, nSamples)
        
        # Insert type and id of input data
        if(id not in self.ids_inputs):
            self.ids_inputs.append(id)
            if(type not in self.__accepted_types_inputs):
                raise NotImplementedError('The input type '+type+' is not implemented. The list of valid types are the following: '+str(self.__accepted_types_inputs))
            self.types_inputs.append(type)
        else:
            raise Exception('An input with id '+id+' is already loaded into the Database instance.')
        
        offset = 0
        order = ['train', 'val', 'test']
        set_split = []
        for i in range(len(split)):
            last = int(math.ceil(nSamples*split[i]))
            set_split.append(set_num[offset:offset+last])
            offset += last
            
            # Insert into the corresponding list
            if(len(set_split[i]) > 0):
                self.__setInput([set[elem] for elem in set_split[i]], order[i], id=id)
        
    
    
    def setList(self, path_list, set_name, type='image', id='image'):
        """
            DEPRECATED
        """
        logging.info("WARNING: The method setList() is deprecated, consider using setInput() instead.")
        self.setInput(path_list, set_name, type, id)
    
    
    def setInput(self, path_list, set_name, type='image', id='image', repeat_set=1, required=True,
                 img_size=[256, 256, 3], img_size_crop=[227, 227, 3],                     # 'image' / 'video'
                 max_text_len=35, tokenization='tokenize_basic',offset=0, fill='end',     # 'text'
                 build_vocabulary=False, max_words=0,
                 feat_len = 1024,                                                         # 'image-features' / 'video-features'
                 max_video_len=26                                                         # 'video'
                 ):
        """
            Loads a list of samples which can contain all samples from the 'train', 'val', or
            'test' sets (specified by set_name).
            
            # General parameters
            
            :param path_list: can either be a path to a text file containing the paths to the images or a python list of paths
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
            :param repeat_set: repats the inputs given (useful when we have more outputs than inputs). Int or array of ints.
            :param required: flag for optional inputs

            
            # 'image'-related parameters
            
            :param img_size: size of the input images (any input image will be resized to this)
            :param img_size_crop: size of the cropped zone (when dataAugmentation=False the central crop will be used)
            
            
            # 'text'-related parameters
            
            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text').
            :param max_text_len: maximum text length, the rest of the data will be padded with 0s (only applicable if the output data is of type 'text').
            :param max_words: a maximum of 'max_words' words from the whole vocabulary will be chosen by number or occurrences
            :param offset: number of timesteps that the text is shifted to the right (for *_cond models)
            :param fill: select whether padding before or after the sequence

            # 'image-features' and 'video-features'- related parameters
            
            :param feat_len: length of the feature vectors if we are using types 'image-features' or 'video-features'
            
            
            # 'video'-related parameters
            
            :param max_video_len: maximum video length, the rest of the data will be padded with 0s (only applicable if the input data is of type 'video' or video-features').
        """
        self.__checkSetName(set_name)
        
        # Insert type and id of input data
        keys_X_set = eval('self.X_'+set_name+'.keys()')
        if(id not in self.ids_inputs):
            self.ids_inputs.append(id)
            self.types_inputs.append(type)
            if not required:
                self.optional_inputs.append(id)
        elif id in keys_X_set:
            raise Exception('An input with id "'+id+'" is already loaded into the Database.')

        if(type not in self.__accepted_types_inputs):
            raise NotImplementedError('The input type "'+type+'" is not implemented. The list of valid types are the following: '+str(self.__accepted_types_inputs))
        
        # Proprocess the input data depending on its type
        if(type == 'image'):
            data = self.preprocessImages(path_list, id, img_size, img_size_crop)
        elif(type == 'video'):
            data = self.preprocessVideos(path_list, id, set_name, max_video_len, img_size, img_size_crop)
        elif(type == 'text'):
            data = self.preprocessText(path_list, id, tokenization, build_vocabulary, max_text_len, max_words, offset, fill)
        elif(type == 'image-features'):
            data = self.preprocessFeatures(path_list, id, feat_len)
        elif(type == 'video-features'):
            data = self.preprocessVideos(path_list, id, set_name, max_video_len, img_size, img_size_crop)
        elif(type == 'id'):
            data = self.preprocessIDs(path_list, id)
       
        if(repeat_set > 1): 
            data = list(np.repeat(data,repeat_set))
        
        self.__setInput(data, set_name, type, id)
        
    
    def __setInput(self, set, set_name, type, id):
        exec('self.X_'+set_name+'[id] = set')
        exec('self.loaded_'+set_name+'[0] = True')
        exec('self.len_'+set_name+' = len(set)')
        if id not in self.optional_inputs:
            self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set inputs of type "'+type+'" with id "'+id+'" and length '+ str(eval('self.len_'+set_name)) + '.')
        
    
    
    def setLabels(self, labels_list, set_name, type='categorical', id='label'):
        """
            DEPRECATED
        """
        logging.info("WARNING: The method setLabels() is deprecated, consider using () instead.")
        self.setOutput(self, labels_list, set_name, type, id)
    
    def setOutput(self, path_list, set_name, type='categorical', id='label', repeat_set=1,
                  tokenization='tokenize_basic', max_text_len=0, offset=0, fill='end',                         # 'text'
                  build_vocabulary=False, max_words=0):
        """
            Loads a set of output data, usually (type=='categorical') referencing values in self.classes (starting from 0)
            
            # General parameters
            
            :param path_list: can either be a path to a text file containing the labels or a python list of labels.
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test').
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_outputs).
            :param id: identifier of the input data loaded.
            :param repeat_set: repeats the outputs given (useful when we have more inputs than outputs). Int or array of ints.
            
            # 'text'-related parameters
            
            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text').
            :param max_text_len: maximum text length, the rest of the data will be padded with 0s (only applicable if the output data is of type 'text') Set to 0 if the whole sentence will be used as an output class.
            :param max_words: a maximum of 'max_words' words from the whole vocabulary will be chosen by number or occurrences
            :param offset: number of timesteps that the text is shifted to the right (for *_cond models)
            :param fill: select whether padding before or after the sequence

        """
        self.__checkSetName(set_name)
        
        # Insert type and id of output data
        keys_Y_set = eval('self.Y_'+set_name+'.keys()')
        if(id not in self.ids_outputs):
            self.ids_outputs.append(id)
            self.types_outputs.append(type)
        elif id in keys_Y_set:
            raise Exception('An input with id "'+id+'" is already loaded into the Database.')
        
        if(type not in self.__accepted_types_outputs):
            raise NotImplementedError('The output type "'+type+'" is not implemented. The list of valid types are the following: '+str(self.__accepted_types_outputs))

        # Proprocess the output data depending on its type
        if(type == 'categorical'):
            data = self.preprocessCategorical(path_list)
        elif(type == 'text'):
            data = self.preprocessText(path_list, id, tokenization, build_vocabulary, max_text_len, max_words, offset, fill)
        elif(type == 'binary'):
            data = self.preprocessBinary(path_list)
        elif(type == 'id'):
            data = self.preprocessIDs(path_list, id)
            
        if(repeat_set > 1):  
            data = list(np.repeat(data,repeat_set))

        self.__setOutput(data, set_name, type, id)
    
    
    def __setOutput(self, labels, set_name, type, id):
        exec('self.Y_'+set_name+'[id] = labels')
        exec('self.loaded_'+set_name+'[1] = True')
        exec('self.len_'+set_name+' = len(labels)')
        self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set outputs of type "'+type+'" with id "'+id+'" and length '+ str(eval('self.len_'+set_name)) + '.')
           
        
    # ------------------------------------------------------- #
    #       TYPE 'categorical' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def setClasses(self, path_classes, id):
        """
            Loads the list of classes of the dataset.
            Each line must contain a unique identifier of the class.
        """

        if(isinstance(path_classes, str) and os.path.isfile(path_classes)):
            classes = []
            with open(path_classes, 'r') as list_:
                for line in list_:
                    classes.append(line.rstrip('\n'))
            self.classes[id] = classes
        elif isinstance(path_classes, list):
            self.classes[id] = path_classes
        else:
            raise Exception('Wrong type for "path_classes". It must be a path to a text file with the classes or an instance of the class list.')
        
        self.dic_classes[id] = dict()
        for c in range(len(self.classes[id])):
            self.dic_classes[id][self.classes[id][c]] = c
        
        if(not self.silence):
            logging.info('Loaded classes list with ' + str(len(self.classes[id])) + " different labels.")
    
    def preprocessCategorical(self, labels_list):
        
        if(isinstance(labels_list, str) and os.path.isfile(labels_list)):
            labels = []
            with open(labels_list, 'r') as list_:
                for line in list_:
                    labels.append(int(line.rstrip('\n')))
        elif(isinstance(labels_list, list)):
            labels = labels_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file with the labels or an instance of the class list.')
        
        return labels
    
    # ------------------------------------------------------- #
    #       TYPE 'binary' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
       
    def preprocessBinary(self, labels_list):
        
        if(isinstance(labels_list, list)):
            labels = labels_list
        else:
            raise Exception('Wrong type for "path_list". It must be an instance of the class list.')
        
        return labels
    
    # ------------------------------------------------------- #
    #       TYPE 'features' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessFeatures(self, path_list, id, feat_len):
        
        # file with a list, each line being a path to a .npy file with a feature vector
        if(isinstance(path_list, str) and os.path.isfile(path_list)): 
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    #data.append(np.fromstring(line.rstrip('\n'), sep=','))
                    data.append(line.rstrip('\n'))
        elif(isinstance(path_list, list)):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file. Each line must contain a path to a .npy file storing a feature vector. Alternatively "path_list" can be an instance of the class list.')
        
        self.features_lengths[id] = feat_len

        return data
    
    
    def loadFeatures(self, X, feat_len, normalization_type='L2', normalization=False, loaded=False, external=False):
        
        if(normalization and normalization_type not in self.__available_norm_feat):
            raise NotImplementedError('The chosen normalization type '+ normalization_type +' is not implemented for the type "image-features" and "video-features".')
        
        n_batch = len(X)
        features = np.zeros((n_batch, feat_len))
        
        for i, feat in enumerate(X):
            if(not external):
                feat = self.path +'/'+ feat

            # Check if the filename includes the extension
            feat = np.load(feat)
            
            if(normalization):
                if normalization_type == 'L2':
                    feat = feat / np.linalg.norm(feat,ord=2)
                    
            features[i] = feat
            
        return np.array(features)
    
    # ------------------------------------------------------- #
    #       TYPE 'text' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessText(self, annotations_list, id, tokenization, build_vocabulary, max_text_len, max_words, offset, fill):
        
        sentences = []
        if(isinstance(annotations_list, str) and os.path.isfile(annotations_list)):
            with open(annotations_list, 'r') as list_:
                for line in list_:
                    sentences.append(line.rstrip('\n'))
        else:
            raise Exception('Wrong type for "annotations_list". It must be a path to a text file with the sentences or a list of sentences. '
                            'It currently is: %s'%(str(annotations_list)))
            
        # Check if tokenization method exists
        if(hasattr(self, tokenization)):
            tokfun = eval('self.'+tokenization)
        else:
            raise Exception('Tokenization procedure "'+ tokenization +'" is not implemented.')
            
        # Tokenize sentences
        if(max_text_len != 0): # will only tokenize if we are not using the whole sentence as a class
            for i in range(len(sentences)):
                sentences[i] = tokfun(sentences[i])
    
        # Build vocabulary
        if(build_vocabulary):
            self.build_vocabulary(sentences, id, tokfun, max_text_len != 0, n_words=max_words)
        
        if(not id in self.vocabulary):
            raise Exception('The dataset must include a vocabulary with id "'+id+'" in order to process the type "text" data. Set "build_vocabulary" to True if you want to use the current data for building the vocabulary.')
    
        # Store max text len
        self.max_text_len[id] = max_text_len
        self.n_classes_text[id] = len(self.vocabulary[id]['words2idx'])
        self.text_offset[id] = offset
        self.fill_text[id] = fill

        return sentences
    
    
    def build_vocabulary(self, captions, id, tokfun, do_split, n_words=0):
        """
            Vocabulary builder for data of type 'text'
        """
        if(not self.silence):
            logging.info("Creating vocabulary for data with id '"+id+"'.")
        
        counters = []
        sentence_counts = []
        counter = Counter()
        sentence_count = 0
        for line in captions:
            if(do_split):
                #tokenized = tokfun(line)
                #words = tokenized.strip().split(' ')
                words = line.strip().split(' ')
                words_low = map(lambda x: x.lower(), words)
                counter.update(words_low)
            else:
                counter.update([line])
            sentence_count += 1
            
        if(not do_split and not self.silence):
            logging.info('Using whole sentence as a single word.')
            
        counters.append(counter)
        sentence_counts.append(sentence_count)
        #logging.info("\t %d unique words in %d sentences with a total of %d words." %
        #      (len(counter), sentence_count, sum(counter.values())))

        combined_counter = reduce(add, counters)
        if(not self.silence):
            logging.info("\t Total: %d unique words in %d sentences with a total of %d words." %
              (len(combined_counter), sum(sentence_counts),sum(combined_counter.values())))


        if n_words > 0:
            vocab_count = combined_counter.most_common(n_words - len(self.extra_words))
            if(not self.silence):
                logging.info("Creating dictionary of %s most common words, covering "
                        "%2.1f%% of the text."
                        % (n_words,
                           100.0 * sum([count for word, count in vocab_count]) /
                           sum(combined_counter.values())))
        else:
            if(not self.silence):
                logging.info("Creating dictionary of all words")
            vocab_count = counter.most_common()

        dictionary = {}
        for i, (word, count) in enumerate(vocab_count):
            dictionary[word] = i + len(self.extra_words)
                
        for w,k in self.extra_words.iteritems():
            dictionary[w] = k
        
        self.vocabulary[id] = dict()
        self.vocabulary[id]['words2idx'] = dictionary
        inv_dictionary = {v: k for k, v in dictionary.items()}
        self.vocabulary[id]['idx2words'] = inv_dictionary
        
        self.vocabulary_len[id] = len(vocab_count) + len(self.extra_words)


    def loadText(self, X, vocabularies, max_len, offset, fill):
        """
            Text encoder. Transforms samples from a text representation into a numerical one.
            If fill=='start' the resulting vector will be filled with 0s at the beginning, 
            if fill=='end' it will be filled with 0s at the end.
        """
        vocab = vocabularies['words2idx']
        n_batch = len(X)
        if(max_len == 0): # use whole sentence as class
            X_out = np.zeros((n_batch)).astype('int32')
            for i in range(n_batch):
                w = X[i]
                if w in vocab:
                    X_out[i] = vocab[w]
                else:
                    X_out[i] = vocab['<unk>']
            
        else: # process text as a sequence of words
            X_out = np.ones((n_batch, max_len)).astype('int32') * self.extra_words['<pad>']
        
            # fills text vectors with each word (fills with 0s or removes remaining words w.r.t. max_len)
            for i in range(n_batch):
                x = X[i].split(' ')
                len_j = len(x)
                if(fill=='start'):
                    offset_j = max_len - len_j
                else:
                    offset_j = 0
                    len_j = min(len_j, max_len)
                if offset_j < 0:
                    len_j = len_j + offset_j
                    offset_j = 0
                for j, w in zip(range(len_j),x[:len_j]):
                    if w in vocab:
                        X_out[i,j+offset_j] = vocab[w]
                    else:
                        X_out[i,j+offset_j] = vocab['<unk>']

                if offset > 0 and fill == 'start': # Move the text to the left
                    X_out[i] = np.append(X_out[i, offset:], [vocab['<pad>']]*offset)
                if offset > 0 and fill == 'end': # Move the text to the right
                    X_out[i] = np.append([vocab['<pad>']]*offset, X_out[i, :-offset])

        return X_out

    
    # ------------------------------------------------------- #
    #       Tokenization functions
    # ------------------------------------------------------- #

    def tokenize_basic(self, caption, lowercase=True):
        """
            Basic tokenizer for the input/output data of type 'text':
                Splits punctuation
                Lowercase
        """
        punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
        def processPunctuation(inText):
            outText = inText
            for p in punct:
                outText = outText.replace(p, ' ' + p + ' ')
            return outText
        resAns = caption.lower() if lowercase else caption
        resAns = resAns.replace('\n', ' ')
        resAns = resAns.replace('\t', ' ')
        resAns = processPunctuation(resAns)
        resAns = resAns.replace('  ', ' ')
        return resAns


    def tokenize_questions(self, caption):
        """
            Basic tokenizer for the input/output data of type 'text'
        """
        contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
                "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t",
                "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve",
                "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve",
                "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve",
                "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve",
                "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t",
                "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", "mustnt": "mustn’t",
                "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t",
                "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t",
                "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve",
                "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve",
                "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve",
                "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d",
                "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll",
                "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve",
                "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s",
                "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve", "therere": "there’re",
                "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve",
                "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t",
                "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll",
                "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered":
                    "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d", "whod’ve": "who’d’ve",
                "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll",
                "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t",
                "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll",
                "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve",
                "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll",
                "youre": "you’re", "youve": "you’ve"}
        punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
        commaStrip = re.compile("(\d)(\,)(\d)")
        periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
             'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
        articles = ['a', 'an', 'the']

        def processPunctuation(inText):
            outText = inText
            for p in punct:
                if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
                    outText = outText.replace(p, '')
                else:
                    outText = outText.replace(p, ' ')
            outText = periodStrip.sub("", outText, re.UNICODE)
            return outText

        def processDigitArticle(inText):
            outText = []
            tempText = inText.lower().split()
            for word in tempText:
                word = manualMap.setdefault(word, word)
                if word not in articles:
                    outText.append(word)
                else:
                    pass
            for wordId, word in enumerate(outText):
                if word in contractions:
                    outText[wordId] = contractions[word]
            outText = ' '.join(outText)
            return outText

        resAns = caption.lower()
        resAns = resAns.replace('\n', ' ')
        resAns = resAns.replace('\t', ' ')
        resAns = resAns.strip()
        resAns = processPunctuation(resAns.decode("utf-8").encode("utf-8"))
        resAns = processDigitArticle(resAns)

        return resAns
    
    
    # ------------------------------------------------------- #
    #       TYPE 'video' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessVideos(self, path_list, id, set_name, max_video_len, img_size, img_size_crop):
        
        if(isinstance(path_list, list) and len(path_list) == 2):
            # path to all images in all videos
            data = []
            with open(path_list[0], 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
            # frame counts
            counts_frames = []
            with open(path_list[1], 'r') as list_:
                for line in list_:
                    counts_frames.append(int(line.rstrip('\n')))
            
            if(id not in self.paths_frames):
                self.paths_frames[id] = dict()
            self.paths_frames[id][set_name] = data
            self.max_video_len[id] = max_video_len
            self.img_size[id] = img_size
            self.img_size_crop[id] = img_size_crop
        else:
            raise Exception('Wrong type for "path_list". It must be a list containing two paths: a path to a text file with the paths to all images in all videos in [0] and a path to another text file with the number of frames of each video in each line in [1] (which will index the paths in the first file).')
        return counts_frames
    
    
    def loadVideos(self, n_frames, id, last, set_name, max_len, normalization_type, normalization, meanSubstraction, dataAugmentation):
        n_videos = len(n_frames)
        V = np.zeros((n_videos, max_len*3, self.img_size_crop[id][0], self.img_size_crop[id][1]))
        
        idx = [0 for i in range(n_videos)]
        # recover all indices from image's paths of all videos
        for v in range(n_videos):
            this_last = last+v                
            if this_last >= n_videos:
                v = this_last%n_videos
                this_last = v
            idx[v] = int(sum(eval('self.X_'+set_name+'[id][:this_last]')))
        
        # load images from each video
        for enum, (n, i) in enumerate(zip(n_frames, idx)):
            paths = self.paths_frames[id][set_name][i:i+n]
            # returns numpy array with dimensions (batch, channels, height, width)
            images = self.loadImages(paths, id, normalization_type, normalization, meanSubstraction, dataAugmentation)
            # fills video matrix with each frame (fills with 0s or removes remaining frames w.r.t. max_len)
            len_j = images.shape[0]
            offset_j = max_len - len_j
            if offset_j < 0:
                len_j = len_j + offset_j
                offset_j = 0
            for j in range(len_j):
                V[enum, (j+offset_j)*3:(j+offset_j+1)*3] = images[j]
        
        return V
    
    
    # ------------------------------------------------------- #
    #       TYPE 'id' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessIDs(self, path_list, id):
        
        logging.info('WARNING: inputs or outputs with type "id" will not be treated in any way by the dataset.')
        if(isinstance(path_list, str) and os.path.isfile(path_list)): # path to list of IDs
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
        elif(isinstance(path_list, list)):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file with an id in each line or an instance of the class list with an id in each position.')
    
        return data
    
    # ------------------------------------------------------- #
    #       TYPE 'image' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessImages(self, path_list, id, img_size, img_size_crop):
        
        if(isinstance(path_list, str) and os.path.isfile(path_list)): # path to list of images' paths
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
        elif(isinstance(path_list, list)):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a text file with an image path in each line or an instance of the class list with an image path in each position.')
            
        self.img_size[id] = img_size
        self.img_size_crop[id] = img_size_crop
            
        # Tries to load a train_mean file from the dataset folder if exists
        mean_file_path = self.path+'/train_mean'
        for s in range(len(self.img_size[id])):
            mean_file_path += '_'+str(self.img_size[id][s])
        mean_file_path += '_'+id+'_.jpg'
        if(os.path.isfile(mean_file_path)):
            self.setTrainMean(mean_file_path, id)
            
        return data
       
    
    def setTrainMean(self, mean_image, id, normalization=False):
        """
            Loads a pre-calculated training mean image, 'mean_image' can either be:
            
            - numpy.array (complete image)
            - list with a value per channel
            - string with the path to the stored image.
            
            :param id: identifier of the type of input whose train mean is being introduced.
        """
        if(isinstance(mean_image, str)):
            if(not self.silence):
                logging.info("Loading train mean image from file.")
            mean_image = misc.imread(mean_image)
        elif(isinstance(mean_image, list)):
            mean_image = np.array(mean_image)
        self.train_mean[id] = mean_image.astype(np.float32)
        
        if(normalization):
            self.train_mean[id] = self.train_mean[id]/255.0
            
        if(self.train_mean[id].shape != tuple(self.img_size[id])):
            if(len(self.train_mean[id].shape) == 1 and self.train_mean[id].shape[0] == self.img_size[id][2]):
                if(not self.silence):
                    logging.info("Converting input train mean pixels into mean image.")
                mean_image = np.zeros(tuple(self.img_size[id]))
                for c in range(self.img_size[id][2]):
                    mean_image[:, :, c] = self.train_mean[id][c]
                self.train_mean[id] = mean_image
            else:
                logging.warning("The loaded training mean size does not match the desired images size.\nChange the images size with setImageSize(size) or recalculate the training mean with calculateTrainMean().")
    
    def calculateTrainMean(self, id):
        """
            Calculates the mean of the data belonging to the training set split in each channel.
        """
        calculate = False
        if(not isinstance(self.train_mean[id], np.ndarray)):
            calculate = True
        elif(self.train_mean[id].shape != tuple(self.img_size[id])):
            calculate = True
            if(not self.silence):
                logging.warning("The loaded training mean size does not match the desired images size. Recalculating mean...")
            
        if(calculate):
            if(not self.silence):
                logging.info("Start training set mean calculation...")
            
            I_sum = np.zeros(self.img_size[id], dtype=np.longlong)
            
            # Load images in batches and sum all of them
            init = 0
            batch = 200
            for final in range(batch, self.len_train, batch):
                I = self.getX('train', init, final, resizeImage=True, meanSubstraction=False)
                for im in I:
                    I_sum += im
                if(not self.silence):
                    logging.info("\tProcessed "+str(final)+'/'+str(self.len_train)+' images...')
                init = final
            I = self.getX('train', init, self.len_train, resizeImage=True, meanSubstraction=False)
            for im in I:
                I_sum += im
            if(not self.silence):
                logging.info("\tProcessed "+str(final)+'/'+str(self.len_train)+' images...')
            
            # Mean calculation
            self.train_mean[id] = I_sum/self.len_train
            
            # Store the calculated mean
            mean_name = self.path+'/train_mean'
            for s in range(len(self.img_size[id])):
                mean_name += '_'+str(self.img_size[id][s])
            mean_name += '_'+id+'_.jpg'
            store_path = self.path+'/'+mean_name
            misc.imsave(store_path, self.train_mean[id])
            
            self.train_mean[id] = self.train_mean[id].astype(np.float32)/255.0
            
            if(not self.silence):
                logging.info("Image mean stored in "+ store_path)
            
        # Return the mean
        return self.train_mean[id]
    
        
    def loadImages(self, images, id, normalization_type='0-1', normalization=False, meanSubstraction=True, dataAugmentation=True, external=False, loaded=False):
        """
            Loads a set of images from disk.
            
            :param images : list of image string names or list of matrices representing images
            :param normalization_type: type of normalization applied
            :param normalization : whether we applying a 0-1 normalization to the images
            :param meanSubstraction : whether we are removing the training mean
            :param dataAugmentation : whether we are applying dataAugmentatino (random cropping and horizontal flip)
            :param external : if True the images will be loaded from an external database, in this case the list of images must be absolute paths
            :param loaded : set this option to True if images is a list of matricies instead of a list of strings
        """
        # Check if the chosen normalization type exists
        if(normalization and normalization_type not in self.__available_norm_im_vid):
            raise NotImplementedError('The chosen normalization type '+ normalization_type +' is not implemented for the type "image" and "video".')
        
        # Prepare the training mean image
        if(meanSubstraction): # remove mean
            #if(not isinstance(self.train_mean[id], np.ndarray)):
            if(id not in self.train_mean):
                raise Exception('Training mean is not loaded or calculated yet for the input with id "'+id+'".')
            train_mean = copy.copy(self.train_mean[id])
            # Take central part
            left = np.round(np.divide([self.img_size[id][0]-self.img_size_crop[id][0], self.img_size[id][1]-self.img_size_crop[id][1]], 2.0))
            right = left + self.img_size_crop[id][0:2]
            train_mean = train_mean[left[0]:right[0], left[1]:right[1], :]
            # Transpose dimensions
            if(len(self.img_size[id]) == 3): # if it is a 3D image
                # Convert RGB to BGR
                if(self.img_size[id][2] == 3): # if has 3 channels
                    aux = copy.copy(train_mean)
                    train_mean[:,:,0] = aux[:,:,2]
                    train_mean[:,:,2] = aux[:,:,0]
                train_mean = np.transpose(train_mean, (2, 0, 1))
            else:
                pass
            
        prob_flip_horizontal = 0.5
        prob_flip_vertical = 0.0
        nImages = len(images)
        
        type_imgs = np.float32
        if(len(self.img_size[id]) == 3):
            I = np.zeros([nImages]+[self.img_size_crop[id][2]]+self.img_size_crop[id][0:2], dtype=type_imgs)
        else:
            I = np.zeros([nImages]+self.img_size_crop[id], dtype=type_imgs)
            
        ''' Process each image separately '''
        for i in range(nImages):
            im = images[i]
            
            if(not loaded):
                if(not external):
                    im = self.path +'/'+ im
                
                # Check if the filename includes the extension
                [path, filename] = ntpath.split(im)
                [filename, ext] = os.path.splitext(filename)
                
                # If it doesn't then we find it
                if(not ext):
                    filename = fnmatch.filter(os.listdir(path), filename+'*')
                    if(not filename):
                        raise Exception('Non existent image '+ im)
                    else:
                        im = path+'/'+filename[0]
                
                # Read image
                im = misc.imread(im)
            
            # Resize and convert to RGB (if in greyscale)
            im = misc.imresize(im, tuple(self.img_size[id])).astype(type_imgs)
            if(len(self.img_size[id]) == 3 and len(im.shape) < 3): # convert grayscale into RGB (or any other channel#)
                nCh = self.img_size[id][2]
                rgb_im = np.empty((im.shape[0], im.shape[1], nCh), dtype=np.float32)
                for c in range(nCh):
                    rgb_im[:,:,c] = im
                im = rgb_im
                
            # Normalize
            if(normalization):
                if(normalization_type == '0-1'):
                    im = im/255.0
                
            # Data augmentation
            if(not dataAugmentation):
                # Take central image
                left = np.round(np.divide([self.img_size[id][0]-self.img_size_crop[id][0], self.img_size[id][1]-self.img_size_crop[id][1]], 2))
                right = left + self.img_size_crop[id][0:2]
                im = im[left[0]:right[0], left[1]:right[1], :]
            else:
                # Take random crop
                margin = [self.img_size[id][0]-self.img_size_crop[id][0], self.img_size[id][1]-self.img_size_crop[id][1]]
                left = random.sample([k_ for k_ in range(margin[0])], 1) + random.sample([k for k in range(margin[1])], 1)
                right = np.add(left, self.img_size_crop[id][0:2])
                im = im[left[0]:right[0], left[1]:right[1], :]
                
                # Randomly flip (with a certain probability)
                flip = random.random()
                if(flip < prob_flip_horizontal): # horizontal flip
                    im = np.fliplr(im)
                flip = random.random()
                if(flip < prob_flip_vertical): # vertical flip
                    im = np.flipud(im)
            
            # Permute dimensions
            if(len(self.img_size[id]) == 3):
                # Convert RGB to BGR
                if(self.img_size[id][2] == 3): # if has 3 channels
                    aux = copy.copy(im)
                    im[:,:,0] = aux[:,:,2]
                    im[:,:,2] = aux[:,:,0]
                im = np.transpose(im, (2, 0, 1))
            else:
                pass
            
            # Substract training images mean
            if(meanSubstraction): # remove mean
                im = im - train_mean
            
            I[i] = im

        return I
    
    
    def getClassID(self, class_name, id):
        """
            Returns the class id (int) for a given class string.
        """
        return self.dic_classes[id][class_name]
    
    
    # ------------------------------------------------------- #
    #       GETTERS
    #           [X,Y] pairs or X only
    # ------------------------------------------------------- #
        
    def getX(self, set_name, init, final, normalization_type='0-1', normalization=False,
             meanSubstraction=True, dataAugmentation=True, debug=False):
        """
            Gets all the data samples stored between the positions init to final
            
            :param set_name: 'train', 'val' or 'test' set
            :param init: initial position in the corresponding set split. Must be bigger or equal than 0 and smaller than final.
            :param final: final position in the corresponding set split.
            :param debug: if True all data will be returned without preprocessing
            
            
            # 'image', 'video', 'image-features' and 'video-features'-related parameters
            
            :param normalization: indicates if we want to normalize the data.
            
            
            # 'image-features' and 'video-features'-related parameters
            
            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.
            
            
            # 'image' and 'video'-related parameters
            
            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)
        """
        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)
        
        if(final > eval('self.len_'+set_name)):
            raise Exception('"final" index must be smaller than the number of samples in the set.')
        if(init < 0):
            raise Exception('"init" index must be equal or greater than 0.')
        if(init >= final):
            raise Exception('"init" index must be smaller than "final" index.')
        
        X = []
        for id_in,type_in in zip(self.ids_inputs, self.types_inputs):
            x = eval('self.X_'+set_name+'[id_in][init:final]')
            
            if(not debug):
                if(type_in == 'image'):
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif(type_in == 'video'):
                    x = self.loadVideos(x, id_in, last, set_name, self.max_video_len[id_in], 
                                        normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif(type_in == 'text'):
                    x = self.loadText(x, self.vocabulary[id_in], 
                                      self.max_text_len[id_in], self.text_offset[id_in], 
                                      fill=self.fill_text[id_in])
                elif(type_in == 'image-features'):
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization)
                elif(type_in == 'video-features'):
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization)
            X.append(x)
        
        return X
        
        
    def getXY(self, set_name, k, normalization_type='0-1', normalization=False, meanSubstraction=True, dataAugmentation=True, debug=False):
        """
            Gets the [X,Y] pairs for the next 'k' samples in the desired set.
            
            :param set_name: 'train', 'val' or 'test' set
            :param k: number of consecutive samples retrieved from the corresponding set.
            :param debug: if True all data will be returned without preprocessing
            
            
            # 'image', 'video', 'image-features' and 'video-features'-related parameters
            
            :param normalization: indicates if we want to normalize the data.
            
            
            # 'image-features' and 'video-features'-related parameters
            
            :param normalization_type: indicates the type of normalization applied. See available types in self.__available_norm_im_vid for 'image' and 'video' and self.__available_norm_feat for 'image-features' and 'video-features'.
            
            
            # 'image' and 'video'-related parameters
            
            :param meanSubstraction: indicates if we want to substract the training mean from the returned images (only applicable if normalization=True)
            :param dataAugmentation: indicates if we want to apply data augmentation to the loaded images (random flip and cropping)
        """
        self.__checkSetName(set_name)
        self.__isLoaded(set_name, 0)
        self.__isLoaded(set_name, 1)
        
        [new_last, last, surpassed] = self.__getNextSamples(k, set_name)
        
        # Recover input samples
        X = []
        for id_in, type_in in zip(self.ids_inputs, self.types_inputs):
            if(surpassed):
                x = eval('self.X_'+set_name+'[id_in][last:]') + eval('self.X_'+set_name+'[id_in][0:new_last]')
            else:
                x = eval('self.X_'+set_name+'[id_in][last:new_last]')
                
            #if(set_name=='val'):
            #    logging.info(x)
                
            # Pre-process inputs
            if(not debug):
                if(type_in == 'image'):
                    x = self.loadImages(x, id_in, normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif(type_in == 'video'):
                    x = self.loadVideos(x, id_in, last, set_name, self.max_video_len[id_in], 
                                        normalization_type, normalization, meanSubstraction, dataAugmentation)
                elif(type_in == 'text'):
                    x = self.loadText(x, self.vocabulary[id_in],
                                      self.max_text_len[id_in], self.text_offset[id_in],
                                      fill=self.fill_text[id_in])
                elif(type_in == 'image-features'):
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization)
                elif(type_in == 'video-features'):
                    x = self.loadFeatures(x, self.features_lengths[id_in], normalization_type, normalization)
            X.append(x)
            
        # Recover output samples
        Y = []
        for id_out, type_out in zip(self.ids_outputs, self.types_outputs):
            if(surpassed):
                y = eval('self.Y_'+set_name+'[id_out][last:]') + eval('self.Y_'+set_name+'[id_out][0:new_last]')
            else:
                y = eval('self.Y_'+set_name+'[id_out][last:new_last]')
            
            #if(set_name=='val'):
            #    logging.info(y)
            
            # Pre-process outputs
            if(not debug):
                if(type_out == 'categorical'):
                    nClasses = len(self.classes[id_out])
                    y = np_utils.to_categorical(y, nClasses).astype(np.uint8)
                elif(type_out == 'binary'):
                    y = np.array(y).astype(np.uint8)
                elif(type_out == 'text'):
                    y = self.loadText(y, self.vocabulary[id_out], 
                                      self.max_text_len[id_out], self.text_offset[id_out], 
                                      fill=self.fill_text[id_out])
                    y_aux = np.zeros(list(y.shape)+[self.n_classes_text[id_out]]).astype(np.uint8)
                    if self.max_text_len[id_out] == 0:
                        y_aux = np_utils.to_categorical(y, self.n_classes_text[id_out]).astype(np.uint8)
                    else:
                        for idx in range(y.shape[0]):
                            y_aux[idx] = np_utils.to_categorical(y[idx], self.n_classes_text[id_out]).astype(np.uint8)
                    y = y_aux
            Y.append(y)
        
        if debug:
            return [X, Y, [new_last, last, surpassed]]
 
        return [X,Y]
        
    
    # ------------------------------------------------------- #
    #       AUXILIARY FUNCTIONS
    #           
    # ------------------------------------------------------- #
        
    def __isLoaded(self, set_name, pos):
        if(eval('not self.loaded_'+set_name+'[pos]')):
            if(pos==0):
                raise Exception('Set '+set_name+' samples are not loaded yet.')
            elif(pos==1):
                raise Exception('Set '+set_name+' labels are not loaded yet.')
        return 
    
    
    def __checkSetName(self, set_name):
        if(set_name != 'train' and set_name != 'val' and set_name != 'test'):
            raise Exception('Incorrect set_name specified "'+set_name+ '"\nOnly "train", "val" or "test" are valid set names.')
        return 
        
    
    def __checkLengthSet(self, set_name):
        if(eval('self.loaded_'+set_name+'[0] and self.loaded_'+set_name+'[1]')):
            lengths = []
            for id_in in self.ids_inputs:
                if id_in not in self.optional_inputs:
                    exec('lengths.append(len(self.X_'+ set_name +'[id_in]))')
            for id_out in self.ids_outputs:
                exec('lengths.append(len(self.Y_'+ set_name +'[id_out]))')
            if(lengths[1:] != lengths[:-1]):
                raise Exception('Inputs and outputs size ('+str(lengths)+') for "' +set_name+ '" set do not match.')
            
                
    def __getNextSamples(self, k, set_name):
        """
            Gets the indices to the next K samples we are going to read.
        """
        self.__lock_read.acquire() # LOCK (for avoiding reading the same samples by different threads)
        
        new_last = eval('self.last_'+set_name+'+k')
        last = eval('self.last_'+set_name)
        length = eval('self.len_'+set_name)
        if(new_last > length):
            new_last = new_last - length
            surpassed = True
        else:
            surpassed = False
        exec('self.last_'+set_name+ '= new_last')
        
        self.__lock_read.release() # UNLOCK
        
        return [new_last, last, surpassed]
            
                    
    def __getstate__(self):
        """
            Behavour applied when pickling a Dataset instance.
        """ 
        obj_dict = self.__dict__.copy()
        del obj_dict['_Dataset__lock_read']
        return obj_dict
        
    
    def __setstate__(self, dict):
        """
            Behavour applied when unpickling a Dataset instance.
        """
        dict['_Dataset__lock_read'] = threading.Lock()
        self.__dict__ = dict

                
                
