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
                 mean_substraction=True):
        
        self.set_split = set_split
        self.dataset = dataset
        self.net = net
        # Several parameters
        self.params = {'batch_size': batch_size, 
                       'data_augmentation': data_augmentation,
                       'mean_substraction': mean_substraction,
                       'normalize_images': normalize_images,
                       'num_iterations': num_iterations}
    
    def generator(self):
            
        if(self.set_split == 'train'):
            data_augmentation = self.params['data_augmentation']
        else:
            data_augmentation = False

        it = 0
        while 1:

            if(self.set_split == 'train' and it%self.params['num_iterations']==0):
                self.dataset.shuffleTraining()
            elif(it%self.params['num_iterations']==0):
                self.dataset.resetCounters(set_name=self.set_split)
            it += 1
            
            # Recovers a batch of data
            X_batch, Y_batch = self.dataset.getXY(self.set_split, self.params['batch_size'], 
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
    
    def __init__(self, name, path, silence=False, size=[256, 256, 3], size_crop=[227, 227, 3]):
        
        # Dataset name
        self.name = name
        # Path to the folder where the images are stored
        self.path = path
        
        # If silence = False, some informative sentences will be printed while using the "Dataset" object instance
        self.silence = silence
        
        
        
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
        # List of identifiers for the inputs and outputs and their respective types (which will define the preprocessing applied)
        self.ids_inputs = []
        # TODO: include 'video' type
        self.types_inputs = [] # see accepted types in self.__accepted_types_inputs
        self.ids_outputs = []
        # TODO: include 'text' type
        self.types_outputs = [] # see accepted types in self.__accepted_types_outputs
        
        # List of implemented input and output data types
        self.__accepted_types_inputs = ['image', 'features', 'text']
        self.__accepted_types_outputs = ['categorical', 'binary', 'text']
        #################################################
        
        
        ############################ Parameters used for inputs of type 'text'
        self.vocabulary = dict()
        #################################################
        
        
        ############################ Parameters used for inputs of type 'image'
        # Image resize dimensions used for all the returned images
        self.img_size = size
        # Image crop dimensions for the returned images
        self.img_size_crop = size_crop
        # Set training mean to NaN (not calculated yet)
        self.train_mean = []
        # Tries to load a train_mean file from the dataset folder if exists
        mean_file_path = self.path+'/train_mean'
        for s in range(len(self.img_size)):
            mean_file_path += '_'+str(self.img_size[s])
        mean_file_path += '.jpg'
        if(os.path.isfile(mean_file_path)):
            self.setTrainMean(mean_file_path)
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
        
        
    def setClasses(self, path_classes):
        """
            Loads the list of classes of the dataset.
            Each line must contain a unique identifier of the class.
        """
        classes = []
        with open(path_classes, 'r') as list_:
            for line in list_:
                classes.append(line.rstrip('\n'))
        self.classes = classes
        
        self.dic_classes = dict()
        for c in range(len(self.classes)):
            self.dic_classes[self.classes[c]] = c
        
        if(not self.silence):
            logging.info('Loaded classes list with ' + str(len(self.classes)) + " different labels.")
        
        
        
    def setListGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True, type='image', id='image'):
        """
            Deprecated
        """
        logging.info("WARNING: The method setListGeneral() is deprecated, consider using setInputGeneral() instead.")
        self.setInputGeneral(path_list, split, shuffle, type, id)
    
    def setInputGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True, type='image', id='image'):
        """ 
            Loads a single list of samples from which train/val/test divisions will be applied. 
            
            :param path_list: path to the .txt file with the list of images.
            :param split: percentage of images used for [training, validation, test].
            :param shuffle: whether we are randomly shuffling the input samples or not.
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
        """
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
                raise NotImplementedException('The input type '+type+' is not implemented. The list of valid types are the following: '+str(self.__accepted_types_inputs))
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
            Deprecated
        """
        logging.info("WARNING: The method setList() is deprecated, consider using setInput() instead.")
        self.setInput(path_list, set_name, type, id)
    
    
    def setInput(self, path_list, set_name, type='image', id='image', tokenization='tokenize_basic', build_vocabulary=False):
        """
            Loads a list of samples which can contain all samples from the 'train', 'val', or
            'test' sets (specified by set_name).
            
            :param path_list: can either be a path to a .txt file containing the paths to the images or a python list of paths
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_inputs)
            :param id: identifier of the input data loaded
            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text').
        """
        self.__checkSetName(set_name)
        
        # Insert type and id of input data
        keys_X_set = eval('self.X_'+set_name+'.keys()')
        if(id not in self.ids_inputs):
            self.ids_inputs.append(id)
            if(type not in self.__accepted_types_inputs):
                raise NotImplementedException('The input type "'+type+'" is not implemented. The list of valid types are the following: '+str(self.__accepted_types_inputs))
            self.types_inputs.append(type)
        elif id in keys_X_set:
            raise Exception('An input with id "'+id+'" is already loaded into the Database.')
        
        # Proprocess the input data depending on its type
        if(type == 'image'):
            data = self.preprocessImages(path_list)
        elif(type == 'text'):
            data = self.preprocessText(path_list, id, tokenization, build_vocabulary)
        elif(type == 'features'):
            data = self.preprocessFeatures(path_list)
        
        self.__setInput(data, set_name, type, id)
        
    
    def __setInput(self, set, set_name, type, id):
        exec('self.X_'+set_name+'[id] = set')
        exec('self.loaded_'+set_name+'[0] = True')
        exec('self.len_'+set_name+' = len(set)')
        
        self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set inputs of type "'+type+'" with id "'+id+'" and length '+ str(eval('self.len_'+set_name)) + '.')
        
    
    
    def setLabels(self, labels_list, set_name, type='categorical', id='label'):
        """
            Deprecated
        """
        logging.info("WARNING: The method setLabels() is deprecated, consider using setOutput() instead.")
        self.setOutput(self, labels_list, set_name, type, id)
    
    def setOutput(self, path_list, set_name, type='categorical', id='label', tokenization='tokenize_basic', build_vocabulary=False):
        """
            Loads a set of output data, usually referencing values in self.classes (starting from 0)
            
            :param path_list: can either be a path to a .txt file containing the labels or a python list of labels
            :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
            :param type: identifier of the type of input we are loading (accepted types can be seen in self.__accepted_types_outputs)
            :param id: identifier of the input data loaded
            :param tokenization: type of tokenization applied (must be declared as a method of this class) (only applicable when type=='text').
            :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not (only applicable when type=='text').
        """
        self.__checkSetName(set_name)
        
        # Insert type and id of output data
        keys_Y_set = eval('self.Y_'+set_name+'.keys()')
        if(id not in self.ids_outputs):
            self.ids_outputs.append(id)
            if(type not in self.__accepted_types_outputs):
                raise NotImplementedException('The output type "'+type+'" is not implemented. The list of valid types are the following: '+str(self.__accepted_types_outputs))
            self.types_outputs.append(type)
        elif id in keys_Y_set:
            raise Exception('An input with id "'+id+'" is already loaded into the Database.')

        # Proprocess the output data depending on its type
        if(type == 'categorical'):
            data = self.preprocessCategorical(path_list)
        elif(type == 'text'):
            data = self.preprocessText(path_list, id, tokenization, build_vocabulary)
        elif(type == 'binary'):
            data = self.preprocessBinary(path_list)
            
        self.__setOutput(data, set_name, type, id)
    
    
    def __setOutput(self, labels, set_name, type, id):
        exec('self.Y_'+set_name+'[id] = labels')
        exec('self.loaded_'+set_name+'[1] = True')
    
        self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set outputs of type "'+type+'" with id "'+id+'" and length '+ str(eval('self.len_'+set_name)) + '.')
           
        
    # ------------------------------------------------------- #
    #       TYPE 'categorical' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessCategorical(self, labels_list):
        
        if(isinstance(labels_list, str) and os.path.isfile(labels_list)):
            labels = []
            with open(labels_list, 'r') as list_:
                for line in list_:
                    labels.append(int(line.rstrip('\n')))
        elif(isinstance(labels_list, list)):
            labels = labels_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a .txt with the labels or an instance of the class list.')
        
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
    
    def preprocessFeatures(self, path_list):
        
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
            raise Exception('Wrong type for "path_list". It must be a path to a .txt file. Each line must contain a path to a .npy file storing a feature vector. Alternatively "path_list" can be an instance of the class list.')
        
        return data
    
    
    def loadFeatures(self, X, loaded=False, external=False):
        
        features = []
        
        for feat in X:
            if(not external):
                feat = self.path +'/'+ feat

            # Check if the filename includes the extension
            feat = np.load(feat)
            features.append(feat)
            
        return np.array(features)
    
    # ------------------------------------------------------- #
    #       TYPE 'text' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessText(self, annotations_list, id, tokenization, build_vocabulary):
        
        labels = []
        if(isinstance(annotations_list, str) and os.path.isfile(annotations_list)):
            with open(annotations_list, 'r') as list_:
                for line in list_:
                    labels.append(line.rstrip('\n'))
        else:
            raise Exception('Wrong type for "annotations_list". It must be a path to a .txt with the sentences or a list of sentences.')
        
        if(build_vocabulary):
            self.build_vocabulary(labels, id)
        
        if(not id in self.vocabulary):
            raise Exception('The dataset must include a vocabulary with id "'+id+'" in order to process the type "text" data. Set "build_vocabulary" to True if you want to use the current data for building the vocabulary.')
            
        # Tokenize sentences
        if(hasattr(self, tokenization)):
            tokfun = eval('self.'+type)
            eval('labels = self.'+type+'(labels)')
        else:
            raise Exception('Tokenization procedure "'+ type +'" is not implemented.')
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                labels[i][j] = tokfun(labels[i][j])
    
        return labels
    
    
    def build_vocabulary(captions, id, n_words=0):
        """
            Vocabulary builder for data of type 'text'
        """
        logging.info("Creating vocabulary for data with id '"+id+"'.")
        
        counters = []
        sentence_counts = []
        counter = Counter()
        sentence_count = 0
        for line in captions:
            tokenized = tokenize(line)
            words = tokenized.strip().split(' ')
            words_low = map(lambda x: x.lower(), words)
            counter.update(words_low)
            sentence_count += 1

        counters.append(counter)
        sentence_counts.append(sentence_count)
        logging.info("\t %d unique words in %d sentences with a total of %d words." %
              (len(counter), sentence_count, sum(counter.values())))

        combined_counter = reduce(add, counters)
        logging.info("\t Total: %d unique words in %d sentences with a total of %d words." %
              (len(combined_counter), sum(sentence_counts),sum(combined_counter.values())))


        if n_words > 0:
            vocab_count = combined_counter.most_common(n_words - 2)
            logging.info("Creating dictionary of %s most common words, covering "
                        "%2.1f%% of the text."
                        % (n_words,
                           100.0 * sum([count for word, count in vocab_count]) /
                           sum(combined_counter.values())))
        else:
            logging.info("Creating dictionary of all words")
            vocab_count = counter.most_common()

        dictionary = {}
        for i, (word, count) in enumerate(vocab_count):
                dictionary[word] = i + 2
        dictionary['<eos>'] = 0
        dictionary['<unk>'] = 1
        
        self.vocabulary[id] = dict()
        self.vocabulary[id]['words2idx'] = dictionary
        inv_dictionary = {v: k for k, v in dictionary.items()}
        self.vocabulary[id]['idx2words'] = dictionary


    def loadText(self, X, vocabularies):
        """
            Text encoder. Transforms samples from a text representation into a numerical one.
        """
        vocab = vocabularies['words2idx']
        for i in range(len(X)):
            X[i] = [vocab[w] if w in vocab else vocab['<unk>'] for w in X[i]]
        
        return X
            
    # ------------------------------------------------------- #
    #       Tokenization functions
    # ------------------------------------------------------- #
            
    def tokenize_basic(caption):
        """
            Basic tokenizer for the input/output data of type 'text'
        """

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
        resAns = processPunctuation(resAns.encode("utf-8"))
        resAns = processDigitArticle(resAns)

        return resAns
    
    
    
    # ------------------------------------------------------- #
    #       TYPE 'image' SPECIFIC FUNCTIONS
    # ------------------------------------------------------- #
    
    def preprocessImages(self, path_list):
        
        if(isinstance(path_list, str) and os.path.isfile(path_list)): # path to list of images' paths
            data = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    data.append(line.rstrip('\n'))
        elif(isinstance(path_list, list)):
            data = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a .txt with an image path in each line or an instance of the class list with an image path in each position.')
        return data
    
        
    def setImageSize(self, size):
        """
            Changes the default image return size.
        """
        self.img_size = size
        
    
    def setImageSizeCrop(self, size_crop):
        """
            Changes the default image return size.
        """
        self.img_size_crop = size_crop
       
    
    def setTrainMean(self, mean_image, normalization=False):
        """
            Loads a pre-calculated training mean image, 'mean_image' can either be:
            
            - numpy.array (complete image)
            - list with a value per channel
            - string with the path to the stored image.
        """
        if(isinstance(mean_image, str)):
            if(not self.silence):
                logging.info("Loading train mean image from file.")
            mean_image = misc.imread(mean_image)
        elif(isinstance(mean_image, list)):
            mean_image = np.array(mean_image)
        self.train_mean = mean_image.astype(np.float32)
        
        if(normalization):
            self.train_mean = self.train_mean/255.0
            
        if(self.train_mean.shape != tuple(self.img_size)):
            if(len(self.train_mean.shape) == 1 and self.train_mean.shape[0] == self.img_size[2]):
                if(not self.silence):
                    logging.info("Converting input train mean pixels into mean image.")
                mean_image = np.zeros(tuple(self.img_size))
                for c in range(self.img_size[2]):
                    mean_image[:, :, c] = self.train_mean[c]
                self.train_mean = mean_image
            else:
                logging.warning("The loaded training mean size does not match the desired images size.\nChange the images size with setImageSize(size) or recalculate the training mean with calculateTrainMean().")
    

    
    # ------------------------------------------------------- #
    #       GETTERS
    #           [X,Y] pairs, X only, image mean, etc.
    # ------------------------------------------------------- #
        
    def getX(self, set_name, init, final, normalization=False, meanSubstraction=True, dataAugmentation=True):
        """
            Gets all the data samples stored between the positions init to final
            
            :param set_name: 'train', 'val' or 'test' set
            :param init: initial position in the corresponding set split. Must be bigger or equal than 0 and bigger than final.
            :param final: final position in the corresponding set split.
            :param normalization: indicates if we want to apply a 0-1 normalization on the images. If normalization then the images are stored in float32, else they are stored in int32
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
        for id,type in zip(self.ids_inputs, self.types_inputs):
            x = eval('self.X_'+set_name+'[id][init:final]')
            if(type == 'image'):
                x = self.loadImages(x, normalization, meanSubstraction, dataAugmentation)
            X.append(x)
        
        #X = eval('self.X_'+set_name+'[init:final]')
        #X = self.loadImages(X, normalization, meanSubstraction, dataAugmentation)
        return X
        
        
    def getXY(self, set_name, k, normalization=False, meanSubstraction=True, dataAugmentation=True):
        """
            Gets the [X,Y] pairs for the next 'k' samples in the desired set.
            
            :param set_name: 'train', 'val' or 'test' set
            :param k: number of consecutive samples retrieved from the corresponding set.
            :param normalization: indicates if we want to apply a 0-1 normalization on the images. If 'normalization' == True then the images are stored in float32, else they are stored in int32
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
            if(type_in == 'image'):
                x = self.loadImages(x, normalization, meanSubstraction, dataAugmentation)
            elif(type_in == 'text'):
                x = self.loadText(x, self.vocabularies[id_in])
            elif(type_in == 'features'):
                x = self.loadFeatures(x)
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
            if(type_out == 'categorical'):
                nClasses = len(self.classes)                
                y = np_utils.to_categorical(y, nClasses).astype(np.uint8)
            elif(type_out == 'binary'):
                y = np.array(y).astype(np.uint8)
            elif(type_out == 'features'):
                y = self.loadFeatures(y)
            Y.append(y)
        
        return [X,Y]
    
        
    def calculateTrainMean(self):
        """
            Calculates the mean of the data belonging to the training set split in each channel.
        """
        calculate = False
        if(not isinstance(self.train_mean, np.ndarray)):
            calculate = True
        elif(self.train_mean.shape != tuple(self.img_size)):
            calculate = True
            if(not self.silence):
                logging.warning("The loaded training mean size does not match the desired images size. Recalculating mean...")
            
        if(calculate):
            if(not self.silence):
                logging.info("Start training set mean calculation...")
            
            I_sum = np.zeros(self.img_size, dtype=np.longlong)
            
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
            self.train_mean = I_sum/self.len_train
            
            # Store the calculated mean
            mean_name = self.path+'/train_mean'
            for s in range(len(self.img_size)):
                mean_name += '_'+str(self.img_size[s])
            mean_name += '.jpg'
            store_path = self.path+'/'+mean_name
            misc.imsave(store_path, self.train_mean)
            
            self.train_mean = self.train_mean.astype(np.float32)/255.0
            
            if(not self.silence):
                logging.info("Image mean stored in "+ store_path)
            
        # Return the mean
        return self.train_mean
    
        
    def loadImages(self, images, normalization=False, meanSubstraction=True, dataAugmentation=True, external=False, loaded=False):
        """
            Loads a set of images from disk.
            
            :param images : list of image string names or list of matrices representing images
            :param normalization : whether we applying a 0-1 normalization to the images
            :param meanSubstraction : whether we are removing the training mean
            :param dataAugmentation : whether we are applying dataAugmentatino (random cropping and horizontal flip)
            :param external : if True the images will be loaded from an external database, in this case the list of images must be absolute paths
            :param loaded : set this option to True if images is a list of matricies instead of a list of strings
        """
        # Prepare the training mean image
        if(meanSubstraction): # remove mean
            if(not isinstance(self.train_mean, np.ndarray)):
                raise Exception('Training mean is not loaded or calculated yet.')
            train_mean = copy.copy(self.train_mean)
            # Take central part
            left = np.round(np.divide([self.img_size[0]-self.img_size_crop[0], self.img_size[1]-self.img_size_crop[1]], 2.0))
            right = left + self.img_size_crop[0:2]
            train_mean = train_mean[left[0]:right[0], left[1]:right[1], :]
            # Transpose dimensions
            if(len(self.img_size) == 3): # if it is a 3D image
                # Convert RGB to BGR
                if(self.img_size[2] == 3): # if has 3 channels
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
        if(len(self.img_size) == 3):
            I = np.zeros([nImages]+[self.img_size_crop[2]]+self.img_size_crop[0:2], dtype=type_imgs)
        else:
            I = np.zeros([nImages]+self.img_size_crop, dtype=type_imgs)
            
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
            im = misc.imresize(im, tuple(self.img_size)).astype(type_imgs)
            if(len(self.img_size) == 3 and len(im.shape) < 3): # convert grayscale into RGB (or any other channel#)
                nCh = self.img_size[2]
                rgb_im = np.empty((im.shape[0], im.shape[1], nCh), dtype=np.float32)
                for c in range(nCh):
                    rgb_im[:,:,c] = im
                im = rgb_im
                
            # Normalize
            if(normalization):
                im = im/255.0
                
            # Data augmentation
            if(not dataAugmentation):
                # Take central image
                left = np.round(np.divide([self.img_size[0]-self.img_size_crop[0], self.img_size[1]-self.img_size_crop[1]], 2.0))
                right = left + self.img_size_crop[0:2]
                im = im[left[0]:right[0], left[1]:right[1], :]
            else:
                # Take random crop
                margin = [self.img_size[0]-self.img_size_crop[0], self.img_size[1]-self.img_size_crop[1]]
                left = random.sample([k_ for k_ in range(margin[0])], 1) + random.sample([k for k in range(margin[1])], 1)
                right = np.add(left, self.img_size_crop[0:2])
                im = im[left[0]:right[0], left[1]:right[1], :]
                
                # Randomly flip (with a certain probability)
                flip = random.random()
                if(flip < prob_flip_horizontal): # horizontal flip
                    im = np.fliplr(im)
                flip = random.random()
                if(flip < prob_flip_vertical): # vertical flip
                    im = np.flipud(im)
            
            # Permute dimensions
            if(len(self.img_size) == 3):
                # Convert RGB to BGR
                if(self.img_size[2] == 3): # if has 3 channels
                    aux = copy.copy(im)
                    im[:,:,0] = aux[:,:,2]
                    im[:,:,2] = aux[:,:,0]
                im = np.transpose(im, (2, 0, 1))
            else:
                pass
            
            # Substract training images mean
            if(meanSubstraction): # remove mean
                im = im - train_mean
            
            if(len(I.shape) == 4):
                I[i,:,:,:] = im
            else:
                I[i,:,:] = im

        return I
    
    
    def getClassID(self, class_name):
        """
            Returns the class id (int) for a given class string.
        """
        return self.dic_classes[class_name]
        
    
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
                exec('lengths.append(len(self.X_'+ set_name +'[id_in]))')
            for id_out in self.ids_outputs:
                exec('lengths.append(len(self.Y_'+ set_name +'[id_out]))')
            if(lengths[1:] != lengths[:-1]):
                raise Exception('Inputs and outputs size for "' +set_name+ '" set do not match.')
            
                
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

                
                
