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
        
        # Indicators for knowing if the data [X, Y] has been loaded for each data split
        self.loaded_train = [False, False]
        self.loaded_val = [False, False]
        self.loaded_test = [False, False]
        self.len_train = 0
        self.len_val = 0
        self.len_test = 0
        
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
        
        # Lock for threads synchronization
        self.__lock_read = threading.Lock()
        
        
        self.resetCounters()
        
    
    def shuffleTraining(self):
        """
            Applies a random shuffling to the training samples.
        """
        if(not self.silence):
            logging.info("Shuffling training samples.")
        
        # Get current samples
        samples = self.X_train
        labels = self.Y_train
        num = self.len_train
        
        # Shuffle
        shuffled_order = random.sample([i for i in range(num)], num)
        samples = []
        labels = []
        for s in shuffled_order:
            samples.append(self.X_train[s])
            labels.append(self.Y_train[s])
        
        # Insert samples again
        silence = self.silence
        self.setSilence(True)
        self.__setList(samples, 'train')
        self.__setLabels(labels, 'train')
        self.setSilence(silence)
    
    # ------------------------------------------------------- #
    #       SETTERS
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
        
        
    def setListGeneral(self, path_list, split=[0.8, 0.1, 0.1], shuffle=True):
        """ 
            Loads a single list of images from which train/val/test divisions will be applied. 
            
            :param path_list: path to the .txt file with the list of images.
            :param split: percentage of images used for [training, validation, test].
            :param shuffle: wether we are randomly shuffling the input samples or not.
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
        
        offset = 0
        order = ['train', 'val', 'test']
        set_split = []
        for i in range(len(split)):
            last = int(math.ceil(nSamples*split[i]))
            set_split.append(set_num[offset:offset+last])
            offset += last
            
            # Insert into the corresponding list
            if(len(set_split[i]) > 0):
                self.__setList([set[elem] for elem in set_split[i]], order[i])
        
        
    def setList(self, path_list, set_name):
        """
            Loads a list of images which can contain all samples from the 'train', 'val', or
            'test' sets (specified by set_name).
            'path_list' can either be a path to a .txt file containing the paths to the images or a python list of paths
        """
        self.__checkSetName(set_name)
        
        if(isinstance(path_list, str) and os.path.isfile(path_list)):
            set = []
            with open(path_list, 'r') as list_:
                for line in list_:
                    set.append(line.rstrip('\n'))
        elif(isinstance(path_list, list)):
            set = path_list
        else:
            raise Exception('Wrong type for "path_list". It must be a path to a .txt with the labels or an instance of the class list.')
        self.__setList(set, set_name)
        
    
    def __setList(self, set, set_name):
        exec('self.X_'+set_name+' = set')
        exec('self.loaded_'+set_name+'[0] = True')
        exec('self.len_'+set_name+' = len(set)')
        
        self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set samples with '+ str(eval('self.len_'+set_name)) + ' samples.')
        
                
    def setLabels(self, labels_list, set_name):
        """
            Loads a set of int labels referencing values in self.classes (starting from 0)
            'labels_list' can either be a path to a .txt file containing the labels or a python list of labels
        """
        self.__checkSetName(set_name)
        
        if(isinstance(labels_list, str) and os.path.isfile(labels_list)):
            labels = []
            with open(labels_list, 'r') as list_:
                for line in list_:
                    labels.append(int(line.rstrip('\n')))
        elif(isinstance(labels_list, list)):
            labels = labels_list
        else:
            raise Exception('Wrong type for "labels_list". It must be a path to a .txt with the labels or an instance of the class list.')
        self.__setLabels(labels, set_name)
    
    
    def __setLabels(self, labels, set_name):
        exec('self.Y_'+set_name+' = labels')
        exec('self.loaded_'+set_name+'[1] = True')
    
        self.__checkLengthSet(set_name)
        
        if(not self.silence):
            logging.info('Loaded "' + set_name + '" set labels with '+ str(eval('self.len_'+set_name)) + ' samples.')
    
    
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
        
        X = eval('self.X_'+set_name+'[init:final]')
        X = self.loadImages(X, normalization, meanSubstraction, dataAugmentation)
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
        if(surpassed):
            Y = eval('self.Y_'+set_name+'[last:]') + eval('self.Y_'+set_name+'[0:new_last]')
            X = eval('self.X_'+set_name+'[last:]') + eval('self.X_'+set_name+'[0:new_last]')
        else:
            Y = eval('self.Y_'+set_name+'[last:new_last]')
            X = eval('self.X_'+set_name+'[last:new_last]')
        
        X = self.loadImages(X, normalization, meanSubstraction, dataAugmentation)
        
        nClasses = len(self.classes)
        Y = np_utils.to_categorical(Y, nClasses).astype(np.uint8)
        
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
            train_mean = train_mean[int(left[0]):int(right[0]), int(left[1]):int(right[1]), :]
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
            exec('lenX = len(self.X_'+ set_name +')')
            exec('lenY = len(self.Y_'+ set_name +')')
            if(lenX != lenY):
                raise Exception('Samples (' +str(lenX)+ ') and labels (' +str(lenY)+ ') size for "' +set_name+ '" set do not match.')
        return 
            
                
    def __getNextSamples(self, k, set_name):
        """
            Gets the indices to the next K samples we are going to read.
        """
        self.__lock_read.acquire() # LOCK
        
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

                
                
