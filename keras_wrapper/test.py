#from Dataset_ImageDescription import Dataset_ImageDescription as Dataset_ID
from keras_wrapper.dataset import Dataset

import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def main_test():
    """This is a test comment"""
    
    ds = Dataset('test_name', '/media/HDD_2TB/DATASETS/Food_101_Dataset/images')
    ds.setImageSizeCrop([227, 227, 3])
    ds.setClasses('/media/HDD_2TB/DATASETS/Food_101_Dataset/meta/classes.txt')
    #ds.setListGeneral('/media/HDD_2TB/DATASETS/Food_101_Dataset/meta/train.txt', split=[0.8, 0.2, 0.0], shuffle=True)
    ds.setList('/media/HDD_2TB/DATASETS/Food_101_Dataset/meta/test.txt', 'test')
    ds.setList('/media/HDD_2TB/DATASETS/Food_101_Dataset/meta/val_split.txt', 'val')
    ds.setList('/media/HDD_2TB/DATASETS/Food_101_Dataset/meta/train_split.txt', 'train')
    
    logging.info(ds.name +' loaded.')

    
    
    
main_test()