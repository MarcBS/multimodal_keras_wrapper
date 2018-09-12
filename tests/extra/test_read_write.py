# -*- coding: utf-8 -*-
import pytest
import sys
import os
import numpy
from six import iteritems
from keras_wrapper.extra.read_write import *
from keras_wrapper.utils import flatten_list_of_lists


def test_dirac():
    assert dirac(1, 1) == 1
    assert dirac(2, 1) == 0


def test_create_dir_if_not_exists():
    create_dir_if_not_exists('test_directory')
    assert os.path.isdir('test_directory')


def test_clean_dir():
    clean_dir('test_directory')
    assert os.path.isdir('test_directory')


def test_file2list():
    reference_text = 'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^'.decode('utf-8') if sys.version_info.major == 2 else 'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^'
    stripped_list = file2list('tests/data/test_data.txt', stripfile=True)
    assert len(stripped_list) == 3
    assert stripped_list[1] == reference_text


def test_numpy2hdf5():
    filepath = 'test_file'
    data_name = 'test_data'
    my_np = np.random.rand(10, 10).astype('float32')
    numpy2hdf5(filepath, my_np, data_name=data_name)
    assert os.path.isfile(filepath)
    my_np_loaded = np.asarray(load_hdf5_simple(filepath, dataset_name=data_name)).astype('float32')
    assert np.all(my_np == my_np_loaded)


def test_numpy2file():
    filepath = 'test_file'
    my_np = np.random.rand(10, 10).astype('float32')
    numpy2file(filepath, my_np)
    assert os.path.isfile(filepath)
    my_np_loaded = np.asarray(np.load(filepath)).astype('float32')
    assert np.all(my_np == my_np_loaded)


def test_listoflists2file():
    mylist = [['This is a text file. Containing characters of different encodings.'],
              ['ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^'],
              ['首先 ，']
              ]
    filepath = 'saved_list'
    listoflists2file(filepath, mylist)
    loaded_list = file2list('saved_list')
    flatten_list = [encode_list(sublist) for sublist in mylist]
    flatten_list = flatten_list_of_lists(flatten_list)
    assert loaded_list == flatten_list


def test_list2file():
    mylist = ['This is a text file. Containing characters of different encodings.',
              'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^',
              '首先 ，'
              ]
    filepath = 'saved_list'
    list2file(filepath, mylist)
    loaded_list = file2list('saved_list')
    my_encoded_list = encode_list(mylist)
    assert loaded_list == my_encoded_list


def test_list2stdout():
    mylist = ['This is a text file. Containing characters of different encodings.',
              'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^',
              '首先 ，'
              ]
    list2stdout(mylist)


def test_nbest2file():
    my_nbest_list = [
        [[1, 'This is a text file. Containing characters of different encodings.', 0.1],
         [1, 'Other hypothesis. Containing characters of different encodings.', 0.2]
         ],
        [[2, 'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^', 0.3]],
        [[3, '首先 ，', 90.3]]
    ]
    filepath = 'saved_nbest'
    nbest2file(filepath, my_nbest_list)
    nbest = file2list(filepath)
    assert nbest == encode_list(['1 ||| This is a text file. Containing characters of different encodings. ||| 0.1',
                                 '1 ||| Other hypothesis. Containing characters of different encodings. ||| 0.2',
                                 '2 ||| ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^ ||| 0.3',
                                 '3 ||| 首先 ， ||| 90.3'])


def test_dump_load_hdf5_simple():
    filepath = 'test_file'
    data_name = 'test_data'
    data = np.random.rand(10, 10).astype('float32')
    dump_hdf5_simple(filepath, data_name, data)
    loaded_data = load_hdf5_simple(filepath, dataset_name=data_name)
    assert np.all(loaded_data == data)


def test_dict2file():
    filepath = 'saved_dict'
    mydict = {1: 'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^', '首先': 9}
    title = None
    dict2file(mydict, filepath, title, permission='w')
    loaded_dict = file2list(filepath)
    assert loaded_dict == encode_list(['1:ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^', '首先:9'])
    title = 'Test dict'
    dict2file(mydict, filepath, title, permission='w')
    loaded_dict = file2list(filepath)
    assert loaded_dict == encode_list(['Test dict', '1:ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^', '首先:9'])


def test_dict2pkl_pkl2dict():
    filepath = 'saved_dict'
    mydict = {1: 'ẁñ á é í ó ú à è ì ò ù ä ë ï ö ü ^', '首先': 9}
    dict2pkl(mydict, filepath)
    loaded_dict = pkl2dict(filepath + '.pkl')
    assert loaded_dict == mydict


if __name__ == '__main__':
    pytest.main([__file__])
