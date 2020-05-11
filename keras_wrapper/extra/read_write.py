# -*- coding: utf-8 -*-
"""
Reads from input file or writes to the output file.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de

Modified by: Marc Bola\~nos
             \'Alvaro Peris
"""
from __future__ import print_function
from six import iteritems
import json
import os
import codecs
import numpy as np
import tables
import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

if sys.version_info.major == 3:
    import _pickle as pk

    unicode_fn = str
else:
    import cPickle as pk

    unicode_fn = unicode


# Helpers


def encode_list(mylist):
    """
    Encode list as utf-8 if we are working with Python 2.x or as str if we are working with Python 3.x.
    :param mylist:
    :return:
    """
    return [l.decode('utf-8') if isinstance(l, str) else unicode(l) for l in
            mylist] if sys.version_info.major == 2 else [str(l) for l in mylist]


def dirac(pred,
          gt):
    """
    Chechks whether pred == gt.
    :param pred: Prediction
    :param gt: Ground-truth.
    :return:
    """
    return int(pred == gt)


def create_dir_if_not_exists(directory):
    """
    Creates a directory if it doen't exist

    :param directory: Directory to create
    :return: None
    """
    if not os.path.exists(directory):
        logger.info("<<< creating directory " + directory + " ... >>>")
        os.makedirs(directory)


def clean_dir(directory):
    """
    Creates (or empties) a directory
    :param directory: Directory to create
    :return: None
    """

    if os.path.exists(directory):
        import shutil
        logger.warning('<<< Deleting directory: %s >>>' % directory)
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        os.makedirs(directory)


# Main functions
def file2list(filepath,
              stripfile=True):
    """
    Loads a file into a list. One line per element.
    :param filepath: Path to the file to load.
    :param stripfile: Whether we should strip the lines of the file or not.
    :return: List containing the lines read.
    """
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        lines = [k for k in [k.strip() for k in f.readlines()] if len(k) > 0] if stripfile else [k for k in
                                                                                                 f.readlines()]
        return lines


def numpy2hdf5(filepath,
               mylist,
               data_name='data',
               permission='w'):
    """
    Saves a numpy array as HDF5.
    """
    if 'w' in permission:
        f = tables.open_file(filepath,
                             mode=permission)
        atom = tables.Float32Atom()
        array_c = f.create_earray(f.root, data_name, atom,
                                  tuple([0] + [mylist.shape[i] for i in range(1, len(mylist.shape))]))
        array_c.append(mylist)
        f.close()
    elif permission == 'a':
        f = tables.open_file(filepath, mode='a')
        f.root.data.append(mylist)
        f.close()


def numpy2file(filepath,
               mylist,
               permission='wb',
               split=False):
    """
    Saves a numpy array as a file.
    :param filepath: Destination path.
    :param mylist: Numpy array to save.
    :param permission: Write permission.
    :param split: Whether we save each element from mylist in a separate file or not.
    :return:
    """
    mylist = np.asarray(mylist)
    if split:
        for i, filepath_ in list(enumerate(filepath)):
            with open(filepath_, permission) as f:
                np.save(f, mylist[i])
    else:
        with open(filepath, permission) as f:
            np.save(f, mylist)


def numpy2imgs(folder_path,
               mylist,
               imgs_names,
               dataset):
    """
    Save a numpy array as images.
    :param folder_path: Folder of the images to save.
    :param mylist: Numpy array containing the images.
    :param imgs_names: Names of the images to be saved.
    :param dataset:
    :return:
    """
    from PIL import Image as pilimage
    create_dir_if_not_exists(folder_path)
    n_classes = mylist.shape[-1]

    for img, name in zip(mylist, imgs_names):
        name = '_'.join(name.split('/'))
        file_path = folder_path + "/" + name  # image file

        out_img = dataset.getImageFromPrediction_3DSemanticLabel(img, n_classes)

        # save the segmented image
        out_img = pilimage.fromarray(np.uint8(out_img))
        out_img.save(file_path)


def listoflists2file(filepath,
                     mylist,
                     permission='w'):
    """
    Saves a list of lists into a file. Each element in a line.
    :param filepath: Destination file.
    :param mylist: List of lists to save.
    :param permission: Writing permission.
    :return:
    """
    mylist = [encode_list(sublist) for sublist in mylist]
    mylist = [item for sublist in mylist for item in sublist]
    mylist = u'\n'.join(mylist)
    with codecs.open(filepath, permission, encoding='utf-8') as f:
        f.write(mylist)
        f.write('\n')


def list2file(filepath,
              mylist,
              permission='w'):
    """
    Saves a list into a file. Each element in a line.
    :param filepath: Destination file.
    :param mylist: List to save.
    :param permission: Writing permission.
    :return:
    """
    mylist = encode_list(mylist)
    mylist = u'\n'.join(mylist)
    with codecs.open(filepath,
                     permission,
                     encoding='utf-8') as f:
        f.write(mylist)
        f.write('\n')


def list2stdout(mylist):
    """
    Prints a list in STDOUT
    :param mylist: List to print.
    """
    mylist = encode_list(mylist)
    mylist = '\n'.join(mylist)
    print(mylist)


def nbest2file(filepath,
               mylist,
               separator=u'|||',
               permission='w'):
    """
    Saves an N-best list into a file.
    :param filepath: Destination path.
    :param mylist: List to save.
    :param separator: Separator between N-best list components.
    :param permission: Writing permission.
    :return:
    """
    newlist = []
    for l in mylist:
        for l2 in l:
            a = []
            for l3 in l2:
                if isinstance(l3, list):
                    l3 = l3[0]
                if sys.version_info.major == 2:
                    if isinstance(l3, str):
                        a.append(l3.decode('utf-8') + u' ' + separator)
                    else:
                        a.append(unicode(l3) + u' ' + separator)
                else:
                    a.append(str(l3) + ' ' + separator)
            a = ' '.join(a + [' '])
            newlist.append(a.strip()[:-len(separator)].strip())
    mylist = '\n'.join(newlist)
    if isinstance(mylist[0], str) and sys.version_info.major == 2:
        mylist = mylist.encode('utf-8')
    with codecs.open(filepath, permission, encoding='utf-8') as f:
        f.write(mylist)


def list2vqa(filepath,
             mylist,
             qids,
             permission='w',
             extra=None):
    """
    Saves a list with the VQA format.
    """
    res = []
    for i, (ans, qst) in list(enumerate(zip(mylist, qids))):
        line = {'answer': ans, 'question_id': int(qst)}
        if extra is not None:
            line['reference'] = extra['reference'][i]
            line['top5'] = str(
                [[extra['vocab'][p], extra['probs'][i][p]] for p in np.argsort(extra['probs'][i])[::-1][:5]])
            line['max_prob'] = str(max(extra['probs'][i]))
        res.append(line)
    with codecs.open(filepath, permission, encoding='utf-8') as f:
        json.dump(res, f)


def dump_hdf5_simple(filepath,
                     dataset_name,
                     data):
    """
    Saves a HDF5 file.
    """
    import h5py
    h5f = h5py.File(filepath,
                    'w')
    h5f.create_dataset(dataset_name,
                       data=data)
    h5f.close()


def load_hdf5_simple(filepath,
                     dataset_name='data'):
    """
    Loads a HDF5 file.
    """
    import h5py
    h5f = h5py.File(filepath, 'r')
    tmp = h5f[dataset_name][:]
    h5f.close()
    return tmp


def model_to_json(path,
                  model):
    """
    Saves model as a json file under the path.
    """
    json_model = model.to_json()
    with open(path, 'w') as f:
        json.dump(json_model, f)


def json_to_model(path):
    """
    Loads a model from the json file.
    """
    from keras.models import model_from_json
    with open(path, 'r') as f:
        json_model = json.load(f)
    model = model_from_json(json_model)
    return model


def model_to_text(filepath, model_added):
    """
    Save the model to text file.
    """
    pass


def text_to_model(filepath):
    """
    Loads the model from the text file.
    """
    pass


def print_qa(questions,
             answers_gt,
             answers_gt_original,
             answers_pred,
             era,
             similarity=dirac,
             path=''):
    """
    In:
        questions - list of questions
        answers_gt - list of answers (after modifications like truncation)
        answers_gt_original - list of answers (before modifications)
        answers_pred - list of predicted answers
        era - current era
        similarity - measure that measures similarity between gt_original and prediction;
            by default dirac measure
        path - path for the output (if empty then stdout is used)
            by fedault an empty path
    Out:
        the similarity score
    """
    if len(questions) != len(answers_gt):
        raise AssertionError('Diferent questions and answers_gt lengths.')
    if len(questions) != len(answers_pred):
        raise AssertionError('Diferent questions and answers_pred lengths.')

    output = ['-' * 50, 'Era {0}'.format(era)]
    score = 0.0
    for k, q in list(enumerate(questions)):
        a_gt = answers_gt[k]
        a_gt_original = answers_gt_original[k]
        a_p = answers_pred[k]
        score += dirac(a_p, a_gt_original)
        if isinstance(q[0], unicode_fn):
            tmp = unicode_fn('question: {0}\nanswer: {1}\nanswer_original: {2}\nprediction: {3}\n')
        else:
            tmp = 'question: {0}\nanswer: {1}\nanswer_original: {2}\nprediction: {3}\n'
        output.append(tmp.format(q, a_gt, a_gt_original, a_p))
    score = (score / len(questions)) * 100.0
    output.append('Score: {0}'.format(score))
    if path == '':
        print('%s' % '\n'.join(map(str, output)))
    else:
        list2file(path, output)
    return score


def dict2file(mydict,
              path,
              title=None,
              separator=':',
              permission='a'):
    """
    In:
        mydict - dictionary to save in a file
        path - path where mydict is stored
        title - the first sentence in the file;
            useful if we write many dictionaries
            into the same file
    """
    tmp = [encode_list([x[0]])[0] + separator + encode_list([x[1]])[0] for x in list(iteritems(mydict))]
    if title is not None:
        output_list = [title]
        output_list.extend(tmp)
    else:
        output_list = tmp
    list2file(path,
              output_list,
              permission=permission)


def dict2pkl(mydict,
             path):
    """
    Saves a dictionary object into a pkl file.
    :param mydict: dictionary to save in a file
    :param path: path where my_dict is stored
    :return:
    """
    if path[-4:] == '.pkl':
        extension = ''
    else:
        extension = '.pkl'
    with open(path + extension, 'wb') as f:
        pk.dump(mydict,
                f,
                protocol=-1)


def pkl2dict(path):
    """
    Loads a dictionary object from a pkl file.

    :param path: Path to the pkl file to load
    :return: Dict() containing the loaded pkl
    """
    with open(path, 'rb') as f:
        if sys.version_info.major == 2:
            return pk.load(f)
        else:
            return pk.load(f,
                           encoding='latin1')
