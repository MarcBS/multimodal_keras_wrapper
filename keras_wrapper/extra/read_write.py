"""
Reads from input file or writes to the output file.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de

Modified by: Marc Bola\~nos
             \'Alvaro Peris
"""

import json
import logging
import os
import codecs
import numpy as np
import tables


###
# Helpers
###
def _dirac(pred, gt):
    return int(pred == gt)


def create_dir_if_not_exists(directory):
    """
    Creates a directory if it doen't exist

    :param directory: Directory to create
    :return: None
    """
    if not os.path.exists(directory):
        logging.info("<<< creating directory " + directory + " ... >>>")
        os.makedirs(directory)


def clean_dir(directory):
    """
    Creates (or empties) a directory
    :param directory: Directory to create
    :return: None
    """

    if os.path.exists(directory):
        import shutil
        print '<<< Warning!: Deleting directory: %s >>>' % directory
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        os.makedirs(directory)


###
# Main functions
###
def file2list(filepath, stripfile=True):
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        lines = [k for k in [k.strip() for k in f.readlines()] if len(k) > 0] if stripfile else [k for k in
                                                                                                 f.readlines()]
        return lines


def numpy2hdf5(filepath, mylist, data_name='data', permission='w'):
    if permission == 'w':
        f = tables.open_file(filepath, mode=permission)
        atom = tables.Float32Atom()
        array_c = f.create_earray(f.root, data_name, atom,
                                  tuple([0] + [mylist.shape[i] for i in range(1, len(mylist.shape))]))
        array_c.append(mylist)
        f.close()
    elif permission == 'a':
        f = tables.open_file(filepath, mode='a')
        f.root.data.append(mylist)
        f.close()


def numpy2file(filepath, mylist, permission='w', split=False):
    mylist = np.asarray(mylist)
    if split:
        for i, filepath_ in enumerate(filepath):
            with open(filepath_, permission) as f:
                np.save(f, mylist[i])
    else:
        with open(filepath, permission) as f:
            np.save(f, mylist)


def numpy2imgs(folder_path, mylist, imgs_names, dataset):
    from PIL import Image as pilimage
    create_dir_if_not_exists(folder_path)
    n_samples, wh, n_classes = mylist.shape

    for img, name in zip(mylist, imgs_names):
        name = '_'.join(name.split('/'))
        file_path = folder_path + "/" + name  # image file

        out_img = dataset.getImageFromPrediction_3DSemanticLabel(img, n_classes)

        # save the segmented image
        out_img = pilimage.fromarray(np.uint8(out_img))
        out_img.save(file_path)


def listoflists2file(filepath, mylist, permission='w'):
    mylist = [str(sublist) for sublist in mylist]
    mylist = '\n'.join(mylist)
    if type(mylist[0]) is unicode:
        mylist = mylist.encode('utf-8')
    with open(filepath, permission) as f:
        f.writelines(mylist)


def list2file(filepath, mylist, permission='w'):
    mylist = [l for l in mylist]
    mylist = '\n'.join(mylist)
    if type(mylist[0]) is unicode:
        mylist = mylist.encode('utf-8')
    with open(filepath, permission) as f:
        f.writelines(mylist)


def list2stdout(mylist):
    mylist = [str(l) for l in mylist]
    mylist = '\n'.join(mylist)
    if type(mylist[0]) is unicode:
        mylist = mylist.encode('utf-8')
    print mylist


def nbest2file(filepath, mylist, separator='|||', permission='w'):
    newlist = []
    for l in mylist:
        for l2 in l:
            a = []
            for l3 in l2:
                if type(l3) is list:
                    l3 = l3[0]
                a.append(str(l3) + ' |||')
            a = ' '.join(a + [' '])
            newlist.append(a.strip()[:-len(separator)].strip())
    mylist = '\n'.join(newlist)
    if type(mylist[0]) is unicode:
        mylist = mylist.encode('utf-8')
    with open(filepath, permission) as f:
        f.writelines(mylist)


def list2vqa(filepath, mylist, qids, permission='w', extra=None):
    res = []
    for i, (ans, qst) in enumerate(zip(mylist, qids)):
        line = {'answer': ans, 'question_id': int(qst)}
        if extra is not None:
            line['reference'] = extra['reference'][i]
            # line['probs'] = str(extra['probs'][i]) # vector of probabilities for all outputs
            line['top5'] = str(
                [[extra['vocab'][p], extra['probs'][i][p]] for p in np.argsort(extra['probs'][i])[::-1][:5]])
            line['max_prob'] = str(max(extra['probs'][i]))
        res.append(line)
    with open(filepath, permission) as f:
        json.dump(res, f)


def dump_hdf5_simple(filepath, dataset_name, data):
    import h5py
    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset(dataset_name, data=data)
    h5f.close()


def load_hdf5_simple(filepath, dataset_name='data'):
    import h5py
    h5f = h5py.File(filepath, 'r')
    tmp = h5f[dataset_name][:]
    h5f.close()
    return tmp


def pickle_model(
        path,
        model,
        word2index_x,
        word2index_y,
        index2word_x,
        index2word_y):
    import sys
    import cPickle
    modifier = 10
    tmp = sys.getrecursionlimit()
    sys.setrecursionlimit(tmp * modifier)
    with open(path, 'wb') as f:
        p_dict = {'model': model,
                  'word2index_x': word2index_x,
                  'word2index_y': word2index_y,
                  'index2word_x': index2word_x,
                  'index2word_y': index2word_y}
        cPickle.dump(p_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    sys.setrecursionlimit(tmp)


def unpickle_model(path):
    import cPickle
    with open(path, 'rb') as f:
        model = cPickle.load(f)['model']
    return model


def unpickle_vocabulary(path):
    import cPickle
    p_dict = {}
    with open(path, 'rb') as f:
        pickle_load = cPickle.load(f)
        p_dict['word2index_x'] = pickle_load['word2index_x']
        p_dict['word2index_y'] = pickle_load['word2index_y']
        p_dict['index2word_x'] = pickle_load['index2word_x']
        p_dict['index2word_y'] = pickle_load['index2word_y']
    return p_dict


def unpickle_data_provider(path):
    import cPickle
    with open(path, 'rb') as f:
        dp = cPickle.load(f)['data_provider']
    return dp


def model_to_json(path, model):
    """
    Saves model as a json file under the path.
    """
    import json
    json_model = model.to_json()
    with open(path, 'w') as f:
        json.dump(json_model, f)


def json_to_model(path):
    """
    Loads a model from the json file.
    """
    import json
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


def print_qa(questions, answers_gt, answers_gt_original, answers_pred,
             era, similarity=_dirac, path=''):
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
    assert (len(questions) == len(answers_gt))
    assert (len(questions) == len(answers_pred))
    output = ['-' * 50, 'Era {0}'.format(era)]
    score = 0.0
    for k, q in enumerate(questions):
        a_gt = answers_gt[k]
        a_gt_original = answers_gt_original[k]
        a_p = answers_pred[k]
        score += _dirac(a_p, a_gt_original)
        if type(q[0]) is unicode:
            tmp = unicode(
                'question: {0}\nanswer: {1}\nanswer_original: {2}\nprediction: {3}\n')
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


def dict2file(mydict, path, title=None, separator=':'):
    """
    In:
        mydict - dictionary to save in a file
        path - path where acc_dict is stored
        title - the first sentence in the file;
            useful if we write many dictionaries
            into the same file
    """
    tmp = [str(x[0]) + separator + str(x[1]) for x in mydict.items()]
    if title is not None:
        output_list = [title]
        output_list.extend(tmp)
    else:
        output_list = tmp
    list2file(path, output_list, 'a')


def dict2pkl(mydict, path):
    """
    Saves a dictionary object into a pkl file.
    :param mydict: dictionary to save in a file
    :param path: path where my_dict is stored
    :return:
    """
    import cPickle
    if path[-4:] == '.pkl':
        extension = ''
    else:
        extension = '.pkl'
    with open(path + extension, 'w') as f:
        cPickle.dump(mydict, f, protocol=cPickle.HIGHEST_PROTOCOL)


def pkl2dict(path):
    """
    Loads a dictionary object from a pkl file.

    :param path: Path to the pkl file to load
    :return: Dict() containing the loaded pkl
    """
    import cPickle
    return cPickle.load(open(path))
