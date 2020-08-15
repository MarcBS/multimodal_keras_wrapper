import json
import logging
import codecs
import numpy as np
import pandas as pd

from keras_wrapper.extra.read_write import list2file
from localization_utilities import *


########################################
# EVALUATION FUNCTIONS SELECTOR
########################################


def get_coco_score(pred_list, verbose, extra_vars, split):
    """
    COCO challenge metrics
    :param pred_list: dictionary of hypothesis sentences (id, sentence)
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: extra variables, here are:
            extra_vars['references'] - dict mapping sample indices to list with all valid captions (id, [sentences])
            extra_vars['tokenize_f'] - tokenization function used during model training (used again for validation)
            extra_vars['detokenize_f'] - detokenization function used during model training (used again for validation)
            extra_vars['tokenize_hypotheses'] - Whether tokenize or not the hypotheses during evaluation
    :param split: split on which we are evaluating
    :return: Dictionary with the coco scores
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.meteor import accepted_langs
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.ter.ter import Ter

    gts = extra_vars[split]['references']
    if extra_vars.get('tokenize_hypotheses', False):
        hypo = {idx: map(extra_vars['tokenize_f'], [lines.strip()]) for (idx, lines) in enumerate(pred_list)}
    else:
        hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(pred_list)}

    # Tokenize refereces if needed
    if extra_vars.get('tokenize_references', False):
        refs = {idx: map(extra_vars['tokenize_f'], gts[idx]) for idx in gts.keys()}
    else:
        refs = gts

    # Detokenize references if needed.    
    if extra_vars.get('apply_detokenization', False):
        refs = {idx: map(extra_vars['detokenize_f'], refs[idx]) for idx in refs}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Ter(), "TER"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    if extra_vars.get('language', 'en') in accepted_langs:
        scorers.append((Meteor(language=extra_vars['language']), "METEOR"))

    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(refs, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    if verbose > 0:
        logging.info('Computing coco scores on the %s split...' % split)
    for metric in sorted(final_scores):
        value = final_scores[metric]
        logging.info(metric + ': ' + str(value))

    return final_scores


def eval_vqa(pred_list, verbose, extra_vars, split):
    """
    VQA challenge metrics
    :param pred_list: dictionary of hypothesis sentences (id, sentence)
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: extra variables
            extra_vars['quesFile'] - path to the .json file where the questions are stored
            extra_vars['annFile'] - path to the .json file where the annotated answers are stored
            extra_vars['question_ids'] - question identifiers
    :param split: split on which we are evaluating
    :return: Dictionary of VQA accuracies
    """
    import datetime
    from pycocoevalcap.vqa import vqaEval, visual_qa
    from read_write import list2vqa

    quesFile = extra_vars[split]['quesFile']
    annFile = extra_vars[split]['annFile']

    # create temporal resFile
    resFile = 'tmp_res_file_' + str(datetime.datetime.now()) + '.json'
    list2vqa(resFile, pred_list, extra_vars[split]['question_ids'])

    # create vqa object and vqaRes object
    vqa_ = visual_qa.VQA(annFile, quesFile)
    vqaRes = vqa_.loadRes(resFile, quesFile)
    vqaEval_ = vqaEval.VQAEval(vqa_, vqaRes,
                               n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval_.evaluate()
    os.remove(resFile)  # remove temporal file

    # get results
    acc_overall = vqaEval_.accuracy['overall']
    acc_yes_no = vqaEval_.accuracy['perAnswerType']['yes/no']
    acc_number = vqaEval_.accuracy['perAnswerType']['number']
    acc_other = vqaEval_.accuracy['perAnswerType']['other']
    # acc_per_class = vqaEval_.accuracy['perAnswerClass']
    # acc_class_normalized = vqaEval_.accuracy['classNormalizedOverall']

    if verbose > 0:
        logging.info('VQA Metric: Accuracy yes/no is {0}, other is {1}, number is {2}, overall is {3}'.
                     format(acc_yes_no, acc_other, acc_number, acc_overall))  # , acc_class_normalized))
    return {'overall accuracy': acc_overall,
            'yes/no accuracy': acc_yes_no,
            'number accuracy': acc_number,
            'other accuracy': acc_other}


def multilabel_metrics(pred_list, verbose, extra_vars, split):
    """
    Multiclass classification metrics. see multilabel ranking metrics in sklearn library for more info:
        http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics

    :param pred_list: dictionary of hypothesis sentences
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: extra variables
                        extra_vars['word2idx'] - dictionary mapping from words to indices
                        extra_vars['references'] - list of GT labels
    :param split: split on which we are evaluating
    :return: Dictionary of multilabel metrics
    """
    from sklearn import metrics as sklearn_metrics

    word2idx = extra_vars[split]['words2idx']

    # check if an additional dictionary matching raw to basic and general labels is provided
    # in that case a more general evaluation will be considered
    raw2basic = extra_vars[split].get('raw2basic', None)
    if raw2basic is not None:
        logging.info('Applying general evaluation with raw2basic dictionary.')

    if raw2basic is None:
        n_classes = len(word2idx)
    else:
        basic_values = set(raw2basic.values())
        n_classes = len(basic_values)
    n_samples = len(pred_list)

    # Create prediction matrix
    y_pred = np.zeros((n_samples, n_classes))
    for i_s, sample in enumerate(pred_list):
        for word in sample:
            if raw2basic is None:
                y_pred[i_s, word2idx[word]] = 1
            else:
                word = word.strip()
                y_pred[i_s, raw2basic[word]] = 1

    # Prepare GT
    gt_list = extra_vars[split]['references']

    if raw2basic is None:
        #y_gt = np.array(gt_list)
        y_gt = np.zeros((n_samples, n_classes))
        for i_s, sample in enumerate(gt_list):
            for word in sample.split():
                y_gt[i_s, word2idx[word]] = 1
    else:
        idx2word = {v: k for k, v in word2idx.iteritems()}
        y_gt = np.zeros((n_samples, n_classes))
        for i_s, sample in enumerate(gt_list):
            for raw_idx, is_active in enumerate(sample):
                if is_active:
                    word = idx2word[raw_idx].strip()
                    y_gt[i_s, raw2basic[word]] = 1

    # Compute Coverage Error
    coverr = sklearn_metrics.coverage_error(y_gt, y_pred)
    # Compute Label Ranking AvgPrec
    avgprec = sklearn_metrics.label_ranking_average_precision_score(y_gt, y_pred)
    # Compute Label Ranking Loss
    rankloss = sklearn_metrics.label_ranking_loss(y_gt, y_pred)
    # Compute Precision, Recall and F1 score
    precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(y_gt, y_pred, average='micro')

    if verbose > 0:
        logging.info(
            '"coverage_error" (best: avg labels per sample = %f): %f' % (float(np.sum(y_gt)) / float(n_samples), coverr))
        logging.info('Label Ranking "average_precision" (best: 1.0): %f' % avgprec)
        logging.info('Label "ranking_loss" (best: 0.0): %f' % rankloss)
        logging.info('precision: %f' % precision)
        logging.info('recall: %f' % recall)
        logging.info('f1: %f' % f1)

    return {'coverage_error': coverr,
            'average_precision': avgprec,
            'ranking_loss': rankloss,
            'precision': precision,
            'recall': recall,
            'f1': f1}


def multilabel_metrics_custom(pred_list, extra_vars, split, verbose=1):
    """
    Multiclass classification metrics. See multilabel ranking metrics in sklearn library for more info:
        http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
    :param pred_list: list of predictions
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: dictionary extra variables. 
            Must contain:
                - ['n_classes'] with the total number of existent classes
                - [split]['references'] with the GT values corresponding to each sample of the current data split
            Optional:
                - ['words_as_classes'] as True if our output is of type text and each word in the sentence is a label for a video frame
                - ['save_classes_report'] as True if we want to ave a classification report for each class
                - ['save_path'] as the path to save a classification report for each class
                - ['target_names'] as the classes names for the classification report
    :param split: split of the data where we are applying the evaluation
    :return: dictionary of multiclass metrics
    """
    from sklearn import metrics as sklearn_metrics

    n_classes = extra_vars['n_classes']
    
    n_samples = len(pred_list)
    logging.info("---# of samples: "+str(n_samples))
    
    gt_list = extra_vars[split]['references']
    try:
        values_gt = gt_list.values()
    except:
        values_gt = gt_list

    # Extra case added. For text output when each word is the label/class of each corresponding video frame.
    if extra_vars.get('words_as_classes', False):

        # Create prediction matrix
        y_pred = np.zeros((n_samples, n_classes))
        y_gt = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            if i==0:
                print values_gt[i]
                print pred_list[i]
            
            line_gt = values_gt[i].split()  # List of ground truth values
            for j, value in enumerate(line_gt):
                value = int(value)
                y_gt[i, value] = 1

            for j, value in enumerate(pred_list[i]):
                try:
                    value = int(value)
                    y_pred[i, value] = 1 
                except:
                    if value=='':
                        y_pred[i, 0] = 1 
                

    # Compute Precision, Recall and F1 score
    precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(y_gt, y_pred, average='samples')

    if False:
        df = pd.DataFrame({"precision": precision, "recall": recall, "f1 score": f1_classes, "support": support}, index=extra_vars['target_names'])

        # Don't use background for metrics computation
        support = support[1:]
        precision = precision[1:]
        recall = recall[1:]

        #print "F1 score computed with average precision and recalls"
        #precision_ = np.mean(np.take(precision, np.nonzero(support)))
        #recall_ = np.mean(np.take(recall, np.nonzero(support)))
        #f1_ = 2*precision_*recall_ / (precision_+recall_)
        #print f1_
        #print
        
        # Precision and recall arrays, only of elements with non zero support
        precision_ = np.nan_to_num(np.take(precision, np.nonzero(support))[0])
        recall_ = np.nan_to_num(np.take(recall, np.nonzero(support))[0])

        # F1-score per class
        f1_ = 2*precision_*recall_ / (precision_+recall_)
        f1_ = np.nan_to_num(f1_)  # It is possible to divide by 0 and get nan values. So we set those to 0.0

        # Get mean metrics between all classes
        precision = np.mean(precision_)
        recall = np.mean(recall_)
        f1 = np.mean(f1_)

        accuracy_balanced = recall  # In this case these metrics are the same

        if extra_vars.get('save_classes_report', False):
            save_path = extra_vars['save_path']

            df.to_csv(save_path + "/classification_report.csv", index = True, sep=',', encoding='utf-8')

            report = sklearn_metrics.classification_report(y_gt, y_pred, target_names=extra_vars['target_names'])
            with codecs.open(save_path + "/classification_report_real.txt",'w','utf-8') as f:
                f.write(report)
                f.close()

            with open(save_path + "/f1-score_classes_real.txt", 'w') as f:
                for v in f1_classes:
                    f.write(str(v)+'\n')
                f.close()

    if verbose > 0:
        logging.info('Precision: %f' % precision)
        logging.info('Recall: %f' % recall)
        logging.info('F1 score: %f' % f1)

    dic_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return dic_metrics


def multiclass_metrics_custom(pred_list, extra_vars, split, verbose=1):
    """
    Multiclass classification metrics. See multilabel ranking metrics in sklearn library for more info:
        http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
    :param pred_list: list of predictions
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: dictionary extra variables. 
            Must contain:
                - ['n_classes'] with the total number of existent classes
                - [split]['references'] with the GT values corresponding to each sample of the current data split
            Optional:
                - ['words_as_classes'] as True if our output is of type text and each word in the sentence is a label for a video frame
                - ['save_classes_report'] as True if we want to ave a classification report for each class
                - ['save_path'] as the path to save a classification report for each class
                - ['target_names'] as the classes names for the classification report
    :param split: split of the data where we are applying the evaluation
    :return: dictionary of multiclass metrics
    """
    from sklearn import metrics as sklearn_metrics

    n_classes = extra_vars['n_classes']
    
    n_samples = len(pred_list)
    logging.info("---# of samples: "+str(n_samples))
    
    gt_list = extra_vars[split]['references']
    try:
        values_gt = gt_list.values()
    except:
        values_gt = gt_list

    # Extra case added. For text output when each word is the label/class of each corresponding video frame.
    if extra_vars.get('words_as_classes', False):

        n_classes += 3

        # Create flat list for ground truth and predictions values
        flat_gt = []
        flat_pred = []

        for i in range(n_samples):
            line_gt = values_gt[i].split()  # List of ground truth values

            for j, value in enumerate(line_gt):  # Only iterate until number of ground truth classes, as we want to ignore '<pad>' for the metrics
                flat_gt.append(int(value)) 
                pred = pred_list[i][j]
                try:
                    flat_pred.append(int(pred))
                except:
                    if pred == '<pad>':
                        flat_pred.append(n_classes-3)  # n_classes-3
                    elif pred == '<unk>':
                        flat_pred.append(n_classes-2)  # n_classes-2
                    elif pred == '<null>':
                        flat_pred.append(n_classes-1)  # n_classes-1
        
        # Create prediction matrix
        y_pred = np.zeros((len(flat_gt), n_classes))
        y_gt = np.zeros((len(flat_gt), n_classes))
        counts_per_class = np.zeros((n_classes,))

        for i in range(len(flat_gt)):
            y_pred[i, flat_pred[i]] = 1
            y_gt[i, flat_gt[i]] = 1
            counts_per_class[flat_gt[i]] += 1

    else:

        # Create prediction matrix
        y_pred = np.zeros((n_samples, n_classes))
        y_gt = np.zeros((n_samples, n_classes))
        counts_per_class = np.zeros((n_classes,))

        pred_class_list = [np.argmax(sample_score) for sample_score in pred_list]

        for i_s, pred_class in enumerate(pred_class_list):
            y_pred[i_s, pred_class] = 1

        for i_s, gt_class in enumerate(values_gt):
            y_gt[i_s, gt_class] = 1
            counts_per_class[gt_class] += 1


    # Compute accuracy
    top_n_accuracies = [3, 5]
    accuracy = sklearn_metrics.accuracy_score(y_gt, y_pred)
    acc_top_n = {}
    for topn in top_n_accuracies:
        acc_top_n[topn] = __top_k_accuracy(y_gt, y_pred, topn)

    # Compute Precision, Recall and F1 score
    precision, recall, f1_classes, support = sklearn_metrics.precision_recall_fscore_support(y_gt, y_pred, average=None)

    if extra_vars.get('save_classes_report', False):
        df = pd.DataFrame({"precision": precision, "recall": recall, "f1 score": f1_classes, "support": support}, index=extra_vars['target_names'])

    # Don't use background for metrics computation
    support = support[1:]
    precision = precision[1:]
    recall = recall[1:]

    #print "F1 score computed with average precision and recalls"
    #precision_ = np.mean(np.take(precision, np.nonzero(support)))
    #recall_ = np.mean(np.take(recall, np.nonzero(support)))
    #f1_ = 2*precision_*recall_ / (precision_+recall_)
    #print f1_
    #print
    
    # Precision and recall arrays, only of elements with non zero support
    precision_ = np.nan_to_num(np.take(precision, np.nonzero(support))[0])
    recall_ = np.nan_to_num(np.take(recall, np.nonzero(support))[0])

    # F1-score per class
    f1_ = 2*precision_*recall_ / (precision_+recall_)
    f1_ = np.nan_to_num(f1_)  # It is possible to divide by 0 and get nan values. So we set those to 0.0

    # Get mean metrics between all classes
    precision = np.mean(precision_)
    recall = np.mean(recall_)
    f1 = np.mean(f1_)

    accuracy_balanced = recall  # In this case these metrics are the same

    if extra_vars.get('save_classes_report', False):
        save_path = extra_vars['save_path']

        df.to_csv(save_path + "/classification_report.csv", index = True, sep=',', encoding='utf-8')

        report = sklearn_metrics.classification_report(y_gt, y_pred, target_names=extra_vars['target_names'])
        with codecs.open(save_path + "/classification_report.txt",'w','utf-8') as f:
            f.write(report)
            f.close()

        with open(save_path + "/f1-score_classes.txt", 'w') as f:
            for v in f1_classes:
                f.write(str(v)+'\n')
            f.close()
    

    if verbose > 0:
        logging.info('Accuracy: %f' % accuracy)
        for topn in top_n_accuracies:
            logging.info('Accuracy top-%d: %f' % (topn, acc_top_n[topn]))
        logging.info('Balanced Accuracy: %f' % accuracy_balanced)
        logging.info('Precision: %f' % precision)
        logging.info('Recall: %f' % recall)
        logging.info('F1 score: %f' % f1)

    dic_metrics = {'accuracy': accuracy,
            'accuracy_balanced': accuracy_balanced,
            'precision': precision,
            'recall': recall,
            'f1': f1}

    for topn in top_n_accuracies:
        dic_metrics['accuracy_top_'+str(topn)] = acc_top_n[topn]

    return dic_metrics


def multiclass_metrics(pred_list, verbose, extra_vars, split):
    """
    Multiclass classification metrics. See multilabel ranking metrics in sklearn library for more info:
        http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
    :param pred_list: list of predictions
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: dictionary extra variables. 
            Must contain:
                - ['n_classes'] with the total number of existent classes
                - [split]['references'] with the GT values corresponding to each sample of the current data split
            Optional:
                - ['words_as_classes'] as True if our output is of type text and each word in the sentence is a label for a video frame
                - ['save_classes_report'] as True if we want to ave a classification report for each class
                - ['save_path'] as the path to save a classification report for each class
                - ['target_names'] as the classes names for the classification report
    :param split: split of the data where we are applying the evaluation
    :return: dictionary of multiclass metrics
    """
    from sklearn import metrics as sklearn_metrics

    n_classes = extra_vars['n_classes']
    
    n_samples = len(pred_list)
    logging.info("---# of samples: "+str(n_samples))
    
    gt_list = extra_vars[split]['references']
    try:
        values_gt = gt_list.values()
    except:
        values_gt = gt_list

    # Extra case added. For text output when each word is the label/class of each corresponding video frame.
    if extra_vars.get('words_as_classes', False):

        # Create flat list for ground truth and predictions values
        flat_gt = []
        flat_pred = []

        for i in range(n_samples):
            line_gt = values_gt[i].split()  # List of ground truth values

            for j, value in enumerate(line_gt):  # Only iterate until number of ground truth classes, as we want to ignore '<pad>' for the metrics
                flat_gt.append(int(value)) 
                pred = pred_list[i][j]
                try:
                    flat_pred.append(int(pred))
                except:
                    #print "Exception. '<pad>' value received. Set to value n_classes-1." 
                    flat_pred.append(n_classes-1)
        
        # Create prediction matrix
        y_pred = np.zeros((len(flat_gt), n_classes))
        y_gt = np.zeros((len(flat_gt), n_classes))
        counts_per_class = np.zeros((n_classes,))

        for i in range(len(flat_gt)):
            y_pred[i, flat_pred[i]] = 1
            y_gt[i, flat_gt[i]] = 1
            counts_per_class[flat_gt[i]] += 1

    else:

        # Create prediction matrix
        y_pred = np.zeros((n_samples, n_classes))
        y_gt = np.zeros((n_samples, n_classes))
        counts_per_class = np.zeros((n_classes,))

        pred_class_list = [np.argmax(sample_score) for sample_score in pred_list]

        for i_s, pred_class in enumerate(pred_class_list):
            y_pred[i_s, pred_class] = 1

        for i_s, gt_class in enumerate(values_gt):
            y_gt[i_s, gt_class] = 1
            counts_per_class[gt_class] += 1

        # Apply balanced accuracy per class
        inverse_counts_per_class = [sum(counts_per_class) - c_i for c_i in counts_per_class]
        weights_per_class = [float(c_i) / sum(inverse_counts_per_class) for c_i in inverse_counts_per_class]
        sample_weights = np.zeros((n_samples,))
        for i_s, gt_class in enumerate(values_gt):
            sample_weights[i_s] = weights_per_class[gt_class]

    # Compute accuracy
    top_n_accuracies = [3, 5]
    accuracy = sklearn_metrics.accuracy_score(y_gt, y_pred)
    acc_top_n = {}
    for topn in top_n_accuracies:
        acc_top_n[topn] = __top_k_accuracy(y_gt, y_pred, topn)
    # accuracy_balanced = sklearn_metrics.accuracy_score(y_gt, y_pred, sample_weight=sample_weights, )

    # The following two lines should both provide the same measure (balanced accuracy)
    _, accuracy_balanced, _, _ = sklearn_metrics.precision_recall_fscore_support(y_gt, y_pred, average='macro')
    #accuracy_balanced = sklearn_metrics.balanced_accuracy_score(y_gt, y_pred)

    # Compute Precision, Recall and F1 score
    precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(y_gt, y_pred, average='micro')
    
    if extra_vars.get('save_classes_report', False):
        save_path = extra_vars['save_path']
        report = sklearn_metrics.classification_report(y_gt, y_pred, target_names=extra_vars['target_names'])
        with codecs.open(save_path + "/classification_report_last_epoch.txt",'w','utf-8') as f:
        #with open(save_path + "/classification_report_last_epoch.txt", 'w') as f:
            f.write(report)
    
    if verbose > 0:
        logging.info('Accuracy: %f' % accuracy)
        for topn in top_n_accuracies:
            logging.info('Accuracy top-%d: %f' % (topn, acc_top_n[topn]))
        logging.info('Balanced Accuracy: %f' % accuracy_balanced)
        logging.info('Precision: %f' % precision)
        logging.info('Recall: %f' % recall)
        logging.info('F1 score: %f' % f1)

    dic_metrics = {'accuracy': accuracy,
            'accuracy_balanced': accuracy_balanced,
            'precision': precision,
            'recall': recall,
            'f1': f1}
    for topn in top_n_accuracies:
        dic_metrics['accuracy_top_'+str(topn)] = acc_top_n[topn]

    return dic_metrics


def __top_k_accuracy(truths, preds, k):
    """
    Both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
    :param preds:
    :param truths:
    :param k:
    :return:
    """
    best_k = np.argsort(preds, axis=1)[:, -k:]
    ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_k[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]


def semantic_segmentation_accuracy(pred_list, verbose, extra_vars, split):
    """
    Semantic Segmentation Accuracy metric

    :param pred_list: list of predictions
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: dictionary extra variables. Must contain:
                       - ['n_classes'] with the total number of existent classes
                       - [split]['references'] with the GT values corresponding to each sample of the current data split
                       - [discard_classes] with the classes not taken into account in the performance evaluation
    :param split: split of the data where we are applying the evaluation
    :return: dictionary containing the semantic global accuracy
    """
    from sklearn import metrics as sklearn_metrics

    n_classes = extra_vars['n_classes']
    gt_list = extra_vars[split]['references']
    discard_classes = extra_vars['discard_classes']  # [11]

    pred_class_list = []
    for sample_score in pred_list:
        pred_class_list += list(np.argmax(sample_score, axis=1))
    n_samples = len(pred_class_list)

    values_gt = []
    for gt in gt_list:
        values_gt += list(gt)

    # Create ground truth and prediction matrices
    y_gt = np.zeros((n_samples, n_classes))
    y_pred = np.zeros((n_samples, n_classes))

    ind_i = 0
    for i_s, (gt_class, pred_class) in enumerate(zip(values_gt, pred_class_list)):
        if not any([d == gt_class for d in discard_classes]):
            y_pred[ind_i, pred_class] = 1
            y_gt[ind_i, gt_class] = 1
            ind_i += 1
    n_samples = ind_i

    # Cut to real n_samples size
    y_gt = y_gt[:n_samples]
    y_pred = y_pred[:n_samples]

    # Compute Coverage Error
    accuracy = sklearn_metrics.accuracy_score(y_gt, y_pred)
    if verbose > 0:
        logging.info('Accuracy: %f' % accuracy)

    return {'semantic global accuracy': accuracy}


def semantic_segmentation_meaniou(pred_list, verbose, extra_vars, split):
    """
    Semantic Segmentation Mean IoU metric

    :param pred_list: list of predictions
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: dictionary extra variables. Must contain:
                    - ['n_classes'] with the total number of existent classes
                    - [split]['references'] with the GT values corresponding to each sample of the current data split
                    - [discard_classes] with the classes not taken into account in the performance evaluation
    :param split: split of the data where we are applying the evaluation
    :return: dictionary containing the mean IoU and the semantic global accuracy metrics
    """
    from sklearn import metrics as sklearn_metrics

    n_classes = extra_vars['n_classes']
    gt_list = extra_vars[split]['references']
    discard_classes = extra_vars['discard_classes']

    np.save('/media/HDD_3TB/bea/gt.npy', gt_list)
    np.save('/media/HDD_3TB/bea/pred.npy', pred_list)

    pred_class_list = []
    for sample_score in pred_list:
        pred_class_list += list(np.argmax(sample_score, axis=1))
    n_samples = len(pred_class_list)

    values_gt = []
    for gt in gt_list:
        values_gt += list(gt)

    # Create ground truth and prediction matrices
    y_gt = np.zeros((n_samples,))
    y_pred = np.zeros((n_samples,))

    ind_i = 0
    for i_s, (gt_class, pred_class) in enumerate(zip(values_gt, pred_class_list)):
        if not any([d == gt_class for d in discard_classes]):
            y_gt[ind_i] = gt_class
            y_pred[ind_i] = pred_class
            ind_i += 1
    n_samples = ind_i

    # Cut to real n_samples size
    y_gt = y_gt[:n_samples]
    y_pred = y_pred[:n_samples]

    # Computer mean IoU (Jaccard index) and pixel accuracy
    cm = sklearn_metrics.confusion_matrix(y_gt, y_pred)
    cm_t = cm.transpose()

    inter = np.zeros(n_classes - len(discard_classes))
    union = np.zeros(n_classes - len(discard_classes))
    ind_i = 0

    for l in range(0, n_classes):
        if not any([d == l for d in discard_classes]):
            tp = cm[l][l]
            fn = np.sum(cm[l]) - tp
            fp = np.sum(cm_t[l]) - tp
            inter[ind_i] = float(tp)
            union[ind_i] = float(tp + fp + fn)
            ind_i += 1

    mean_iou = np.mean(inter / union)
    acc = np.sum(inter) / np.sum(cm)
    if verbose > 0:
        logging.info('Mean IoU: %f' % mean_iou)
        logging.info('Accuracy: %f' % float(acc))

    return {'mean IoU': mean_iou, 'semantic global accuracy': acc}


def averagePrecision(pred_list, verbose, extra_vars, split):
    """
    Computes a Precision-Recall curve and its associated mAP score given a set of precalculated reports.
    The parameter "report_all" must include the following information for each sample:
        [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y]
    The parameter 'threshods' defines the range of min_prediction_scores to be tested for computing the
    precision-recall curve.

    :param pred_list: list of hypotheses with the following info [predicted_bboxes, predicted_Y, predicted_scores]
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: extra variables, here are:
                extra_vars[split]['references'] - list of samples with the following info [GT_bboxes, GT_Y]
                extra_vars['n_classes'] - number of classes in the detection task
    :param split: split of the data where we are applying the evaluation
    :return: dictionary containing the aforementioned metrics.
    """

    thresholds = np.arange(0, 1, 0.1)
    fixedIoU = 0.5

    gts = extra_vars[split]['references']
    n_classes = extra_vars['n_classes']

    n_samples = len(pred_list)
    n_thresholds = len(thresholds)

    # prepare variables for storing all precision-recall values
    general_measures = [[] for _ in range(n_thresholds)]
    class_measures = [[] for _ in range(n_thresholds)]

    # compute precision-recall measures for each min_prediction_score threshold
    for thres in range(n_thresholds):

        # Counters for computing general precision-recall curve
        FP = 0
        TP = 0
        FN = 0
        total_GT = 0
        total_pred = 0

        # Counters for computing class-specific precision-recall curve
        TP_classes = np.zeros((n_classes,))
        FP_classes = np.zeros((n_classes,))
        FN_classes = np.zeros((n_classes,))
        total_GT_classes = np.zeros((n_classes,))
        total_pred_classes = np.zeros((n_classes,))

        for s in range(n_samples):

            # Re-use report of the current image provided in the parameters (or recently computed)
            [predicted_bboxes, predicted_Y, predicted_scores] = pred_list[s]
            [GT_bboxes, GT_Y] = gts[s]

            # Filter bounding boxes which are above the current threshold 'thres'
            aux_predicted_bboxes = []
            aux_predicted_Y = []
            aux_predicted_scores = []
            for pos, score in enumerate(predicted_scores):
                if score > thresholds[thres]:
                    aux_predicted_bboxes.append(predicted_bboxes[pos])
                    aux_predicted_Y.append(predicted_Y[pos])
                    aux_predicted_scores.append(predicted_scores[pos])
            predicted_bboxes = aux_predicted_bboxes
            predicted_Y = aux_predicted_Y
            predicted_scores = aux_predicted_scores

            # Compute TPs, FPs and FNs
            [TP_, FP_, FN_, TP_c, FP_c, FN_c] = _computeMeasures(fixedIoU, n_classes, predicted_bboxes, predicted_Y,
                                                                 predicted_scores, GT_bboxes, GT_Y)

            total_GT += len(GT_bboxes)
            total_pred += len(predicted_bboxes)
            TP += TP_
            FP += FP_
            FN += FN_

            for pos in range(len(TP_c)):
                TP_classes[pos] += TP_c[pos]
                FP_classes[pos] += FP_c[pos]
                FN_classes[pos] += FN_c[pos]

            for y in GT_Y:
                total_GT_classes[y] += 1
            for y in predicted_Y:
                total_pred_classes[y] += 1

        # Compute general precision / recall / accuracy measures
        if (TP + FP) == 0:
            precision = 0.0
        else:
            precision = float(TP) / (TP + FP)
        if (TP + FN) == 0:
            recall = 0.0
        else:
            recall = float(TP) / (TP + FN)
        if (FP + FN + TP) == 0:
            accuracy = 0.0
        else:
            accuracy = float(TP) / (FP + FN + TP)

        # Compute class-specific precision - recall
        precision_classes = np.zeros((n_classes,))
        recall_classes = np.zeros((n_classes,), dtype=int)
        accuracy_classes = np.zeros((n_classes,), dtype=int)
        for c in range(n_classes):
            TP = TP_classes[c]
            FP = FP_classes[c]
            FN = FN_classes[c]
            if (TP + FP) == 0:
                precision_classes[c] = 0.0
            else:
                precision_classes[c] = float(TP) / (TP + FP)
            if (TP + FN) == 0:
                recall_classes[c] = 0.0
            else:
                recall_classes[c] = float(TP) / (TP + FN)
            if (FP + FN + TP) == 0:
                accuracy_classes[c] = 0.0
            else:
                accuracy_classes[c] = float(TP) / (FP + FN + TP)

        # store results
        general_measures[thres] = [precision, recall, accuracy, total_GT, total_pred]
        class_measures[thres] = [precision_classes, recall_classes, accuracy_classes, total_GT_classes,
                                 total_pred_classes]

    # Compute average precision (AP) (adapted from PASCAL VOC evaluation code VOCap.m)
    prec = [general_measures[thres][0] for thres in range(n_thresholds)][::-1]
    rec = [general_measures[thres][1] for thres in range(n_thresholds)][::-1]
    AP = _computeAP(prec, rec)

    """
    for thres in range(n_thresholds):
        logging.info(
            'Evaluation results (score >= %0.2f):\n\tPrecision: %f\n\tRecall: %f\n
            \tAccuracy: %f\n\tSamples GT: %d\n\tSamples predicted: %d' %
            (thresholds[thres],
             general_measures[thres][0], general_measures[thres][1], general_measures[thres][2],
             general_measures[thres][3], general_measures[thres][4]))
    """

    if verbose > 0:
        logging.info('Average Precision (AP): %f' % AP)
        logging.info(
            'Evaluation results (score >= %0.2f):\n\tPrecision: %f\n\tRecall: %f\n'
            '\tAccuracy: %f\n\tSamples GT: %d\n\tSamples predicted: %d' %
            (thresholds[5],
             general_measures[5][0], general_measures[5][1], general_measures[5][2],
             general_measures[5][3], general_measures[5][4]))

    return_dict = dict()
    return_dict['AP'] = AP
    for thres in range(n_thresholds):
        return_dict['precision_' + str(thres)] = general_measures[thres][0]
        return_dict['recall_' + str(thres)] = general_measures[thres][1]
        return_dict['accuracy_' + str(thres)] = general_measures[thres][2]
        return_dict['total_GT_' + str(thres)] = general_measures[thres][3]
        return_dict['total_pred_' + str(thres)] = general_measures[thres][4]
        for c in range(n_classes):
            return_dict['precision_' + str(thres) + '_' + str(c)] = class_measures[thres][0][c]
            return_dict['recall_' + str(thres) + '_' + str(c)] = class_measures[thres][1][c]
            return_dict['accuracy_' + str(thres) + '_' + str(c)] = class_measures[thres][2][c]
            return_dict['total_GT_' + str(thres) + '_' + str(c)] = class_measures[thres][3][c]
            return_dict['total_pred_' + str(thres) + '_' + str(c)] = class_measures[thres][4][c]

    return return_dict


def _computeAP(prec, rec):
    AP = 0.0
    n_thresholds = len(prec)
    prec = [0] + prec + [0]
    rec = [0] + rec + [1]
    for i in range(n_thresholds, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    for i in range(n_thresholds + 1):
        if rec[i + 1] != rec[i]:
            # increment of recall times precision
            AP += (rec[i + 1] - rec[i]) * prec[i]
    return AP


def _computeMeasures(IoU, n_classes, predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y):
    """
    Computes TP, FP, and FN given a set of GT and Prediction BBoxes

    :param IoU:
    :param n_classes:
    :param predicted_bboxes:
    :param predicted_Y:
    :param predicted_scores:
    :param GT_bboxes:
    :param GT_Y:
    :return:
    """
    # General counters (without applying class distinctions)
    TP = 0
    FP = 0
    FN = 0

    # Class-specific counters
    TP_classes = np.zeros(n_classes)
    FP_classes = np.zeros(n_classes)
    FN_classes = np.zeros(n_classes)

    if len(predicted_Y) > 0:
        using_recognition = True
    else:
        using_recognition = False

    # Compute IoU for each pair of bounding boxes in (GT, pred)
    iou_values = []
    pred_ids = []
    match_bboxes = []
    for i, gt in enumerate(GT_bboxes):
        for j, pred in enumerate(predicted_bboxes):
            # compute IoU
            iou_values.append(computeIoU(gt, pred))
            pred_ids.append(j)
            match_bboxes.append([i, j])

    # Compute matchings (sorted by IoU)
    final_matches = []  # stores the final indices for [gt,pred] matches
    matched_gt = [False for _ in range(len(GT_bboxes))]
    matched_pred = [False for _ in range(len(predicted_bboxes))]
    # max_iou = np.argsort(np.array(iou_values, dtype=np.float))[::-1]
    max_scores = np.argsort(np.array(predicted_scores, dtype=np.float))[::-1]

    # Sort predictions by "scores"
    i = 0
    while i < len(max_scores) and not all(matched_gt):
        # m = match_bboxes[max_iou[i]]
        this_pred_id = max_scores[i]
        m_list = [[p_, match_bboxes[p_]] for p_, p in enumerate(pred_ids) if p == this_pred_id]
        this_iou = [iou_values[p] for p, m in m_list]
        max_iou = np.argsort(np.array(this_iou, dtype=np.float))[::-1]

        # Sort GT by IoU
        j = 0
        while j < len(max_iou) and not matched_pred[this_pred_id]:  # if pred has not been matched yet
            j_ind = max_iou[j]
            if this_iou[j_ind] > IoU and not matched_gt[m_list[j_ind][1][0]]:
                # Assign match
                matched_gt[m_list[j_ind][1][0]] = True
                matched_pred[this_pred_id] = True
                final_matches.append(m_list[j_ind][1])
            j += 1
        i += 1

    # Compute FPs, FNs and TPs on the current image
    for m in matched_gt:
        if m:
            TP += 1
        else:
            FN += 1
    for m in matched_pred:
        if not m:
            FP += 1

    # Compute class-specific counters
    if using_recognition:
        # Check matching pairs
        for m in final_matches:
            y_gt = GT_Y[m[0]]
            y_pred = predicted_Y[m[1]]

            # GT and pred coincide
            if y_gt == y_pred:
                TP_classes[y_gt] += 1
            # Missclassified but correctly localized
            else:
                FN_classes[y_gt] += 1
                FP_classes[y_pred] += 1
        # Check missed GT bboxes
        for i, m in enumerate(matched_gt):
            if not m:
                FN_classes[GT_Y[i]] += 1
        # Check mislocalized Pred bboxes
        for i, m in enumerate(matched_pred):
            if not m:
                FP_classes[predicted_Y[i]] += 1

    return [TP, FP, FN, TP_classes, FP_classes, FN_classes]


def compute_perplexity(y_pred, y_true, verbose, split, mask=None):
    """
    Computes perplexity
    :param y_pred:
    :param y_true:
    :param verbose:
    :param split:
    :param mask:
    :return:
    """

    if mask is not None:
        y_pred /= np.sum(y_pred, axis=-1, keepdims=True)
        mask = np.reshape(mask, y_true.shape[:-1])[:, :, None]
        truth_mask = (y_true * mask).flatten().nonzero()[0]
        predictions = y_pred.flatten()[truth_mask]
        ppl = np.power(2, np.mean(-np.log2(predictions)))
        if verbose > 0:
            logging.info('Computing perplexity scores on the %s split...' % split)
            logging.info('PPL: ' + str(ppl))
        return ppl
    else:
        ppl = np.power(2, np.mean(-np.log2(y_pred)))
        if verbose > 0:
            logging.info('Computing perplexity scores on the %s split...' % split)
            logging.info('PPL: ' + str(ppl))
        return ppl


########################################
# AUXILIARY FUNCTIONS
########################################

def vqa_store(question_id_list, answer_list, path):
    """
    Saves the answers on question_id_list in the VQA-like format.

    In:
        question_id_list - list of the question ids
        answer_list - list with the answers
        path - path where the file is saved
    """
    question_answer_pairs = []
    assert len(question_id_list) == len(answer_list), \
        'must be the same number of questions and answers'
    for q, a in zip(question_id_list, answer_list):
        question_answer_pairs.append({'question_id': q, 'answer': str(a)})
    with open(path, 'w') as f:
        json.dump(question_answer_pairs, f)


def caption_store(samples, path):
    with open(path, 'w') as f:
        print >> f, '\n'.join(samples)


# List of evaluation functions and their identifiers (will be used in params['METRICS'])
selectMetric = {
    'vqa': eval_vqa,  # Metric for the VQA challenge
    'coco': get_coco_score,  # MS COCO evaluation library (BLEU, METEOR and CIDEr scores)
    'multilabel_metrics': multilabel_metrics,  # Set of multilabel classification metrics from sklearn
    'multiclass_metrics': multiclass_metrics,  # Set of multiclass classification metrics from sklearn
    'AP': averagePrecision,
    'sem_seg_acc': semantic_segmentation_accuracy,
    'sem_seg_iou': semantic_segmentation_meaniou,
    'ppl': compute_perplexity,
    'multiclass_metrics_custom': multiclass_metrics_custom,
    'multilabel_metrics_custom': multilabel_metrics_custom,
}

select = selectMetric
