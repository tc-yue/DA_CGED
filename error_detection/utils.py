# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 17:44
# @Author  : Tianchiyue
# @File    : utils.py
# @Software: PyCharm
import logging
import numpy as np
import os
import random
import torch
import pickle
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score
from collections import Counter
from copy import deepcopy



def cal_score(true_list, pred):
    pred = np.array(pred)
    for i in range(50):
        pred_list = deepcopy(pred)
        threshold = i * 0.02
        pred_list[pred_list >= threshold] = 1
        pred_list[pred_list < threshold] = 0
        r = recall_score(true_list, pred_list)
        p = precision_score(true_list, pred_list)
        f = f1_score(true_list, pred_list)
        # print(i,p,r,f)
        if p > 0.985 or Counter(pred_list)[1] < 1000:
            break
    return [str(round(i, 4)) for i in [p, r, f]]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--seed_num', default=147, type=int)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--valid_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--index_path', type=str, default=None)

    parser.add_argument('--bert_config_path', type=str, default='bert_chinese/bert_config.json')
    parser.add_argument('--bert_model_path', type=str, default='bert_chinese/pytorch_model.bin')

    parser.add_argument('--recall_model', default='cnn', type=str)
    parser.add_argument('-e', '--epochs', default=10, type=int)

    parser.add_argument('-d', '--dropout', default=0.2, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--train_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--valid_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--valid_step', default=1000, type=int)

    parser.add_argument('--vocab_size', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_labels', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--embedding_dim', type=int, default=300, help='DO NOT MANUALLY SET')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--trainable_embedding', action='store_true')
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument('--weighted', action='store_true')

    return parser.parse_args(args)


def read_pickle(filepath):
    with open(filepath, 'rb')as f:
        return pickle.load(f)


def write_pickle(filepath, data):
    with open(filepath, 'wb')as f:
        pickle.dump(data, f)


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(num)
    # np.random.seed(num)
    # random.seed(num)
    # torch.manual_seed(num)
    # torch.cuda.manual_seed(num)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
