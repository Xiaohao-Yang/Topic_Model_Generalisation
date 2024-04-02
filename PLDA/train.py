import os
import argparse
from pytorchavitm import AVITM
from pytorchavitm.datasets import BOWDataset
import sys
sys.path.append('../TM_Gen_Github')
from read_data import *
from doc_aug import *
import torch
import numpy as np
import json


parser = argparse.ArgumentParser(description='ProdLDA')
parser.add_argument("--name", type=str, default="PLDA")
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument("--eval_step", default=5, type=int)
parser.add_argument('--regularisation', action='store_true')
parser.add_argument("--aug_rate", default=0.25, type=float)
parser.add_argument("--reg_weight", default=100, type=float)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--DA", default='DA2')
parser.add_argument('--Dist', default='HOT', type=str)
parser.add_argument("--similar_topn", type=int, default=20)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def main(source_file, target_file=None):
    # load data
    if target_file is not None:
        data_tr, train_label, data_te, test_label, word_embedding, voc, test1, test2 = load_train_data(source_file)
        data_tr_target, train_label_target, data_te_target, test_label_target = load_noisy_target(target_file)
    else:
        if args.dataset == 'combined_20News_RestAll':
            data_tr, train_label, data_te, test_label, word_embedding, voc, test1, test2, \
                data_tr_target, train_label_target, data_te_target, test_label_target = load_union_data(source_file)
        else:
            data_tr, train_label, data_te, test_label, word_embedding, voc, test1, test2, \
                data_tr_target, train_label_target, data_te_target, test_label_target = load_combined_data(source_file)

    # create source dataset
    idx2token = {index:word for index, word in enumerate(voc)}
    train_dataset = BOWDataset(data_tr, idx2token)
    test_dataset = BOWDataset(data_te, idx2token)
    test1_data = BOWDataset(test1, idx2token)
    test2_data = BOWDataset(test2, idx2token)

    # create target dataset
    if args.dataset == 'combined_20News_RestAll':
        train_data_target = []
        test_data_target = []
        # 4 different target corpus
        for i in range(4):
            train_data_target.append(BOWDataset(data_tr_target[i], idx2token))
            test_data_target.append(BOWDataset(data_te_target[i], idx2token))
    else:
        train_data_target = BOWDataset(data_tr_target, idx2token)
        test_data_target = BOWDataset(data_te_target, idx2token)

    # prepare them in a dictionary
    data_dict = {'train_data': train_dataset,
                 'test_data': test_dataset,
                 'test1_data': test1_data,
                 'test2_data': test2_data,
                 'train_label': train_label,
                 'test_label': test_label,

                 'train_data_target': train_data_target,
                 'train_label_target': train_label_target,
                 'test_data_target': test_data_target,
                 'test_label_target': test_label_target,

                 'idx2token': idx2token}

    # prepare similarity between voc
    dict_save_path = 'datasets/%s' % args.dataset
    if not os.path.exists('%s/similar_%s.json' % (dict_save_path, args.similar_topn)):
        print('Creating similarity dictionary ...')
        create_similarity_dict(word_embedding, dict_save_path, args.similar_topn)
        print('Done!' + '\n')
    with open('datasets/%s/similar_%s.json' % (args.dataset, args.similar_topn)) as file:
        similarity_dict = json.load(file)
        file.close()
    with open('datasets/%s/dissimilar_%s.json' % (args.dataset, args.similar_topn)) as file:
        dissimilarity_dict = json.load(file)
        file.close()

    # create model
    avitm = AVITM(input_size=len(voc), n_components=args.n_topic, model_type='prodLDA',
                  hidden_sizes=(200, ), activation='softplus', dropout=0.2,
                  learn_priors=True, batch_size=200, lr=args.lr, momentum=0.9,
                  solver='adam', num_epochs=args.epochs, reduce_on_plateau=False)
    # start training
    avitm.fit(data_dict, args, similarity_dict, dissimilarity_dict, word_embedding)


if __name__ == '__main__':
    if not 'combined' in args.dataset:
        source_data_file = os.path.join('datasets', args.dataset, 'data.mat')
        # target is a noisy version of the original corpus
        target_data_file = os.path.join('datasets', args.dataset, 'data_aug.mat')
        main(source_data_file, target_data_file)
    else:
        data_file = os.path.join('datasets', args.dataset, 'data.mat')
        main(data_file, None)