import argparse
from scholar import Scholar
from utils import *
import time
import os
import sys
sys.path.append('../TM_Gen_Github')
from read_data import *
from doc_aug import *
from eval import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CLNTM')
parser.add_argument("--name", type=str, default="CLNTM")
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--eval_step', type=int, default=5)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--Dist', default='HOT', type=str)
parser.add_argument('--regularisation', action='store_true')
parser.add_argument('--reg_weight', type=int, default=300)
parser.add_argument('--aug_rate', type=float, default=0.5)
parser.add_argument('--DA', type=str, default='DA2')
parser.add_argument('--similar_topn', type=int, default=20)


parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--train_prefix', type=str, default='train')
parser.add_argument('--test_prefix', type=str, default='test')
parser.add_argument('--labels', type=str, default=False)
parser.add_argument('--prior_covars', type=str, default=None)
parser.add_argument('--topic_covars', type=str, default=None)
parser.add_argument('--interactions', action="store_true", default=False)
parser.add_argument('--covars_predict', action="store_true", default=False)
parser.add_argument('--min_prior_covar_count', type=int, default=None)
parser.add_argument('--min_topic_covar_count', type=int, default=None)
parser.add_argument('--r', action="store_true", default=False)
parser.add_argument('--l1_topics', type=float, default=0.0)
parser.add_argument('--l1_topic_covars', type=float, default=0.0)
parser.add_argument('--l1_interactions', type=float, default=0.0)
parser.add_argument('--l2_prior_covars', type=float, default=0.0)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--w2v', dest='word2vec_file', type=str, default=None)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--no_bg', action="store_true", default=False)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dist', type=int, default=0)
parser.add_argument('--topk', type=int, default=15)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train(model, data_dict, similarity_dict, dissimilarity_dict, rng, bn_anneal=True):
    X = data_dict['train_data']
    Y = data_dict['train_label']
    n_train, vocab_size = X.shape
    mb_gen = create_minibatch(X, Y, batch_size=args.batch_size, rng=rng)
    total_batch = int(n_train / args.batch_size)
    batches = 0
    eta_bn_prop = 1.0  # interpolation between batch norm and no batch norm in final layer of recon

    # Training cycle
    model.train()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        avg_cost = 0.
        avg_nl = 0.
        avg_kld = 0.
        avg_cl = 0.
        avg_reg = 0.

        # Loop over all batches
        for i in tqdm(range(total_batch), desc='Epoch: %s' % (epoch+1)):
            # get a minibatch
            batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)

            if args.regularisation:
                data_augmented = copy.deepcopy(batch_xs)

                if args.DA == 'DA1':
                    # random similar
                    data_augmented = random_similar(data_augmented, similarity_dict, dissimilarity_dict, args.aug_rate)
                    data_augmented = torch.from_numpy(data_augmented)
                elif args.DA == 'DA2':
                    # high similar
                    data_augmented = high_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA3':
                    #  low similar
                    data_augmented = low_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA4':
                    #  high and low -- similar
                    data_augmented = high_low_similar(data_augmented, args.aug_rate, similarity_dict)

            else:
                data_augmented = None

            # do one minibatch update
            cost, recon_y, thetas, nl, kld, cl, reg = model.fit(batch_xs, batch_ys, args.Dist, data_augmented, batch_pcs, batch_tcs)

            # Compute average loss
            avg_cost += float(cost) / n_train * args.batch_size
            avg_nl += float(nl) / n_train * args.batch_size
            avg_kld += float(kld) / n_train * args.batch_size
            avg_cl += float(cl) / args.batch_size
            avg_reg += float(reg) / args.batch_size

            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                sys.exit()

        meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time() - epoch_start_time)
        print(meta + "| train loss {:5.2f} (nll {:4.2f} kld {:5.2f} contrastive loss {:5.2f} reg loss {:5.2f})"
              .format(avg_cost, avg_nl, avg_kld, avg_cl, avg_reg))

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * args.epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

        if (epoch + 1) % args.eval_step == 0:
            evaluation(model, epoch + 1, data_dict, args)

        model.train()

    # finish training
    model.eval()
    return model


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

    # prepare them in a dictionary
    data_dict = {'train_data': data_tr,
                 'test_data': data_te,
                 'test1_data': test1,
                 'test2_data': test2,
                 'train_label': train_label,
                 'test_label': test_label,
                 'test_1_label': test_label,

                 'train_data_target': data_tr_target,
                 'train_label_target': train_label_target,
                 'test_data_target': data_te_target,
                 'test_label_target': test_label_target,

                 'vocab': voc,
                 'word_embedding': word_embedding}

    rng = np.random.RandomState(args.seed)
    vocab_size = len(voc)
    init_bg = get_init_bg(data_tr)
    if args.no_bg:
        init_bg = np.zeros_like(init_bg)

    network_architecture = make_network(args, vocab_size, word_embedding)
    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # create the model
    model = Scholar(network_architecture, learning_rate=args.lr, seed=args.seed, alpha=args.alpha,
                    init_bg=init_bg, adam_beta1=args.momentum, device=args.device,
                     classify_from_covars=args.covars_predict, topk=args.topk)

    # train the model
    print("Optimizing full model")
    train(model, data_dict, similarity_dict, dissimilarity_dict, rng=rng)


if __name__ == '__main__':
    if not 'combined' in args.dataset:
        source_data_file = os.path.join('datasets', args.dataset, 'data.mat')
        # target is a noisy version of the original corpus
        target_data_file = os.path.join('datasets', args.dataset, 'data_aug.mat')
        main(source_data_file, target_data_file)
    else:
        data_file = os.path.join('datasets', args.dataset, 'data.mat')
        main(data_file, None)