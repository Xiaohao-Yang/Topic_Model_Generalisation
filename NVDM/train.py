import os.path
import argparse
import datetime
from nvdm_torch import *
from dataset import *

import sys
sys.path.append('../Topic_Model_Generalisation-main')
from doc_aug import *
from doc_dist import *
from read_data import *
from eval import evaluation
from torchmetrics.functional import pairwise_euclidean_distance
from tqdm import tqdm


parser = argparse.ArgumentParser(description='NVDM')
parser.add_argument("--name", type=str, default="NVDM")
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--eval_step', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--regularisation', action='store_true')
parser.add_argument('--aug_rate', default=0.5, type=float)
parser.add_argument('--reg_weight', default=300, type=float)
parser.add_argument('--DA', default='DA2')
parser.add_argument('--Dist', default='HOT', type=str)
parser.add_argument('--similar_topn', type=int, default=20)

parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--n_hidden', default=100, type=int)
parser.add_argument('--n_sample', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data_dict, model, epoch_num, similarity_dict, dissimilarity_dict):
    # data loader
    dataloader = DataLoader(data_dict['train_data'], batch_size=args.batch_size, shuffle=True)

    # record value
    loss_sum = 0.0
    ppx_sum = 0.0
    kld_sum = 0.0
    reg_sum = 0.0
    word_count = 0
    doc_count = 0

    # optimiser
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))

    # training loop
    for epoch in range(epoch_num):
        s = datetime.datetime.now()
        for data_batch, count_batch in tqdm(dataloader, desc='Epoch: %s' % (epoch+1)):
            data_batch = data_batch.float().cuda()
            count_batch = count_batch.cuda()

            if args.regularisation:
                # create document augmentations
                data_augmented = copy.deepcopy(data_batch)
                if args.DA == 'DA1':
                    data_augmented = data_augmented.cpu().detach().numpy()
                    # random similar
                    data_augmented = random_similar(data_augmented, similarity_dict, dissimilarity_dict, args.aug_rate)
                    data_augmented = torch.tensor(data_augmented)
                elif args.DA == 'DA2':
                    # high similar
                    data_augmented = high_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA3':
                    #  low similar
                    data_augmented = low_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA4':
                    #  both to similar
                    data_augmented = high_low_similar(data_augmented, args.aug_rate, similarity_dict)
                data_augmented = data_augmented.float().cuda()

                kld, recons_loss, _, theta_ori = model(data_batch)  # original theta (topical representation)
                _, _, _, theta_aug = model(data_augmented)          # aug theta (topical representation)
                beta = model.dec_vec.weight.T                       # topics

                # normalise as distributions
                beta = softmax(beta.to(torch.float64), dim=1)
                theta_ori = softmax(theta_ori.to(torch.float64), dim=1)
                theta_aug = softmax(theta_aug.to(torch.float64), dim=1)

                # distance metrics
                if args.Dist == 'Cos':
                    reg_loss = torch.nn.CosineSimilarity()(theta_ori, theta_aug).mean()
                elif args.Dist == 'Hel':
                    reg_loss = avg_hellinger(theta_ori, theta_aug)
                elif args.Dist == 'L2':
                    reg_loss = pairwise_euclidean_distance(theta_ori, theta_aug).mean()
                else:
                    # HOT
                    reg_loss = doc_hot(theta_ori, theta_aug, beta, data_dict['word_embedding'])

            else:
                kld, recons_loss, _, _ = model(data_batch)
                reg_loss = torch.tensor(0.0)

            # compute loss
            loss = kld + recons_loss + args.reg_weight * reg_loss

            loss_sum += torch.sum(loss).item()
            kld_sum += torch.mean(kld).item()
            reg_sum += reg_loss.item()
            word_count += torch.sum(count_batch).item()
            count_batch = torch.add(count_batch, 1e-12)
            ppx_sum += torch.sum(torch.div(loss, count_batch)).item()
            doc_count += len(data_batch)

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

        e = datetime.datetime.now()
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum / len(dataloader)
        print_reg = reg_sum / len(dataloader)
        print('| Time : {} |'.format(e - s),
              '| Epoch train: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld),
              '| Reg Loss: {:.5}'.format(print_reg))

        # evaluation phase
        if (epoch + 1) % args.eval_step == 0:
            evaluation(model, epoch + 1, data_dict, args)


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
    train_dataset = FeatDataset(data_tr)
    test_dataset = FeatDataset(data_te)
    test1_data = FeatDataset(test1)
    test2_data = FeatDataset(test2)

    # create target dataset
    if args.dataset == 'combined_20News_RestAll':
        train_data_target = []
        test_data_target = []
        # 4 different target corpus
        for i in range(4):
            train_data_target.append(FeatDataset(data_tr_target[i]))
            test_data_target.append(FeatDataset(data_te_target[i]))
    else:
        train_data_target = FeatDataset(data_tr_target)
        test_data_target = FeatDataset(data_te_target)

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

                 'idx2token': idx2token,
                 'word_embedding': word_embedding}

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
    model = NVDM(len(voc), args.n_hidden, args.n_topic, args.n_sample).to(device)

    # start training
    train(data_dict, model, args.epochs, similarity_dict, dissimilarity_dict)


if __name__ == '__main__':
    if not 'combined' in args.dataset:
        source_data_file = os.path.join('datasets', args.dataset, 'data.mat')
        # target is a noisy version of the original corpus
        target_data_file = os.path.join('datasets', args.dataset, 'data_aug.mat')
        main(source_data_file, target_data_file)
    else:
        data_file = os.path.join('datasets', args.dataset, 'data.mat')
        main(data_file, None)
