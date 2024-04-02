import argparse
import os
from parameters import parameter_dict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='combined_20News_RestAll')
parser.add_argument('--model', type=str, default='NVDM', choices=['NVDM', 'PLDA', 'SCHOLAR', 'CLNTM'])
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--eval_step', type=int, default=5)
parser.add_argument('--use_Greg', action='store_true', help='Use Greg or not')
parser.add_argument('--reg_weight', type=int, default=300, help='Regularisation Weight')
parser.add_argument('--aug_rate', type=float, default=0.5, help='Augmentation Rate')
parser.add_argument('--DA_method', type=str, default='DA2', help='Document Augmentation Approaches',
                    choices=['DA1', 'DA2', 'DA3', 'DA4'])
parser.add_argument('--Dist', type=str, default='HOT', choices=['HOT', 'L2', 'Cos', 'Hel'],
                    help='Distance metric for regularisation')
args = parser.parse_args()


if __name__ == '__main__':
    setting = str(args.dataset) + '_K' + str(args.n_topic)   # dataset and K
    paras = parameter_dict[args.model][setting]
    lr = paras[0]                                            # learning rate
    epochs = paras[1]                                        # training epochs

    argument = ('python %s/train.py --dataset=%s --n_topic=%s --seed=%s --epochs=%s --eval_step=%s '
                '--lr=%s --reg_weight=%s --aug_rate=%s --DA=%s --Dist=%s') % (args.model, args.dataset, args.n_topic,
                                                                    args.random_seed, epochs, args.eval_step, lr,
                                                                    args.reg_weight, args.aug_rate, args.DA_method, args.Dist)
    argument += ' --regularisation' if args.use_Greg else ''
    os.system(argument)


