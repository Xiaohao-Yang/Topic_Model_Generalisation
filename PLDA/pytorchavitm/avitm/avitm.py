"""Class to train AVITM models."""
import multiprocessing as mp
import datetime
from torch.nn.functional import softmax
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchavitm.avitm.decoder_network import DecoderNetwork
import sys
sys.path.append('../TM_Gen_Github')
from doc_aug import *
from doc_dist import *
from eval import evaluation
from torchmetrics.functional import pairwise_euclidean_distance
from tqdm import tqdm


class AVITM(object):

    """Class to train AVITM model."""

    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False):
        """
        Initialize AVITM model.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            dropout : float, dropout to use (default 0.2)
            learn_priors : bool, make priors a learnable parameter (default True)
            batch_size : int, size of batch to use for training (default 64)
            lr : float, learning rate to use for training (default 2e-3)
            momentum : float, momentum to use for training (default 0.99)
            solver : string, optimizer 'adam' or 'sgd' (default 'adam')
            num_epochs : int, number of epochs to train for, (default 100)
            reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        """
        assert isinstance(input_size, int) and input_size > 0,\
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and input_size > 0,\
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'],\
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0,\
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1,\
            "momentum must be 0 < float <= 1."
        assert solver in ['adam', 'sgd'], "solver must be 'adam' or 'sgd'."
        assert isinstance(reduce_on_plateau, bool),\
            "reduce_on_plateau must be type bool."

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau

        # init inference avitm network
        self.model = DecoderNetwork(
            input_size, n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors)

        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training atributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        if self.USE_CUDA:
            self.model = self.model.cuda()


    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        loss = KL + RL

        return loss.sum()

    def _train_epoch(self, epoch, loader, args, similarity_dict, dissimilarity_dict, word_embedding):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader, desc='Epoch: %s' % (epoch+1)):
            # batch_size x vocab_size
            X = batch_samples['X']

            if args.regularisation:
                data_augmented = copy.deepcopy(X)

                if args.DA == 'DA1':
                    # random similar
                    data_augmented = random_similar(data_augmented, similarity_dict, dissimilarity_dict, args.aug_rate)
                elif args.DA == 'DA2':
                    # high similar
                    data_augmented = high_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA3':
                    #  low similar
                    data_augmented = low_similar(data_augmented, args.aug_rate, similarity_dict)
                elif args.DA == 'DA4':
                    #  high and low -- similar
                    data_augmented = high_low_similar(data_augmented, args.aug_rate, similarity_dict)

                if self.USE_CUDA:
                    X = X.cuda()
                    data_augmented = data_augmented.cuda()

                prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance, word_dists, \
                theta_ori = self.model(X)
                _, _, _, _, _, _, theta_aug = self.model(data_augmented)

                beta = self.model.beta
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
                    reg_loss = doc_hot(theta_ori, theta_aug, beta, word_embedding)

            else:
                reg_loss = torch.tensor(0.0)
                if self.USE_CUDA:
                    X = X.cuda()
                prior_mean, prior_variance, \
                posterior_mean, posterior_variance, posterior_log_variance, \
                word_dists, _ = self.model(X)

            # backward pass
            self.model.zero_grad()
            loss = self._loss(
                X, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)

            loss = loss + args.reg_weight * reg_loss
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss


    def fit(self, data_dict, args, similarity_dict, dissimilarity_dict, word_embedding, save_dir=None):
        """
        Train the AVITM model.

        Args
            train_dataset : PyTorch Dataset classs for training data.
            val_dataset : PyTorch Dataset classs for validation data.
            save_dir : directory to save checkpoint models to.
        """
        # Print settings to output file
        print("Settings: \n\
               N Components: {}\n\
               Topic Prior Mean: {}\n\
               Topic Prior Variance: {}\n\
               Model Type: {}\n\
               Hidden Sizes: {}\n\
               Activation: {}\n\
               Dropout: {}\n\
               Learn Priors: {}\n\
               Learning Rate: {}\n\
               Momentum: {}\n\
               Reduce On Plateau: {}\n\
               Save Dir: {}".format(
                   self.n_components, 0.0,
                   1. - (1./self.n_components), self.model_type,
                   self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                   self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = data_dict['train_data']

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=mp.cpu_count())

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(epoch, train_loader, args, similarity_dict, dissimilarity_dict, word_embedding)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss, e - s))

            if (epoch + 1) % args.eval_step == 0:
                evaluation(self.model, epoch + 1, data_dict, args)