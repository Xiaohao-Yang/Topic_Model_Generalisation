import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_
import copy
from torchmetrics.functional import pairwise_euclidean_distance
import sys
sys.path.append('../Topic_Model_Generalisation-main')
from doc_dist import *


class Scholar(object):
    def __init__(self, config, alpha=1.0, learning_rate=0.001, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, adam_beta1=0.99, adam_beta2=0.999, device=None, seed=None,
                 classify_from_covars=True, topk=1):

        """
        Create the model
        :param config: a dictionary with the model configuration
        :param alpha: hyperparameter for the document representation prior
        :param learning_rate: learning rate for Adam
        :param init_embeddings: a matrix of embeddings to initialize the first layer of the bag-of-words encoder
        :param update_embeddings: if True, update word embeddings during training
        :param init_bg: a vector of empirical log backgound frequencies
        :param update_background: if True, update the background term during training
        :param adam_beta1: first hyperparameter for Adam
        :param adam_beta2: second hyperparameter for Adam
        :param device: (int) the number of the GPU to use
        """

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.reg_weight = config['reg_weight']
        self.update_embeddings = update_embeddings
        self.update_background = update_background
        self.use_reg = config['reg']

        # create priors on the hidden state
        self.n_topics = (config["n_topics"])

        if device is None:
            self.device = 'cpu'
        else:
            self.device = 'cuda:' + str(device)

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        self._model = torchTrueContrastiveScholar(config, self.alpha, update_embeddings, init_emb=init_embeddings, bg_init=init_bg, device=self.device, classify_from_covars=classify_from_covars,topk=topk).to(self.device)

        # set the criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate, betas=(adam_beta1, adam_beta2))

    def fit(self, X, Y, distance, data_augmented=None, PC=None, TC=None, eta_bn_prop=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l1_beta: np.array of prior variances on the topic weights
        :param l1_beta_c: np.array of prior variances on the weights for topic covariates
        :param l1_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # forward for aug data
        if self.use_reg:
            data_augmented = data_augmented.to(self.device)
            thetas_aug, _, _,  = self._model(data_augmented, Y, PC, TC, compute_loss=False, eta_bn_prop=eta_bn_prop, l1_beta=l1_beta,
                                                           l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci)

        # do a forward pass
        thetas, X_recon, Y_probs, losses = self._model(X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l1_beta=l1_beta,
                                                   l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci)

        loss, nl, kld, contrastive_loss = losses

        if self.use_reg :
            if distance == 'Cos':
                reg_loss = torch.nn.CosineSimilarity()(thetas, thetas_aug).mean()
            elif distance == 'Hel':
                reg_loss = avg_hellinger(thetas, thetas_aug)
            elif distance == 'L2':
                reg_loss = pairwise_euclidean_distance(thetas, thetas_aug).mean()
            else:
                beta = self._model.beta_layer.weight.T.to(torch.float64)
                beta = F.softmax(beta, dim=1)
                word_embedding = self._model.word_embedding
                reg_loss = doc_hot(thetas, thetas_aug, beta, word_embedding).mean()
        else:
            reg_loss = torch.tensor(0.)

        # update model
        loss += self.reg_weight * reg_loss
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), \
               kld.to('cpu').detach().numpy(), contrastive_loss, reg_loss

    def perplexity(self, test_1, test_1_label, test_2, beta, batch_size=200):
        n_test = test_1.shape[0]

        from scipy import sparse
        if sparse.isspmatrix(test_1):
            test_1 = test_1.toarray()
        test_1 = test_1.astype('float32')
        if sparse.isspmatrix(test_2):
            test_2 = test_2.toarray()
        test_2 = test_2.astype('float32')

        num_batches = int(math.ceil(n_test / batch_size))
        acc_loss = 0
        count = 0

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            test1_batch = test_1[start:end, :]
            test1_Label_batch = test_1_label[start:end, :]
            test2_batch = test_2[start:end, :]

            # first half to get theta
            theta = self.get_theta(test1_batch, test1_Label_batch, None, None)

            # get predition loss using second half
            sums_2 = test2_batch.sum(1)
            preds = np.log(np.matmul(theta, beta))
            recon_loss = -(preds * test2_batch).sum(1)
            loss = recon_loss / sums_2
            loss = np.nanmean(loss)
            acc_loss += loss
            count += 1

        cur_loss = acc_loss / count
        ppl_dc = round(math.exp(cur_loss), 1)

        return ppl_dc

    def get_theta(self, X, Y, PC=None, TC=None, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)

        # theta, _, _, _ = self._model(X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        theta = self._model(X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop, compute_theta=True)
        return theta.to('cpu').detach().numpy()

    def get_beta(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.weight.T
        self._model.beta_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class torchTrueContrastiveScholar(nn.Module):
    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu', classify_from_covars=False, topk=1):
        super(torchTrueContrastiveScholar, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.word_embedding = config['word_embedding']
        self.reg_weight = config['reg_weight']
        self.aug_method = config['aug_method']
        self.reg = config['reg']

        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.classifier_layers = config['classifier_layers']
        self.use_interactions = config['use_interactions']
        self.l1_beta_reg = config['l1_beta_reg']
        self.l1_beta_c_reg = config['l1_beta_c_reg']
        self.l1_beta_ci_reg = config['l1_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.device = device
        self.classify_from_covars = classify_from_covars

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.embeddings_x_layer = nn.Linear(self.vocab_size, self.words_emb_dim, bias=False)
        emb_size = self.words_emb_dim
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            else:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

        self.cos = torch.nn.CosineSimilarity()
        self.topk = topk

    def forward(self, X, Y, PC, TC, compute_loss=True, do_average=True, eta_bn_prop=1.0, var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, compute_theta=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """

        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        if compute_theta:
            z_do = z_do.to(torch.float64)
            theta = F.softmax(z_do, dim=1)
            return theta
        else:
            theta = F.softmax(z_do, dim=1)
        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        #eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        ##############################################################################################################################################################################################################################################
        # contrastive component - instead of tdidf, we directly use the weights of reconstructed BoW, which provided similar results but faster computation
        ##############################################################################################################################################################################################################################################

        # negative part 
        max_ids = torch.topk(X_recon, k=self.topk, dim=1).indices
        max_values = torch.topk(X_recon,k=self.topk,dim=1).values
        total_words = torch.sum(X,1).unsqueeze(-1)
        max_values = max_values*total_words
        tmp_x = copy.copy(X)
        negative_X = tmp_x.scatter_(1,max_ids, max_values)

        negative_en0_x = self.embeddings_x_layer(negative_X)
        negative_encoder_parts = [negative_en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            negative_encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            negative_encoder_parts.append(TC)
        if self.n_labels > 0:
            negative_encoder_parts.append(Y)

        if len(negative_encoder_parts) > 1:
            negative_en0 = torch.cat(negative_encoder_parts, dim=1).to(self.device)
        else:
            negative_en0 = negative_en0_x
            
        negative_encoder_output = F.softplus(negative_en0)
        negative_encoder_output_do = self.encoder_dropout_layer(negative_encoder_output)
        negative_posterior_mean = self.mean_layer(negative_encoder_output_do)
        negative_posterior_logvar = self.logvar_layer(negative_encoder_output_do)
        negative_posterior_mean_bn = self.mean_bn_layer(negative_posterior_mean)
        negative_posterior_logvar_bn = self.logvar_bn_layer(negative_posterior_logvar)
        negative_posterior_var = negative_posterior_logvar_bn.exp().to(self.device)
        negative_z = negative_posterior_mean_bn + negative_posterior_var.sqrt() * eps * var_scale

        # positive part
        min_ids = torch.topk(X_recon, k=self.topk,largest=False, dim=1).indices
        min_values = torch.topk(X_recon,k=self.topk,largest=False,dim=1).values
        total_words = torch.sum(X,1).unsqueeze(-1)
        min_values = min_values*total_words
        tmp_x = copy.copy(X)
        positive_X = tmp_x.scatter_(1,min_ids, min_values)

        positive_en0_x = self.embeddings_x_layer(positive_X)
        positive_encoder_parts = [positive_en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            positive_encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            positive_encoder_parts.append(TC)
        if self.n_labels > 0:
            positive_encoder_parts.append(Y)

        if len(negative_encoder_parts) > 1:
            positive_en0 = torch.cat(positive_encoder_parts, dim=1).to(self.device)
        else:
            positive_en0 = positive_en0_x
            
        positive_encoder_output = F.softplus(positive_en0)
        positive_encoder_output_do = self.encoder_dropout_layer(positive_encoder_output)
        positive_posterior_mean = self.mean_layer(positive_encoder_output_do)
        positive_posterior_logvar = self.logvar_layer(positive_encoder_output_do)
        positive_posterior_mean_bn = self.mean_bn_layer(positive_posterior_mean)
        positive_posterior_logvar_bn = self.logvar_bn_layer(positive_posterior_logvar)
        positive_posterior_var = positive_posterior_logvar_bn.exp().to(self.device)
        positive_z = positive_posterior_mean_bn + positive_posterior_var.sqrt() * eps * var_scale

        normalized_z = F.normalize(z)
        normalized_positive_z = F.normalize(positive_z)
        normalized_negative_z = F.normalize(negative_z)

        positive_component = self.cos(normalized_z, normalized_positive_z).mean()
        negative_component = self.cos(normalized_z, normalized_negative_z).mean()
        contrastive_loss = -torch.log(
            torch.exp(positive_component) / (torch.exp(positive_component) + 0.5 * torch.exp(negative_component)))

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar,
                                                       posterior_mean_bn, posterior_logvar_bn, do_average,
                                                       l1_beta, l1_beta_c, l1_beta_ci, contrastive_loss)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None, contrastive_loss=None):

        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)
        if contrastive_loss: loss += contrastive_loss

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean(), contrastive_loss
        else:
            return loss, NL, KLD, contrastive_loss
