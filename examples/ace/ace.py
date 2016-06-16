import lasagne
from lasagne.nonlinearities import *
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, \
    FeaturePoolLayer, BatchNormLayer, NonlinearityLayer, flatten, batch_norm

from ace_layers import Dense3DLayer, SamplingLayer

import theano
from theano import tensor as T
th_rng = theano.tensor.shared_randomstreams.RandomStreams(9999)

import numpy as np
from collections import OrderedDict

from utils import binarize, half_linear


def build_encoder(X, net, num_hidden, num_classes, p_drop_input, p_drop_hidden,
                  error='Gaussian'):
    net['h0'] = batch_norm(DenseLayer(net.values()[-1], num_units=num_hidden[0],
                           nonlinearity=tanh, name='h0'))
    # shape: batch_size x n_h

    # h
    net['drop_h0'] = DropoutLayer(net.values()[-1], p=p_drop_input, name='drop_h0')

    net['h1'] = Dense3DLayer(net.values()[-1], num_units=num_hidden[1], name='h1',
                             dot_axes=[[1], [0]], nonlinearity=linear)
    # shape: batch_size x 2 x n_h2

    net['flatten_h1'] = flatten(net.values()[-1], name='flatten_h1')
    # shape: batch_size x 2*n_h2

    net['max_h1'] = FeaturePoolLayer(net.values()[-1], pool_size=2, axis=1,
                                     name='max_h1', pool_function=theano.tensor.max)
    # maxout shape: batch_size x n_h2

    net['drop_h1'] = DropoutLayer(net['max_h1'], p=p_drop_hidden, name='drop_h1')

    net['h2'] = Dense3DLayer(net.values()[-1], num_units=num_hidden[2], name='h2',
                             dot_axes=[[1], [0]], nonlinearity=linear)
    # shape: batch_size x 2 x n_h3

    net['batchn_h2'] = BatchNormLayer(net.values()[-1], axes=0, name='batchn_h2')
    net['flatten_h2'] = flatten(net.values()[-1], name='flatten_h2')
    # shape: batch_size x 2*n_h3
    
    net['max_h2'] = FeaturePoolLayer(net.values()[-1], pool_size=2, axis=1,
                                     name='max_h2', pool_function=theano.tensor.max)
    # maxout shape: batch_size x 1 x n_h3

    net['drop_h2'] = DropoutLayer(net.values()[-1], p=p_drop_hidden, name='drop_h2')

    net['h3'] = Dense3DLayer(net.values()[-1], num_units=num_hidden[3], name='h3',
                             dot_axes=[[1], [0]])
    # shape: batch_size x 2 x n_h4

    net['flatten_h3'] = flatten(net.values()[-1], name='flatten_h3')
    # shape: batch_size x 2*n_h4
    
    net['max_h3'] = FeaturePoolLayer(net.values()[-1], pool_size=2, axis=1,
                                     name='max_h3', pool_function=theano.tensor.max)
    # maxout shape: batch_size x 1 x n_h4

    net['drop_h3'] = DropoutLayer(net['max_h3'], p=p_drop_hidden, name='drop_h3')

    net['prob_y'] = DenseLayer(net['drop_h3'], num_units=num_classes, b=None,
                               nonlinearity=softmax, name='prob_y')
    # shape: batch_size x n_c

    # dual reconstruction error
    h = lasagne.layers.get_output(net['drop_h0'], X, deterministic=False)
    dual_X_hat = T.dot(h, T.dot(h.T, X)).mean(axis=0)
    if error == 'Gaussian':
        dual_recon_err = 0.5 * T.sqr(X - dual_X_hat)
    elif error == 'Laplacian':
        dual_X_hat =  T.nnet.sigmoid(dual_X_hat)
        dual_recon_err = T.nnet.binary_crossentropy(dual_X_hat, X)
    # shape: batch_size x n_x

    return net['prob_y'], net, dual_recon_err.sum(axis=1).mean()


def build_generative_model(X, net, num_latent, y=None, sampling_class='Gaussian'):
    # gen latent layer
    net['mu'] = Dense3DLayer(net['h0'], num_units=num_latent, name='mu',
                             nonlinearity=linear, dot_axes=[[1], [0]])
    net['log_sigma'] = Dense3DLayer(net['h0'], num_units=num_latent,
                                    nonlinearity=half_linear, # linear,
                                    dot_axes=[[1], [0]], name='log_sigma')
    # shape: batch_size x n_c x n_z

    #sampling
    net['latent_z'] = SamplingLayer(net['mu'], net['log_sigma'],
                                    sampling_class, name='latent_z')

    # gen error
    mu, log_sigma = lasagne.layers.get_output([net['mu'], net['log_sigma']], X,
                                              deterministic=False)
    # shape: batch_size x n_c x n_z

    if sampling_class == 'Gaussian':
        gen_err_stack = - 0.5 * (1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))
    elif sampling_class == 'Laplacian':
        gen_err_stack = (- log_sigma  + T.abs_(mu) / T.sqrt(0.5) + T.exp(log_sigma) * \
                         T.exp(- T.abs_(mu) / T.exp(log_sigma) / T.sqrt(0.5)) - 1)

    return net, gen_err_stack.sum(axis=2) # shape: batch_size x n_c


def build_decoder(X, net, num_hidden, p_drop_hidden, y=None, dist='Gaussian'):
    # decoder
    # z ~ (batch_size, n_c, n_z), W_h_dec ~ (n_c, n_z, n_h)
    # after dimshuffle: (n_c, (batch_size, n_z) x (n_z, n_h)) = (n_c, batch_size, n_h)
    # dimshfl -> (batch_size, n_c, n_h)
    net['h0_dec'] = Dense3DLayer(net['latent_z'], num_units=num_hidden[0],
                                 nonlinearity=tanh, dot_type='batched',
                                 dimshuffle=(1, 0, 2), name='h0_dec')
    # shape: batch_size, n_c, n_h

    net['drop_h0_dec'] = DropoutLayer(net.values()[-1], p=p_drop_hidden,
                                      name='drop_h0_dec')

    net['input_dec'] = Dense3DLayer(net.values()[-1], num_units=num_hidden[1],
                                    nonlinearity=linear, dot_type='batched',
                                    dimshuffle=(1, 0, 2), name='input_dec')
    # shape: batch_size x n_c x n_x

    X_stack = T.stack([X]*num_hidden[-1][0], axis=1)
    if dist == 'Gaussian':
        net['X_hat'] = NonlinearityLayer(net.values()[-1], name='X_hat',
                                         nonlinearity=rectify)
        X_hat = lasagne.layers.get_output(net['X_hat'], X, deterministic=False)
        recon_err_stack = 0.5 * T.log(2 * np.pi) +  0.5 * T.sqr(X_stack - X_hat)
    elif dist == 'Laplacian':
        net['X_hat'] = NonlinearityLayer(net.values()[-1], name='X_hat',
                                         nonlinearity=sigmoid)
        X_hat = lasagne.layers.get_output(net['X_hat'], X, deterministic=False)
        X_bin = binarize(X_stack)
        recon_err_stack = T.nnet.binary_crossentropy(X_hat, X_bin)

    return net, recon_err_stack.sum(axis=2), X_hat # shape: batch_size x n_c


def build_sup_gen_net(Z, net, num_hid, num_lat, num_class, dist='Gaussian'):
    #supervised generative output
    gen_net = OrderedDict()
    gen_net['gen_z'] = InputLayer((None,) + tuple(num_lat), name='gen_z')
    W, b = net['h0_dec'].get_params()
    gen_net['h0_dec_gen'] = Dense3DLayer(gen_net['gen_z'], num_units=num_hid[0],
                                         nonlinearity=tanh, name='h0_dec_gen',
                                         dot_type='batched', W=W, b=b,
                                         dimshuffle=(1, 0, 2))
    W, b = net['input_dec'].get_params()
    gen_net['h1_dec_gen'] = Dense3DLayer(gen_net['h0_dec_gen'], num_units=num_hid[1],
                                         nonlinearity=linear, name='h0_dec_gen',
                                         dot_type='batched', W=W, b=b,
                                         dimshuffle=(1, 0, 2))

    if dist == 'Gaussian':
        gen_net['X_hat_gen'] = NonlinearityLayer(gen_net.values()[-1],
                                                 nonlinearity=rectify,
                                                 name='X_hat_gen')
    elif dist == 'Laplacian':
        gen_net['X_hat_gen'] = NonlinearityLayer(gen_net.values()[-1],
                                                 nonlinearity=sigmoid,
                                                 name='X_hat_gen')

    X_hat_gen = lasagne.layers.get_output(gen_net['X_hat_gen'], Z,
                                          deterministic=False)
    return X_hat_gen


def build_net(batch_size, input_size, enc_hid, gen_hid, dec_hid, num_classes,
              p_drop_input=0., p_drop_hidden=0., distribution='Gaussian',
              supervised=False):
    X = T.fmatrix('X')
    Z = T.ftensor3('Z')

    net = OrderedDict()
    net['input'] = InputLayer((batch_size, input_size))
    net['drop_input'] = DropoutLayer(net['input'], p=p_drop_input)

    # Classifier branch
    output_layer, net, dual_recon_err = build_encoder(X, net, enc_hid, num_classes,
                                                      p_drop_input, p_drop_hidden,
                                                      distribution)
    # Auto-encoder branch
    net, gen_err_stack = build_generative_model(X, net, gen_hid, distribution)
    net, recon_err_stack, X_hat = build_decoder(X, net, dec_hid, p_drop_hidden,
                                                distribution)

    # supervised generative output
    X_hat_gen = build_sup_gen_net(Z, net, dec_hid, gen_hid, num_classes,
                                  distribution)

    output_train = lasagne.layers.get_output(output_layer, X, deterministic=False)
    output_eval = lasagne.layers.get_output(output_layer, X, deterministic=True,
                                            batch_norm_use_averages=False)

    # Error stacks of shape: batch_size x n_c
    # Unsupervised: upper bound for mixture of densities, using weights prob_y
    gen_err = (gen_err_stack * output_train).sum(axis=1).mean()
    # Unsupervised: mixture of densities, using  weights prob_y
    recon_err = (recon_err_stack * output_train).sum(axis=1).mean()

    if supervised: # supervised
        y = T.fmatrix('y')

        # supervised non-generative output
        # X_hat shape: batch_size x n_c x n_x
        sup_X_hat = (X_hat * y.dimshuffle((0, 1, 'x'))).sum(axis=1)
        sup_X_hat_gen = (X_hat_gen * y.dimshuffle((0, 1, 'x'))).sum(axis=1)
        # every X_hat has shape: batch_size x n_x

        task_dict = {
            'y': y,
            'reconstruct': sup_X_hat,
            'sample': sup_X_hat_gen
        }

        # supervised errors
        sup_gen_err = (gen_err_stack * y).sum(axis=1).mean()
        sup_recon_err = (recon_err_stack * y).sum(axis=1).mean()

        # classification error
        class_err =  T.nnet.categorical_crossentropy(output_train, y).mean()

         # classficiation
        costs = [sup_recon_err, sup_gen_err, class_err]
        # cost = sup_recon_err + sup_gen_err + class_err
    else: # unsupervised - density estimation
        task_dict = {
            'reconstruct': X_hat.sum(axis=1),
            'sample': X_hat_gen.sum(axis=1)
        }
        costs = [recon_err, gen_err, dual_recon_err]
        # cost = recon_err + gen_err + dual_recon_err

    network_dump = {'output_layer': output_layer,
                    'output_eval': output_eval,
                    'cost': T.sum(costs),
                    'costs': costs,
                    'net': net,
                    'x': X,
                    'z': Z,
                    }

    network_dump.update(task_dict)

    return network_dump