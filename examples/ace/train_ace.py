from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, OneHotEncoding

import lasagne
from utils import z_vals, visualize

from ace import build_net

import numpy as np
from collections import OrderedDict

import theano
import theano.misc.pkl_utils

import argparse
import cPickle


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
arg_parser.add_argument('-bs', '--batch_size', type=int, default=50)
arg_parser.add_argument('-ep', '--max_epochs', type=int, default=15)
arg_parser.add_argument('-dlr', '--decrease_lr', type=float, default=1.)
arg_parser.add_argument('-dist', '--distribution', type=str, default='Gaussian')
arg_parser.add_argument('-hhid', '--num_hidden', type=str, default='700,700,'
                                                                   '700,700')
arg_parser.add_argument('-hlat', '--num_latent', type=int, default=400)
arg_parser.add_argument('--supervised', action='store_true', default=True)
arg_parser.add_argument('--debug', action='store_true', default=False)
arg_parser.add_argument('--reconstruct', action='store_true', default=False)
arg_parser.add_argument('--generate', action='store_true', default=False)
args = arg_parser.parse_args()

NUM_EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

mnist = MNIST(which_sets=('train',), # sources='features',
              subset=slice(0, 50000), load_in_memory=True)
mnist_val = MNIST(which_sets=('train',), # sources='features',
                  subset=slice(50000, 60000), load_in_memory=True)
mnist_test = MNIST(which_sets=('test',), # sources='features',
                   load_in_memory=True)

data_stream = DataStream(mnist,
                         iteration_scheme=ShuffledScheme(mnist.num_examples,
                                                         batch_size=BATCH_SIZE))
data_stream_val = DataStream(mnist_val,
                             iteration_scheme=ShuffledScheme(
                                 mnist_val.num_examples, batch_size=BATCH_SIZE))
data_stream_test = DataStream(mnist_test,
                              iteration_scheme=ShuffledScheme(
                                  mnist_test.num_examples, batch_size=BATCH_SIZE))

data_stream = Flatten(data_stream, which_sources=('features',))
data_stream_val = Flatten(data_stream_val, which_sources=('features',))
data_stream_test = Flatten(data_stream_test, which_sources=('features',))

num_classes = 10

data_stream = OneHotEncoding(data_stream=data_stream,
                             which_sources=('targets',),
                             num_classes=num_classes)

data_stream_val = OneHotEncoding(data_stream=data_stream_val,
                                 which_sources=('targets',),
                                 num_classes=num_classes)

data_stream_test = OneHotEncoding(data_stream=data_stream_test,
                                  which_sources=('targets',),
                                  num_classes=num_classes)

# build network
n_h, n_h2, n_h3, n_h4 = [int(x) for x in args.num_hidden.split(',')] # [500, 500, 300, 300]
encoder_hid = [n_h, [2, n_h2], [2, n_h3], [2, n_h4]]
num_classes = n_c = 10
n_z = args.num_latent # 350
num_latent = [n_c, n_z]
dec_hid = [[n_c, n_h], [n_c, 784]]
z_shape = (BATCH_SIZE, ) + tuple(num_latent)

network_dump = build_net(None, 784, encoder_hid, num_latent, dec_hid, 10, 0.,
                         0., supervised=True, distribution=args.distribution)
# set up input/output variables
X = network_dump['x']
if args.supervised:
    y = network_dump['supervised'].get('y')
net = network_dump['net']

# training output
# output_train = lasagne.layers.get_output(train_output_l, X, deterministic=False)
cost = network_dump['cost']
costs = network_dump['costs']

# evaluation output. Also includes output of transform for plotting
# output_eval = lasagne.layers.get_output(eval_output_l, X, deterministic=True)
output_eval = network_dump['output_eval']

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

enc_params = lasagne.layers.get_all_params(net['prob_y'], trainable=True)
net_params = OrderedDict.fromkeys(enc_params)
gen_params = OrderedDict.fromkeys(lasagne.layers.get_all_params(net['latent_z'],
                                                                trainable=True))
net_params.update(gen_params)
dec_params = OrderedDict.fromkeys(lasagne.layers.get_all_params(net['X_hat'],
                                                                trainable=True))
net_params.update(dec_params)

updates = lasagne.updates.adam(cost, net_params.keys(), learning_rate=sh_lr)

# get training and evaluation functions
if args.debug:
    mode = theano.compile.mode.Mode(linker='py', optimizer='fast_compile')
else:
    mode = theano.compile.get_default_mode()

# train = theano.function([X, y], [cost] + costs, updates=updates, mode=mode)
if args.supervised:
    train = theano.function([X, y], costs, updates=updates, mode=mode)
else:
    train = theano.function([X], costs, updates=updates, mode=mode)

eval = theano.function([X], [output_eval], mode=mode)

if args.reconstruct:
    sup_X_hat = network_dump['supervised'].get('sup_X_hat')
    reconstruct = theano.function(inputs=[X, y], outputs=sup_X_hat, mode=mode)

if args.generate:
    Z = network_dump['supervised'].get('z')
    sup_X_hat_gen = network_dump['supervised'].get('sup_X_hat_gen')
    generate = theano.function(inputs=[Z, y], outputs=sup_X_hat_gen, mode=mode)


def save_dump(filename,param_values):
    f = file(filename, 'wb')
    cPickle.dump(param_values,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def train_epoch(stream):
    costs = []
    for batch in stream.get_epoch_iterator():
        train_out = train(*batch)
        # cur_cost = train_out[0]
        # single_costs = train_out[1:]
        costs.append(train_out)

    if args.reconstruct:
        recs = reconstruct(*batch)
        visualize(n, recs, [8,8], 'recons_{}_'.format(args.distribution))
    if args.generate:
        samples = generate(z_vals(args.distribution, z_shape), batch[1])
        visualize(n, samples, [8,8], 'samples_{}_'.format(args.distribution))


    rval = np.mean(costs, axis=0)
    print 'TRAIN: \trec err: {}, \tgen err: {}, \tclass err: {}'.format(*rval)
    return rval


def eval_epoch(stream, acc_only=True):
    preds = []
    targets = []
    for batch in stream.get_epoch_iterator():
        preds.extend(eval(batch[0]))
        targets.extend(batch[1])

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    acc = np.mean(preds.argmax(1) == targets.argmax(1)) # accuracy
    if not acc_only:
        nloglik = (np.log(preds) * targets).sum(1).mean()
        # confm = conf_mat(preds, targets)[0].astype(int)
        # CONF_MATS['iter_{}'.format(n)] = confm
        # save_dump('conf_mats_{}.pkl'.format(experiment_name), CONF_MATS)
        # print confm
        # return acc, nloglik, confm
    else:
        return acc

train_costs, train_accs, valid_accs = [], [], []
print 'Start training...'
try:
    for n in range(NUM_EPOCHS):
        train_costs.append(train_epoch(data_stream).sum())
        train_accs.append(eval_epoch(data_stream))
        valid_accs.append(eval_epoch(data_stream_val))

        if (n+1) % 10 == 0:
            new_lr = sh_lr.get_value() * args.decrease_lr
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
        save_dump('accs_{}_ace_mnist.pkl'.format(n),
                  zip(train_accs, valid_accs))
        # theano.misc.pkl_utils.dump(network_dump,
        #                            'iter_{}_ladder_nets_mnist.zip'.format(n))
        print "Epoch {}: Train cost {}, train acc {}, val acc {}".format(
                n, train_costs[-1], train_accs[-1], valid_accs[-1])
        # print 'TIMES: \ttrain {:10.2f}s, \tval {:10.2f}s'.format(t1-t0,
        #                                                          t2-t1)
    # TODO: needs an early stopping
except KeyboardInterrupt:
    pass

# save_dump('final_iter_{}_{}'.format(n, experiment_name),
#           lasagne.layers.get_all_param_values(output_layer))

theano.misc.pkl_utils.dump(network_dump,
                           'final_iter_{}_ace_mnist.pkl'.format(n))