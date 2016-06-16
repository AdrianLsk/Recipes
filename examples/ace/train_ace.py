import lasagne

import numpy as np

import theano
import theano.misc.pkl_utils

from collections import OrderedDict

from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten, OneHotEncoding

from ace import build_net
from utils import z_vals, visualize, save_dump

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
arg_parser.add_argument('-bs', '--batch_size', type=int, default=50)
arg_parser.add_argument('-ep', '--max_epochs', type=int, default=15)
arg_parser.add_argument('-dlr', '--decrease_lr', type=float, default=1.)
arg_parser.add_argument('-dist', '--distribution', type=str, default='Gaussian')
arg_parser.add_argument('-nhid', '--num_hidden', type=str, default='700,700,'
                                                                   '700,700')
arg_parser.add_argument('-nlat', '--num_latent', type=int, default=400)
arg_parser.add_argument('--supervised', action='store_true', default=False)
arg_parser.add_argument('--debug', action='store_true', default=False)
arg_parser.add_argument('--reconstruct', action='store_true', default=False)
arg_parser.add_argument('--generate', action='store_true', default=False)
args = arg_parser.parse_args()

SUPERVISED = 'supervised' if args.supervised else 'unsupervised'

NUM_EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

# set-up fuel datasets and datastreams
mnist = MNIST(which_sets=('train',), subset=slice(0, 50000),
              load_in_memory=True)
mnist_val = MNIST(which_sets=('train',), subset=slice(50000, 60000),
                  load_in_memory=True)
mnist_test = MNIST(which_sets=('test',), load_in_memory=True)

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

# set up network parameters
n_h, n_h2, n_h3, n_h4 = [int(x) for x in args.num_hidden.split(',')]
encoder_hid = [n_h, [2, n_h2], [2, n_h3], [2, n_h4]]
num_classes = n_c = 10
n_z = args.num_latent
num_latent = [n_c, n_z]
dec_hid = [[n_c, n_h], [n_c, 784]]
z_shape = (BATCH_SIZE, ) + tuple(num_latent)

# build network
network_dump = build_net(None, 784, encoder_hid, num_latent, dec_hid, 10, 0.,
                         0., supervised=args.supervised,
                         distribution=args.distribution)

# set up input/output variables
X = network_dump['x']
net = network_dump['net']

# get training output: cost = sum of costs
cost = network_dump['cost']
costs = network_dump['costs']

# evaluation output. Also includes output of transform for plotting
output_eval = network_dump['output_eval']

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

# collect parameters
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

# classification or density estimation
if args.supervised:
    y = network_dump['y']
    train = theano.function([X, y], costs, updates=updates, mode=mode)
else:
    train = theano.function([X], costs, updates=updates, mode=mode)

# classifier
eval = theano.function([X], [output_eval], mode=mode)

# data reconstruction
if args.reconstruct:
    X_hat = network_dump['reconstruct']
    inputs = [X, y] if args.supervised else [X]
    reconstruct = theano.function(inputs=inputs, outputs=X_hat, mode=mode)

# data sampling
if args.generate:
    Z = network_dump['z']
    X_hat_gen = network_dump['sample']
    inputs = [Z, y] if args.supervised else [Z]
    generate = theano.function(inputs=inputs, outputs=X_hat_gen, mode=mode)


def train_epoch(stream):
    costs = []
    for batch in stream.get_epoch_iterator():
        train_out = train(*batch) if args.supervised else train(batch[0])
        costs.append(train_out)

    if args.reconstruct:
        recs = reconstruct(*batch) if args.supervised else reconstruct(batch[0])
        visualize(n, recs, [8,8], 'recons_{}_{}_'.format(args.distribution,
                                                         SUPERVISED))
    if args.generate:
        samples = generate(z_vals(args.distribution, z_shape), batch[1]) if \
            args.supervised else generate(z_vals(args.distribution, z_shape))
        visualize(n, samples, [8,8], 'samples_{}_{}_'.format(args.distribution,
                                                             SUPERVISED))


    rval = np.mean(costs, axis=0)
    task = 'nll' if args.supervised else 'dual recon err'
    to_print = rval.tolist() + [task]
    print 'TRAIN: \trec err: {0}, \tgen err: {1}, \t{3}: {2}'.format(*to_print)
    return rval


def eval_epoch(stream):
    preds = []
    targets = []
    for batch in stream.get_epoch_iterator():
        preds.extend(eval(batch[0]))
        targets.extend(batch[1])

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    acc = np.mean(preds.argmax(1) == targets.argmax(1)) # accuracy
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
        print "Epoch {}: Train cost {}, train acc {}, val acc {}".format(
                n, train_costs[-1], train_accs[-1], valid_accs[-1])
        # print 'TIMES: \ttrain {:10.2f}s, \tval {:10.2f}s'.format(t1-t0,
        #                                                          t2-t1)
except KeyboardInterrupt:
    pass

# dump learning curves
save_dump('accs_ace_mnist.pkl', zip(train_accs, valid_accs))
# dump network
theano.misc.pkl_utils.dump(network_dump, 'final_iter_ace_mnist.pkl')