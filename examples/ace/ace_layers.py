from lasagne import init, nonlinearities
from lasagne.layers import Layer, MergeLayer

import theano
from theano import tensor as T
th_rng = theano.tensor.shared_randomstreams.RandomStreams(9999)

class Dense3DLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)
    A fully connected layer.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units1, num_units2)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, Dense3DLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = Dense3DLayer(l_in, num_units=[50, 100])
    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, dot_axes=[[1], [1]],
                 dot_type='tensor', dimshuffle=(1, 0, 2),
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(Dense3DLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.dot_axes = dot_axes
        self.dot_type = dot_type
        self.shffl_pttrn = dimshuffle

        num_inputs = self.input_shape[-1]
        self.W = self.add_param(W, (num_inputs, ) + tuple(num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, tuple(num_units), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + tuple(self.num_units)

    def get_output_for(self, input, **kwargs):
#         if input.ndim == 2:
#             input = input.dimshuffle('x', 0, 1)
        if input.ndim > 3:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(3)

        if self.dot_type == 'tensor':
            activation = T.tensordot(input, self.W, axes=self.dot_axes)
        elif self.dot_type == 'batched':
            activation = T.batched_dot(input.dimshuffle(self.shffl_pttrn),
                                       self.W.dimshuffle(self.shffl_pttrn)).dimshuffle(self.shffl_pttrn)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0, 1)
        return self.nonlinearity(activation)


def _generate_latent(mu, log_sigma, sampling_class='Gaussian'):
    if sampling_class == 'Gaussian':
        eps = th_rng.normal(mu.shape, dtype=theano.config.floatX)
        z = mu + eps * T.exp(log_sigma)
    elif sampling_class == 'Laplacian':
        eps0 = th_rng.uniform(mu.shape, dtype=theano.config.floatX)
        if T.lt(eps0 , 0.5) == 1:
            z = mu + T.exp(log_sigma) * T.sqrt(0.5) * T.log(eps0 + eps0)
        else:
            z = mu - T.exp(log_sigma) * T.sqrt(0.5) * T.log(2.0 - 2 * eps0)
    # shape: batch_size x n_c x n_z

    return z


class SamplingLayer(MergeLayer):
    """
    """
    def __init__(self, incoming_mu, incoming_log_sigma, sampling_class, **kwargs):
        super(SamplingLayer, self).__init__(
            [incoming_mu, incoming_log_sigma], **kwargs)
        self.sampling_class = sampling_class
        mu_shp, log_sigma_shp = self.input_shapes

#         if len(mu_shp) != 2:
#             raise ValueError("The input network must have a 2-dimensional "
#                              "output shape: (batch_size, num_hidden)")

#         if len(log_sigma_shp) != 2:
#             raise ValueError("The input network must have a 2-dimensional "
#                              "output shape: (batch_size, num_hidden)")
        assert len(mu_shp) == len(log_sigma_shp)


    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        mu, log_sigma = inputs
        return _generate_latent(mu, log_sigma, self.sampling_class)