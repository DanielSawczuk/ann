import numpy as np
import ann_utils as ut
import time

class ANN:
    """
    Simple implementation of fully-connected feed-forward neural network and its
    learning with batch stochastic gradient descent and back-propagation.
    To build network use ann.ANN.NetworkBuilder.
    """

    def __init__(self, shape, activation_functions, cost_function, regularization_expression=None,
                 last_layer_backpropag=None, dtype=np.float32):
        """
        Use ann.ANN.NetworkBuilder.
        """
        self.shape = tuple(shape)
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.regularization_expression = regularization_expression
        self.last_layer_backpropag = last_layer_backpropag
        self.type = dtype
        self.weights = []
        self.biases = []
        self.init_weights()
        self.num_weights = sum([w.size for w in self.weights])
        self.regularization_rate = 0

    def init_weights(self):
        """
        Initializes weights and biases. Can be use to reset a network.
        A neuron's weights are initialized with normal distribution
        with mean 0 and standard deviation 1/sqrt(n) where n is
        a number of inputs of the neuron. Biases are initialized
        with normal distribution distribution of mean 0 and variance 1.
        :return:
        """
        self.weights = np.array([np.random.normal(0, 1 / np.sqrt(y), (y, x)).astype(self.type) for x, y in
                                 zip(self.shape[:-1], self.shape[1:])])
        self.biases = np.array([np.random.randn(y, 1).astype(self.type) for y in self.shape[1:]])

    def train(self, training_data, epochs, learning_rate, batch_size, regularization_rate=0, callback=None):
        """
        Fits network to given training data with batch stochastic gradient descent
        and back-propagation algorithm. Calls callback after every epoch. You can stop
        learning before all epochs are over by returning False in callback method.

        Training data should be a tuple.
        The first element of this tuple should be a matrix of inputs
        and the second a matrix of expected outputs.
        In corresponding columns of input matrix and output matrix
        should be respectively: input to the network and expected response.
        (i.e. single input x = training_data[0][:, 0],
            single output y = training_data[1][:, 0])

        A callback function will be called after every epoch.
        To the callback function will be passed: callback(epoch, test, time)
            epoch: number of last epoch
            test: function, see :func:`~ann.Ann.test`
            time: last epoch learning time in seconds [s]
        If the callback returns 'False' then training will stop.
        Can be used to early stopping implementation.

        :param training_data:
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :param regularization_rate:
        :param callback: function
        :return:
        """
        xs = training_data[0].astype(self.type)
        ys = training_data[1].astype(self.type)

        self.regularization_rate = regularization_rate
        reg_learn = regularization_rate * learning_rate / self.num_weights

        if callback is not None:
            callback(0, self.test, 0)

        d_w = np.array([np.zeros((y, x), dtype=self.type) for x, y in zip(self.shape[:-1], self.shape[1:])])
        d_b = np.array([np.zeros((y, 1), dtype=self.type) for y in self.shape[1:]])

        dC_da = np.array([np.zeros((x, batch_size), dtype=self.type) for x in self.shape[1:]])
        dC_dz = np.array([np.zeros((x, batch_size), dtype=self.type) for x in self.shape[1:]])

        perm = np.arange(xs.shape[1])

        for epoch in range(1, epochs + 1):
            start = time.time()
            np.random.shuffle(perm)
            xs = xs[:, perm]
            ys = ys[:, perm]

            for k in range(0, xs.shape[1], batch_size):
                x = xs[:, k:k + batch_size]
                y = ys[:, k:k + batch_size]

                ut.fill_all(0, d_w, d_b, dC_da, dC_dz)

                a, da_dz = self.feedforward(x)

                self.last_layer_backpropag(self, a, dC_da, dC_dz, d_b, d_w, da_dz, y)

                for l in range(len(self.shape) - 3, -1, -1):
                    dC_da[l] += np.dot(dC_dz[l + 1].T, self.weights[l + 1]).T
                    dC_dz[l] = dC_da[l] * da_dz[l]
                    d_b[l] += ut.vec_transpose_to_2d(np.sum(dC_dz[l], axis=1))
                    dC_dw = dC_dz[l].dot(a[l].T)  # a_ls length is greater by 1, so there is l-1 actually
                    d_w[l] += dC_dw

                d_w = np.nan_to_num(d_w * learning_rate / x.shape[1])
                d_b = np.nan_to_num(d_b * learning_rate / x.shape[1])

                if self.regularization_expression is not None:
                    regularization = self.regularization_expression[1](self.weights, self.num_weights)
                    d_w += reg_learn * regularization

                self.weights -= d_w
                self.biases -= d_b

            end = time.time()
            if callback is not None:
                if callback(epoch, self.test, end - start) is False:
                    return

    def simple_output_layer(self, a, dC_da, dC_dz, d_b, d_w, da_dz, y):
        dC_da[-1] = self.cost_function[1](y, a[-1])
        dC_dz[-1] += dC_da[-1] * da_dz[-1]  # dC_dz <-> dC_db
        d_b[-1] += ut.vec_transpose_to_2d(np.sum(dC_dz[-1], axis=1))
        dC_dw = dC_dz[-1].dot(a[-2].T)
        d_w[-1] += dC_dw

    def cross_entropy_output_layer_backpropag(self, a, dC_da, dC_dz, d_b, d_w, da_dz, y):
        dC_dz[-1] += a[-1] - y  # dC_dz <-> dC_db
        d_b[-1] += ut.vec_transpose_to_2d(np.sum(dC_dz[-1], axis=1))
        dC_dw = dC_dz[-1].dot(a[-2].T)
        d_w[-1] += dC_dw

    def softmax_last_output_backpropag(self, a, dC_da, dC_dz, d_b, d_w, da_dz, y):
        dC_dz[-1] += a[-1] - y  # dC_dz <-> dC_db
        d_b[-1] += ut.vec_transpose_to_2d(np.sum(dC_dz[-1], axis=1))
        dC_dw = dC_dz[-1].dot(a[-2].T)
        d_w[-1] += dC_dw

    def feedforward(self, x):
        a = []
        z = []
        da_dz = []
        a.append(x)
        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            z_l = np.dot(w, a[-1]) + b
            z.append(z_l)
            da_dz.append(f[1](z_l))
            a.append(f[0](z[-1]))
        return a, da_dz

    def evaluate(self, a):
        """
        Returns network output when is given 'a'.
        Argument should be a column vector and
        have size of input layer.

        :param a: network input
        :return:
        """
        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            a = (f[0](np.dot(w, a) + b))
        return a

    def accuracy(self, real_values, approximations):
        val_max = np.argmax(real_values, axis=0)
        app_max = np.argmax(approximations, axis=0)
        return sum(int(x == y) for (x, y) in zip(app_max, val_max))

    def test(self, xs, ys, accuracy=True, cost=True):
        """
        Tests network with given test data and return respectively
        accuracy and/or cost.
        In corresponding columns of xs and ys parameters
        should be respectively: input to the network (xs)
        and expected response (ys).
        (i.e. a network output is a column in approximations matrix)
        :param xs: input to the network
        :param ys: expected response
        :return: network accuracy and cost
        """
        xs = xs.astype(self.type)
        ys = ys.astype(self.type)
        network_output = self.evaluate(xs)
        score = []
        if accuracy:
            score.append(self.accuracy(ys, network_output))
        if cost:
            cst = self.cost_function[0](ys, network_output)
            if self.regularization_expression is not None:
                cst += self.regularization_rate \
                       * self.regularization_expression[0](self.weights, self.num_weights)
            score.append(cst)

        if len(score) == 1:
            return score[0]
        else:
            return tuple(score)

    class NetworkBuilder:
        """ ANN builder. It is required to call at least:
        :func:`~ann.ANN.NetworkBuilder.addInputLayer` and
        :func:`~ann.ANN.NetworkBuilder.addStandardOutputLayer`
        or :func:`~ann.ANN.NetworkBuilder.addOutputLayer`
        For more details read methods' documentations.
        """
        cross_entropy = "CROSS_ENTROPY"
        softmax = "SOFTMAX"
        L1 = "L1"
        L2 = "L2"

        def __init__(self):
            self._shape = []
            self._activation_functions = []
            self._cost_function = None
            self._callback = None
            self._regularization_expression = None
            self._last_layer_backpropag = None
            self._dtype = np.float32
            self._output_layer = None
            self._input_layer = None

        def build(self):
            self._shape.insert(0, self._input_layer)
            self.addLayer(*self._output_layer)
            return ANN(self._shape,
                       self._activation_functions,
                       self._cost_function,
                       self._regularization_expression,
                       self._last_layer_backpropag,
                       self._dtype)

        def addInputLayer(self, input_size):
            """
            Adds input layer with 'input_size' neurons.
            Auxiliary layer, only passes input to proper layers.
            :param input_size:
            :return:
            """
            self._input_layer = input_size
            return self

        def addLayer(self, size, activation_function, activation_function_derivative):
            """
            Adds layer with 'size' neurons and given activation function.
            Call in order rom input to output layers.
            :param size: number of neurons
            :param activation_function:
            :param activation_function_derivative:
            :return:
            """
            self._shape.append(size)
            self._activation_functions.append((activation_function, activation_function_derivative))
            return self

        def addOutputLayer(self, output_size, layer_type: str):
            """
            Adds output layer with 'output_size' neurons.
            Currently only 'softmax' and 'cross_entropy' output layers
            are implemented. Use :func:`~ann.ANN.NetworkBuilder.addStandardOutputLayer`
            and :func:`~ann.ANN.NetworkBuilder.addCostFunction`
            to add different output layers. Adds adequate cost function. Do not override with
            :func:`~and ann.ANN.NetworkBuilder.addCostFunction`.
            :param output_size: number of neurons
            :param layer_type:
            :return:
            """
            if layer_type.upper() == self.cross_entropy:
                self._cost_function = (ut.cross_entropy, ut.placeholder)
                self._output_layer = (output_size, ut.sigmoid, ut.placeholder)
                self._last_layer_backpropag = ANN.cross_entropy_output_layer_backpropag
            elif layer_type.upper() == self.softmax:
                self._cost_function = (ut.log_likelihood, ut.placeholder)
                self._output_layer = (output_size, ut.softmax, ut.placeholder)
                self._last_layer_backpropag = ANN.softmax_last_output_backpropag
            else:
                raise Exception('Supported output layers are: '
                                '\'{0}\', \'{1}\'. Incorrect argument:{2}. '
                                'Use ann.ANN.NetworkBuilder.addStandardOutputLayer '
                                'and ann.ANN.NetworkBuilder.addCostFunction'
                                'to add different output layers.'
                                .format(self.cross_entropy, self.softmax, layer_type))
            return self

        def addStandardOutputLayer(self, output_size, activation_function, activation_function_derivative):
            """
            Adds output layer with given activation function and with 'output_size' neurons.
            Adds MSE as a cost function. You can override cost function
            by calling :func:`~ann.ANN.NetworkBuilder.addCostFunction` after this method.
            :param output_size: number of neurons
            :param activation_function:
            :param activation_function_derivative:
            :return:
            """
            self._output_layer = (output_size, activation_function, activation_function_derivative)
            self._last_layer_backpropag = ANN.simple_output_layer
            self._cost_function = (ut.MSE, ut.MSE_derivative)
            return self

        def addCostFunction(self, cost_function, cost_function_derivative):
            """
            Specifies cost function. Be careful, it overrides a cost function that
            could have been provided earlier by :func:`~ann.ANN.NetworkBuilder.addOutputLayer`
            or :func:`~ann.ANN.NetworkBuilder.addStandardOutputLayer`.
            If :func:`~ann.ANN.NetworkBuilder.addOutputLayer` is specified
            it will have no effect on learning.
            :param cost_function:
            :param cost_function_derivative:
            :return:
            """
            self._cost_function = (cost_function, cost_function_derivative)
            return self

        def addEarlyStopping(self):
            """
            Placeholder. Implement early stopping with your callback function.
            If callback returns 'False' then training will stop.
            See :func:`~ann.ANN.train`
            """
            pass

        def addRegularization(self, regularization_type: str):
            """
            Adds given regularization to model. Currently only 'L1'
            and 'L1' regularizations are implemented. You can define
            your own regularization with :func:`~ann.ANN.addRegularizationExpression`
            Provide regularization rate to :func:`~ann.ANN.train`
            :param regularization_type: 'L1' or 'L2' string
            :return:
            """
            if regularization_type.upper() == self.L1:
                self._regularization_expression = (ut.L1, ut.L1_derivative)
            elif regularization_type.upper() == self.L2:
                self._regularization_expression = (ut.L2, ut.L2_derivative)
            else:
                raise Exception('Supported regularizations are: '
                                '\'L1\', \'L2\'. Incorrect argument:{}'.format(str))
            return self

        def set_dtype(self, dtype):
            """
            Specifies precision of calculations.
            Default is 'numpy.float32'.
            :param dtype: numpy dtype (e.g numpy.float64)
            """
            self._dtype = dtype
            return self
