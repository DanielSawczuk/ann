import numpy as np
import matplotlib.pyplot as plt
import ann
import ann_utils as ut
import math as m
import imageio
import mnist_loader


def train_mnist():
    # prepare data
    training_data, test_data = mnist_loader.load_mnist()

    x_t, y_t = test_data
    x_tr, y_tr = training_data

    # shuffle training set
    perm = np.arange(x_tr.shape[1])
    np.random.shuffle(perm)
    x_tr = x_tr[:, perm]
    y_tr = y_tr[:, perm]

    # distinguish validation set
    validation_set_size = 10000
    x_v = x_tr[:, :validation_set_size]
    y_v = y_tr[:, :validation_set_size]

    x_tr = x_tr[:, validation_set_size:]
    y_tr = y_tr[:, validation_set_size:]

    # implement callback
    learning = []

    def callback(epoch, test, time):
        acc_v, cost_v = test(x_v, y_v)
        acc_t, cost_t = test(x_tr, y_tr)
        learning.append([epoch, acc_v, cost_v, time, acc_t, cost_t])
        print("Epoch: {0}\t | Validation: accuracy:{1} cost:{2:.3f}"
              " Training: accuracy:{3} cost:{4:.3f} {5:.3f}s"
              .format(epoch, acc_v, cost_v, acc_t, cost_t, time))

    # build network
    network = ann.ANN.NetworkBuilder() \
        .addInputLayer(784) \
        .addLayer(100, ut.sigmoid, ut.sigmoid_derivative) \
        .addLayer(100, ut.sigmoid, ut.sigmoid_derivative) \
        .addOutputLayer(10, ann.ANN.NetworkBuilder.softmax) \
        .addRegularization(ann.ANN.NetworkBuilder.L2) \
        .build()

    # start learning
    network.train(training_data=(x_tr, y_tr),
                  epochs=35,
                  learning_rate=0.5,
                  batch_size=50,
                  regularization_rate=5,
                  callback=callback)

    # evaluate on test data
    acc, cost = network.test(x_t, y_t)
    print("Test: accuracy: {}/{} | cost: {}".format(acc, x_t.shape[1], cost))
    plot_mnist_learning(learning)

    # load 28x28 gray-scale image and use network to recognize the digit
    recognize_digit(network, "./three.bmp")

    # serialize network
    print("Saving network: \"./net\"")
    ut.save(network, "./net")

    # load network
    print("Loading network: \"./net\"")
    network = ut.load("./net")

    # load 28x28 gray-scale image and use network to recognize the digit
    print("After loading:")
    recognize_digit(network, "./three.bmp")

    learning = []
    # more training
    network.train(training_data=(x_tr, y_tr),
                  epochs=20,
                  learning_rate=0.01,
                  batch_size=50,
                  regularization_rate=5,
                  callback=callback)

    # evaluate on test data
    acc, cost = network.test(x_t, y_t)
    print("Test: accuracy: {}/{} | cost: {}".format(acc, x_t.shape[1], cost))

    # load 28x28 gray-scale image and use network to recognize the digit
    print("After more training")
    recognize_digit(network, "./three.bmp")

    plot_mnist_learning(learning)


def train_fun():
    # prepare data
    fun = lambda x: np.sin(x - 2) + np.tanh(2 * x)
    learning_interval = (-5 * m.pi, 5 * m.pi)
    training_data = generate_training_data(100, learning_interval, fun)
    test_data = generate_training_data(100, learning_interval, fun, noise=False)
    validation_data = generate_training_data(100, learning_interval, fun, noise=False)

    x_v, y_v = ut.column_stack(validation_data)
    x_t, y_t = ut.column_stack(test_data)
    x_tr, y_tr = ut.column_stack(training_data)

    # implement callback
    learning = []

    def callback(epoch, test, time):
        cost_v = test(x_v, y_v, accuracy=False)
        cost_tr = test(x_tr, y_tr, accuracy=False)
        learning.append([epoch, cost_v, time, cost_tr])
        print("Epoch: {0}\t | Validation: cost: {1:.3f} Training: cost:{2:.3f} {3:.3f}s"
              .format(epoch, cost_v, cost_tr, time))

    # build network
    network = ann.ANN.NetworkBuilder() \
        .addInputLayer(1) \
        .addLayer(100, lambda x: np.tanh(x), lambda x: 1 - (np.tanh(x) * np.tanh(x))) \
        .addLayer(80, lambda x: np.tanh(x), lambda x: 1 - (np.tanh(x) * np.tanh(x))) \
        .addStandardOutputLayer(1, lambda x: x, lambda x: 1) \
        .addCostFunction(ut.MSE, ut.MSE_derivative) \
        .addRegularization(ann.ANN.NetworkBuilder.L2) \
        .set_dtype(np.float32) \
        .build()

    # start learning
    network.train(training_data=(x_tr, y_tr),
                  epochs=2500,
                  learning_rate=0.004,
                  batch_size=25,
                  regularization_rate=1,
                  callback=callback)

    # evaluate on test data
    cost = network.test(x_t, y_t, accuracy=False)
    print("Test: cost: {}".format(cost))

    # plot function, training data and approximation
    xs = np.arange(-8 * m.pi, 8 * m.pi, 0.02)
    ys = fun(xs)
    x_y = np.array([[x, network.evaluate(x)] for x in xs])

    learning = np.array(learning)

    fig, (ax11, ax12, ax13) = plt.subplots(1, 3)

    ax11.set(title="Validation cost")
    ax11.set_ylabel("cost")
    ax11.set_xlabel("epochs")
    ax11.plot(learning[:, 0], learning[:, 1], 'k')
    ax11.grid()

    ax12.set(title="Training cost")
    ax12.set_ylabel("cost")
    ax12.set_xlabel("epochs")
    ax12.plot(learning[:, 0], learning[:, 3], 'k')
    ax12.grid()

    ax13.plot(training_data[:, 0], training_data[:, 1], 'r.', label='training data')
    ax13.plot(x_y[:, 0], x_y[:, 1], 'k', label="approximation")
    ax13.plot(xs, ys, 'b', label="function")
    ax13.set(title="Regression")
    ax13.legend()
    ax13.grid()

    plt.show()


def recognize_digit(network, path):
    im = np.array(imageio.imread(path, as_gray="True"))
    im /= 255
    im = [ut.vec_transpose_to_2d(t) for t in im]
    im = np.row_stack(im)
    ans = network.evaluate(im)
    print("Should be '3' is ", np.argmax(ans))


def generate_training_data(size, interval: tuple, fun, noise=True):
    arguments = np.random.random(size)
    interval_length = interval[1] - interval[0]
    arguments *= interval_length
    arguments += interval[0]
    arguments.sort()
    data = np.array([[x, fun(x)] for x in arguments])
    if noise:
        noise = np.random.normal(0, 0.15, size)
        data[:, 1] += noise
    return data


def plot_mnist_learning(learning):
    learning = np.array(learning)
    fig, ((ax11, ax12, ax131), (ax21, ax22, ax23)) = plt.subplots(2, 3)

    ax11.set(title="Validation accuracy")
    ax11.set_ylabel("accuracy")
    ax11.set_xlabel("epochs")
    ax11.plot(learning[:, 0], learning[:, 1], 'k')
    ax11.plot([0, learning[:, 0][-1]], [10000, 10000], color='g')
    ax11.grid()

    ax12.set(title="Validation cost")
    ax12.set_ylabel("cost")
    ax12.set_xlabel("epochs")
    ax12.plot(learning[:, 0], learning[:, 2], 'k')
    ax12.grid()

    ax21.set(title="Training accuracy")
    ax21.set_ylabel("accuracy")
    ax21.set_xlabel("epochs")
    ax21.plot(learning[:, 0], learning[:, -2], 'k')
    ax21.plot([0, learning[:, 0][-1]], [50000, 50000], color='g')
    ax21.grid()

    ax22.set(title="Training cost")
    ax22.set_ylabel("cost")
    ax22.set_xlabel("epochs")
    ax22.plot(learning[:, 0], learning[:, -1], 'k')
    ax22.grid()

    cumm_time = learning[:, 3].cumsum()
    ax131.set_ylabel("accuracy", color='b')
    ax131.set_xlabel("time [s]")
    ax131.set(title="Accuracy and cost over time [s]")
    ax131.tick_params('y', colors='b')
    ax131.plot(cumm_time, learning[:, 1], color='b')

    ax132 = ax131.twinx()
    ax132.set_ylabel("cost [MSE]", color='r')
    ax132.tick_params('y', colors='r')
    ax132.plot(cumm_time, learning[:, 2], color='r')
    ax131.grid()

    plt.show()


train_mnist()
train_fun()
