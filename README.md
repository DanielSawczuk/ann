# ann
Implementation of simple ANN created for learning purposes. 
Implemented NN is a fully-connected feed-forward network, which can make use of batch stochastic gradient descent, L1 or L2 regularization, softmax or cross-entropy output layer.
See _example.py_ for example usage and _ann.ANN.NetworkBuilder_ for more information.

__Dependecies:__
* Python >3.5
* numpy

__matplotlib__ and __imgaeio__ are used in the example script but they are no needed for ANN.

```.py
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
```     
