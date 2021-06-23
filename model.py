import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow.keras as k


def make_tensorboard_name(net_name, variables):
    import datetime
    # print(f'{net_name=} {variables=}')
    net_name += datetime.datetime.now().strftime(' %y%m%d-%H%M%S')
    # Python 3.7+ guarantees ordered dictionary. Not sure about locals(), but assume order is ok unless shown otherwise.
    # https://stackoverflow.com/questions/5629023/the-order-of-keys-in-dictionaries
    for k, v in variables.items():
        # print(f'{k=} {v=}')
        net_name += f' {k}={v}'
    # print(f'{net_name=}')
    return net_name

def main():
    # TODO: understand how good is the fit.
    iris = load_iris()

    x = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    print(f'{y=}')
    y = OneHotEncoder().fit_transform(y[:, np.newaxis]).toarray()
    print(f'{y=}')

    x = StandardScaler().fit_transform(x)
    print(f'{x=}')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)
    n_features = x.shape[1]
    n_classes = y.shape[1]
    print(f'{n_features=} {n_classes=}')

    plt.clf()

    def plot_projection(subplot, i, j):
        plt.subplot(1, 2, subplot)
        for target, target_name in enumerate(names):
            x_plot = x[y[:, target] == 1.0]
            plt.plot(x_plot[:, i], x_plot[:, j], linestyle='none', marker='o', label=target_name)
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.axis('equal')
        plt.legend()

    plot_projection(1, 0, 1)
    plot_projection(2, 2, 3)

    model = k.models.Sequential()
    layers = 2
    model.add(k.layers.InputLayer(input_shape=n_features))
    for i in range(layers):
        model.add(k.layers.Dense(units=8, activation=k.activations.relu))
    model.add(k.layers.Dense(n_classes, activation=k.activations.softmax))
    model.summary()
    model.compile(loss=k.losses.categorical_crossentropy, optimizer=k.optimizers.Adam(),
                  metrics=[k.metrics.categorical_accuracy])

    log_dir = 'logs/' + make_tensorboard_name('iris', {'layers': layers})
    tensorboard = k.callbacks.TensorBoard(log_dir=log_dir)
    model.fit(x_train, y_train, batch_size=5, epochs=100, verbose=1,
              validation_data=(x_test, y_test), callbacks=[tensorboard])
    model.save('iris.h5')
    y_predicted = model.predict(x_test)
    print(f'{y_predicted=}')



















if __name__ == "__main__":
    main()
