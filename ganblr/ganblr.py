from .kdb import *
from .utils import *
from .data import DataUtils
import numpy as np
import tensorflow as tf

def _discrim(input_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=input_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class GANBLR:
    """
    The GANBLR Model.
    """
    def __init__(self) -> None:
        self.__d = None
        pass
    
    def fit(self, x, y, batch_size, epochs=10, warmup_epochs=1):
        '''
        Fit the model to the given data.

        Parameters:
        --------
        x, y (numpy.ndarray): Dataset to fit the model. The data should be discrete.
        batch_size (int, optional): Size of the batch to feed the model at each step. Defaults to
            :attr:`200`.
        epochs (int, optional): Number of epochs to use during training. Defaults to :attr:`10`.
        '''
        d = DataUtils(x, y)
        self.constraint = softmax_weight(d.feature_uniques)
        self.__d = d
        self.batch_size = batch_size
        syn_data,weights = self._warmup_run(warmup_epochs)

        discriminator_label = np.stack([np.ones(d.data_size), np.zeros(d.data_size)])
        for i in range(epochs):
            discriminator_input = np.stack([x, syn_data])
            disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)

            disc = _discrim((disc_input.shape[1] - 1))
            disc.fit(disc_input, disc_label, batch_size=batch_size, epochs=1)
            prob_fake = disc.predict(x)
            ls = np.mean(-np.log(np.subtract(1, prob_fake)))
            syn_data,weights = self._run_generator(loss=ls, weights=weights)
        

    def _warmup_run(self, epochs):
        d = self.__d
        tf.keras.backend.clear_session()
        ohex = d.get_ohe_x()
        elr = get_lr(ohex.shape[1], d.class_unique, self.constraint)
        log_elr = elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=epochs)
    
        weights_all = elr.get_weights()
        tf.keras.backend.clear_session()
        weights = elr.get_weights()[0]
        syn_data = sample_synthetic_data(weights, d.feature_uniques, d.class_counts, ohe=False)
        return syn_data, weights_all

    def _run_generator(self, loss):
        d = self.__d
        ohex = d.get_ohe_x()
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(d.num_classes, input_dim=ohex.shape[1], activation='softmax',kernel_constraint=self.constraint))
        model.compile(loss=elr_loss(loss), optimizer='adam', metrics=['accuracy'])
        model.set_weights(weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size,epochs=1)
    
        weights_all = model.get_weights()
        tf.keras.backend.clear_session()
        weights = model.get_weights()[0]
        syn_data = sample_synthetic_data(weights, d.feature_uniques, d.class_counts, ohe=False)
    
        return syn_data, weights_all

    def evaluate(self):
        pass