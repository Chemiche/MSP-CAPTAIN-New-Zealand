import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
np.set_printoptions(suppress=True, precision=3)
import os, glob
import pickle as pkl
import scipy.stats

def build_nn(Xt,
              dense_nodes=None,
              dense_act_f='relu',
              output_nodes=1,
              output_act_f='sigmoid',
              loss_f='mse',
              dropout_rate=0.,
              verbose=1):
    if dense_nodes is None:
        dense_nodes = [32]
    model = keras.Sequential()

    for i in range(len(dense_nodes)):
        model.add(layers.Dense(dense_nodes[i],
                               activation=dense_act_f,
                               input_shape=Xt.shape[1:]))

    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(output_nodes,
                           activation=output_act_f))

    if verbose:
        print(model.summary())

    model.compile(loss=loss_f,
                  optimizer="adam",
                  metrics=['mse'])
    return model


def fit_nn(Xt, Yt, model,
            criterion="val_loss",
            patience=10,
            verbose=1,
            batch_size=100,
            max_epochs=1000,
            validation_split=0.2
            ):
    early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                               patience=patience,
                                               restore_best_weights=True)
    history = model.fit(Xt, Yt,
                        epochs=max_epochs,
                        validation_split=validation_split,
                        verbose=verbose,
                        callbacks=[early_stop],
                        batch_size=batch_size
                        )
    return history