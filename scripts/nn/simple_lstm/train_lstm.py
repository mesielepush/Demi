from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix as cmx
TOP_K = 20000

os.environ['PYTHONHASHSEED']='666'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(666)
np.random.seed(666)
current_path = os.getcwd()
pre_data_dir =  os.path.join(current_path.split("scripts",1)[0],'input','pre_data')
model_dir    = os.path.join(current_path.split("scripts",1)[0],'models')


def lstm_model(
                num_classes,
                num_features,
                embedding_dim,
                dropout_rate,
                ):
    if num_classes == 2:
        last_act = 'sigmoid'
        units = 1
    else:
        last_act = 'softmax'
        units = num_classes

    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_features, embedding_dim),
        tf.keras.layers.Bidirectional(tf.compat.v2.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(embedding_dim/2, activation='relu'),
        tf.keras.layers.Dense(units, activation = last_act)
    ])
    return model

def lstm(data, mod,
                num_classes,
                embedding_dim,
                dropout_rate,
                learning_rate,
                patience,
                epochs,
                batch_size,
                ):

    if not os.path.exists(os.path.join(model_dir,mod,'lstm')):
        os.makedirs(os.path.join(model_dir,mod,'lstm'))
    
    os.chdir(os.path.join(model_dir,mod,'lstm'))

    (x_train, y_train), (x_test, y_test), word_index = data

    num_features = min(len(word_index) + 1, TOP_K)

    model = lstm_model(
                        num_classes,
                        num_features,
                        embedding_dim,
                        dropout_rate,                        
                        )

    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    checkpoint_path = 'cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience = patience,
                                                        verbose  = 2,
                                                        mode     = 'min',
                                                        restore_best_weights=True)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_best_only   = True,
                                                 monitor = 'val_loss',
                                                 mode = 'min',
                                                 save_weights_only= True,
                                                 verbose=2)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    callbacks = [earlyStopping, cp_callback]

    history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            verbose=2,
            batch_size = batch_size)
    history = history.history
    model_things = {
                    'embedding_dim': embedding_dim,
                    'dropout_rate': dropout_rate,
                    'input_shape': x_train.shape[1:],
                    'num_classes': num_classes,
                    'num_features': num_features
                    }
    pred = model.predict_classes(x_test)
    print(cmx(y_test, pred))
    joblib.dump(model_things,os.path.join(model_dir,mod,'lstm','{0}_things.pkl'.format(mod)))
    joblib.dump(history,os.path.join(model_dir,mod,'lstm','{0}_history.pkl'.format(mod)))

    return max(history['val_acc']), min(history['val_loss'])