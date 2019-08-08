from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import random
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix as cmx

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


os.environ['PYTHONHASHSEED']='666'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(666)
np.random.seed(666)
current_path = os.getcwd()
model_dir = os.path.join(current_path.split("scripts",1)[0],'models')

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    
    model = models.Sequential()
    model.add(Dropout(rate = dropout_rate, input_shape = input_shape))

    for _ in range(layers-1):
        model.add(Dense(units = units, activation='relu'))
        model.add(Dropout(rate =dropout_rate))

    model.add(Dense(units = units, activation = activation))
    return model

def train_mlp_model(data,mod,kwargs):

    TOP_K = 200000

    if not os.path.exists(os.path.join(model_dir, mod, 'mlp')):
        os.makedirs(os.path.join(model_dir, mod, 'mlp'))
    
    os.chdir(os.path.join(model_dir, mod, 'mlp'))
    (x_train, y_train), (x_test, y_test) = data

    num_classes = kwargs['num_classes']
    patience = kwargs['patience']
    units = kwargs['units']
    epochs = kwargs['epochs']
    layers = kwargs['layers']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    dropout_rate = kwargs['dropout_rate']
    
    
    model = mlp_model(layers, units, dropout_rate, x_train.shape[1:], num_classes)
    
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr= learning_rate)

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
            verbose=1,  
            batch_size=batch_size)

    model_things = {  'num_classes': num_classes,
                      'batch_size': batch_size,
                      'layers': layers,
                      'units': units,
                      'dropout_rate': dropout_rate}

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    pred = model.predict_classes(x_test)
    print(cmx(y_test, pred))

    joblib.dump(model_things, os.path.join(model_dir,mod, 'mlp',r'{0}_model_things.pkl'.format(mod)))
    joblib.dump(history, os.path.join(model_dir,mod,'mlp',r'{0}_history.pkl'.format(mod)))

    return max(history['val_acc']), history

