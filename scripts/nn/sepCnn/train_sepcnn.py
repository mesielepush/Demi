from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix as cmx

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import AveragePooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

os.environ['PYTHONHASHSEED']='666'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(666)
np.random.seed(666)

current_path =  os.getcwd()

model_dir = os.path.join(current_path.split("scripts",1)[0],'models')

TOP_K = 20000

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features
                 ):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    
    model = models.Sequential()
       
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='tanh',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
        
        model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='tanh',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
        model.add(AveragePooling1D(pool_size=pool_size))
        

    model.add(SeparableConv1D(filters=filters * 2,
                            kernel_size=kernel_size,
                            activation='tanh',
                            bias_initializer='random_uniform',
                            depthwise_initializer='random_uniform',
                            padding='same'))
    
    model.add(SeparableConv1D(filters=filters * 2,
                            kernel_size=kernel_size,
                            activation='tanh',
                            bias_initializer='random_uniform',
                            depthwise_initializer='random_uniform',
                            padding='same'))
    
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units, activation=activation))
    return model


def train_sequence_model(data, num_classes, mod,
                                            patience,
                                            learning_rate,
                                            epochs,  
                                            batch_size,   
                                            blocks,     
                                            filters, 
                                            dropout_rate,
                                            embedding_dim,
                                            kernel_size,  
                                            pool_size,
                                            ):

    if not os.path.exists(os.path.join(model_dir,mod,'sepcnn')):
        os.makedirs(os.path.join(model_dir,mod,'sepcnn'))

    os.chdir(os.path.join(model_dir,mod,'sepcnn'))
    (x_train, y_train), (x_test, y_test), word_index = data
    
    try:
        num_features = min(len(word_index) + 1, TOP_K)
    except:
        num_features = word_index
    
    model = sepcnn_model(blocks,
                        filters,
                        kernel_size,
                        embedding_dim,
                        dropout_rate,
                        pool_size,
                        x_train.shape[1:],
                        num_classes,
                        num_features)

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
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
    model_things = {'blocks':blocks,
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'embedding_dim': embedding_dim,
                    'dropout_rate': dropout_rate,
                    'pool_size': pool_size,
                    'input_shape': x_train.shape[1:],
                    'num_classes': num_classes,
                    'num_features': num_features
                    }
    pred = model.predict_classes(x_test)
    print(cmx(y_test, pred))


    joblib.dump(model_things,os.path.join(model_dir,mod,'sepcnn','{0}_things.pkl'.format(mod)))
    joblib.dump(history,os.path.join(model_dir,mod,'sepcnn','{0}_history.pkl'.format(mod)))

    return max(history['val_acc']), min(history['val_loss'])