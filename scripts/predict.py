import os
import random
import numpy as np 
import pandas as pd
import joblib
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text as text_pre

from nn.mlp.train import mlp_model
from nn.sepCnn.train_sepcnn import sepcnn_model
from nn.simple_lstm.train_lstm import lstm_model

algorithms = ['logReg','randomForest','sgd','svc','svm','simpleTree']
nns = ['mlp','sepcnn']

current_path = os.getcwd()
accounts_dir = os.path.join(current_path.split("scripts",1)[0],'input','accounts')
pre_pred_dir = os.path.join(current_path.split("scripts",1)[0],'input','pred_pred')
pre_data_dir = os.path.join(current_path.split("scripts",1)[0],'input','pre_data')
model_dir    = os.path.join(current_path.split("scripts",1)[0],'models')
predict_dir  = os.path.join(current_path.split("scripts",1)[0],'predictions')

def load_tfidf_data(account, mod):

    if not os.path.exists(os.path.join(pre_pred_dir,account,mod)):
        os.makedirs(os.path.join(pre_pred_dir,account,mod))
    try:
        x_pred_tfidf = joblib.load(os.path.join(pre_pred_dir,account,mod,'x_pred_tfidf.pkl'))
        return x_pred_tfidf
    except: pass
    
    data = joblib.load(os.path.join(accounts_dir, account + '_tweets.pkl')).text.to_list()
   
    tfidf = joblib.load(os.path.join(pre_data_dir, mod,'{0}_tfidf.pkl'.format(mod)))
    selector = joblib.load(os.path.join(pre_data_dir, mod,'{0}_selector.pkl'.format(mod)))
    x_pred_tfidf = tfidf.transform(data)
    x_pred_tfidf = selector.transform(x_pred_tfidf)

    joblib.dump(x_pred_tfidf, os.path.join(pre_pred_dir, account, mod,'x_pred_tfidf.pkl'))

    return x_pred_tfidf
    
def load_seq2vec_data(account, mod):

    data = joblib.load(os.path.join(accounts_dir, account + '_tweets.pkl')).text.to_list()
    date = joblib.load(os.path.join(accounts_dir, account + '_tweets.pkl')).index.to_list()

    word_index, max_len = joblib.load(os.path.join(pre_data_dir, mod, '{0}_word_index.pkl'.format(mod)))
    new = [text_pre.text_to_word_sequence(tweet,filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n') for tweet in data]
    
    x_pred_seq2vec = []
    id_to_drop = []
    counter_ind = 0

    for tweet in new:
        sentence = [word_index.get(i) for i in tweet if word_index.get(i) != None]
        
        if len(sentence)<1:
            x_pred_seq2vec.append(['1','1','1','1'])
            id_to_drop.append(date[counter_ind])
        else:
            x_pred_seq2vec.append(sentence)
        
        counter_ind += 1
    
    x_pred_seq2vec = sequence.pad_sequences(x_pred_seq2vec, maxlen=max_len)
    
    joblib.dump(x_pred_seq2vec, os.path.join(pre_pred_dir, account, mod,'x_pred_seq2vec.pkl'))
    joblib.dump(id_to_drop, os.path.join(pre_pred_dir, account, mod,'id_to_drop.pkl'))

    return x_pred_seq2vec

def predict_ml(account, x_test, mod):

    if not os.path.exists(os.path.join(predict_dir,account,mod)):
        os.makedirs(os.path.join(predict_dir,account,mod))
    
    for algorithm in algorithms:
        try:
            model = joblib.load(os.path.join(model_dir,mod,algorithm,'{0}_{1}.pkl'.format(mod,algorithm)))
            classes = model.predict(x_test)
            joblib.dump(classes, os.path.join(predict_dir, account, mod, algorithm + '.pkl'))
        except:
            pass

def predict_mlp(account, x_test, mod):

    model_things = joblib.load(os.path.join(model_dir,mod,'mlp','{0}_model_things.pkl'.format(mod)))

    data  = x_test

    model = mlp_model(
                    layers = model_things['layers'],
                    units  = model_things['units'],
                    dropout_rate = 0.0,
                    input_shape  = data.shape[1:],
                    num_classes = model_things['num_classes']
                    )

    if model_things['num_classes'] == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    model.load_weights(os.path.join(model_dir,mod,'mlp','cp.ckpt'))

    classes = model.predict_classes(data)
    try:
        print(pd.Series(classes).value_counts())
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'mlp.pkl'))
        print('###### Done Preds from: ', account)
        return classes

    except:
        classes = np.asarray(list(itertools.chain.from_iterable(classes)))
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'mlp.pkl'))
        print('###### Done Preds from: ', account)
        return classes

def predict_sepcnn(account, x_test, mod):

    print('###### Getting Preds from: ', account)
    model_things = joblib.load(os.path.join(model_dir,mod,'sepcnn', r'{0}_things.pkl'.format(mod)))
    
    data = x_test
    
    model = sepcnn_model(blocks = model_things['blocks'],
                        filters= model_things['filters'],
                        kernel_size = model_things['kernel_size'],
                        embedding_dim = model_things['embedding_dim'],
                        dropout_rate = 0.0,
                        pool_size = model_things['pool_size'],
                        input_shape = data.shape[1:],
                        num_classes = model_things['num_classes'],
                        num_features = model_things['num_features'],
                        )

    if model_things['num_classes'] == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.load_weights(os.path.join(model_dir,mod,'sepcnn','cp.ckpt'))

    classes = model.predict_classes(data)
    
    if not os.path.exists(os.path.join(predict_dir, account,mod)):
        os.makedirs(os.path.join(predict_dir, account,mod))
    
    try:
        print(pd.Series(classes).value_counts())
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'sepcnn.pkl'))
        print('###### Done Preds from: ', account)
        return classes

    except:
        classes = np.asarray(list(itertools.chain.from_iterable(classes)))
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'sepcnn.pkl'))
        print('###### Done Preds from: ', account)
        return classes

def predict_lstm(account, x_test, mod):

    print('###### Getting Preds from: ', account)
    model_things = joblib.load(os.path.join(model_dir,mod,'lstm', r'{0}_things.pkl'.format(mod)))

    data = x_test

    model = lstm_model(
                num_classes   = model_things['num_classes'  ],
                num_features  = model_things['num_features' ],
                embedding_dim = model_things['embedding_dim'],
                dropout_rate  = 0.0
                )
    
    if model_things['num_classes'] == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.load_weights(os.path.join(model_dir,mod,'lstm','cp.ckpt'))

    classes = model.predict_classes(data)

    if not os.path.exists(os.path.join(predict_dir, account, mod)):
        os.makedirs(os.path.join(predict_dir, account, mod))
    try:
        print(pd.Series(classes).value_counts())
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'lstm.pkl'))
        print('###### Done Preds from: ', account)
        return classes

    except:
        classes = np.asarray(list(itertools.chain.from_iterable(classes)))
        joblib.dump(classes,os.path.join(predict_dir, account,mod,'lstm.pkl'))
        print('###### Done Preds from: ', account)
        return classes
    

