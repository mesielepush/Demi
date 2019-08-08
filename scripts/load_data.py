import os
import random
import numpy as np 
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

current_path =  os.getcwd()
accounts_dir = os.path.join(current_path.split("scripts",1)[0],'input','accounts')
pre_data_dir = os.path.join(current_path.split("scripts",1)[0],'input','pre_data')


###############  This is to load the LIWC dictionaries if you have them.

#batteries_dir = os.path.join(current_path.split("scripts",1)[0],'input','liwc_batteries')

#values     = joblib.load(os.path.join(batteries_dir,'values.pkl'))
#counter    = joblib.load(os.path.join(batteries_dir,'counter.pkl'))
#categories = joblib.load(os.path.join(batteries_dir,'categories.pkl'))

def getData(mode: 'accounts for the model',mod:'concat name of model data',trim = True):

    """ Loads the neccesary 'tweets.pkl' files.
        Returns a tuple for train/test (80%/20%).
        
        input: model = (a,b) ==> (a_tweets.pkl, b_tweets.pkl). 
        output: (train_data, y_train):80%, (test_data, y_test):20%
    """
    # 'datas' insted of data beacuse this variable,
    #  is going to be the concatenation of at least 2 classes or data.
    datas = []
       
    if not os.path.exists(os.path.join(pre_data_dir,mod)):
        os.makedirs(os.path.join(pre_data_dir,mod))
        
    os.chdir(os.path.join(pre_data_dir,mod))
    
    try:
        # checking if we already have the train/test set
        # since there is no reason to make it more than once
        
        train_data = joblib.load('{0}_train_data.pkl'.format(mod))
        test_data  = joblib.load('{0}_test_data.pkl'.format(mod))
        y_test     = joblib.load('{0}_y_test.pkl'.format(mod))
        y_train    = joblib.load('{0}_y_train.pkl'.format(mod))
        return (train_data, y_train), (test_data, y_test)
    
    except:
        
        for account in mode:

            # appending model data
            raw = os.path.join(accounts_dir,'{0}_tweets.pkl'.format(account))
            raw = joblib.load(raw)
            datas.append(raw)

            print('############')
            print('##### {0} Tweets from {1} |'.format(len(raw.text),account))
            print('############')

        for n, raw in enumerate(datas):
            #changing the class names (str) for an integral.
            
            n = int(n)
            raw['label'] = [n] * len(raw)
            datas[n] = raw.drop('account', axis=1)
        # if your classes are to unbalanced, it's better to cut it to the length of the minor one.
        # it is posible, that you could use the remainder to have more testing data for the bigger classes
        # however here for simplicity that is not implemented
        if trim == True:
            minimum = min([len(dataframe) for dataframe in datas])
            print('########################################### min:')
            print(minimum)
            datas = [unit[:minimum] for unit in datas]
        for i in datas:
            print('#### LENS : ',len(i))
        #stating final concate dic.
        final = {'train':[],'test':[]}
        for data in datas:
            #spliting for train/test
            # it's worth noticing that is spliting each class independently
            # this is to balance for class sizes and it is redundant if trim is set to True
            # it ensures that every class has 20% of itself for testing. 
            train_size = int(len(data) * .8)
            test = data.sample(len(data) - train_size)
            train = data.drop(test.index)
            final['train'].append(train)
            final['test'].append(test)

        # concatenating train and test
        train_data = pd.concat(final['train']).reset_index(drop = True).text.to_list()
        test_data  = pd.concat(final['test'] ).reset_index(drop = True).text.to_list()

        y_train = pd.concat(final['train']).reset_index(drop = True).label.to_list()
        y_test = pd.concat(final['test']).reset_index(drop = True).label.to_list()

        # shuffling train/test
        shufle_train = shuffle(pd.DataFrame({'data': train_data, 'label': y_train}))
        shufle_test = shuffle(pd.DataFrame({'data': test_data, 'label': y_test}))
        # converting pd.Series to array of strings.
        train_data = shufle_train.data.to_list()
        test_data  = shufle_test.data.to_list()
        y_train    = shufle_train.label.to_list()
        y_test     = shufle_test.label.to_list()
        # saving final files. 
        joblib.dump(train_data,'{0}_train_data.pkl'.format(mod))
        joblib.dump(test_data,'{0}_test_data.pkl'.format(mod))
        joblib.dump(y_train,'{0}_y_train.pkl'.format(mod))
        joblib.dump(y_test,'{0}_y_test.pkl'.format(mod))
        
        return (train_data, y_train), (test_data, y_test)

def tfidfvec(data:'tuple from getData()', mod:'concat name of model data'):
    """ 
        Perform tfidf transformation for str data,
        takes the tuple from getData(),
        returns tranform tuple.
        input: getData() ==> (train_data, y_train):80%, (test_data, y_test):20%
        output: tfidf(train_data, y_train), tfidf(test_data, y_test)
    """
    
    
    # choosing the directory so we don't have to state it later
    os.chdir(os.path.join(pre_data_dir, mod))
    

    try:
        # checking if the tfidf transformation was already made,
        # since there are no reason to make it more than once
        (x_train, y_train), (x_test, y_test) = data
        x_train = joblib.load('{0}_x_train_tfidf.pkl'.format(mod))
        x_test = joblib.load('{0}_x_test_tfidf.pkl'.format(mod))
        
        return (x_train, y_train), (x_test, y_test)

    except:
        # choosing the ngrams for the transformation
        NGRAM_RANGE = (1, 2)
        # seting maximum for different words
        TOP_K = 200000
        # seting word as unit for analisys
        TOKEN_MODE = 'word'
        # this one is self-explanatory
        MIN_DOCUMENT_FREQUENCY = 2
        # loads result from getData()
        (x_train, y_train), (x_test, y_test) = data

        kwargs = {
                'ngram_range': NGRAM_RANGE, 
                'dtype': 'int32',
                'decode_error': 'replace',
                'analyzer': TOKEN_MODE,
                'min_df': MIN_DOCUMENT_FREQUENCY,
        }
        # states transformer
        vectorizer = TfidfVectorizer(**kwargs)
        # fit with train and transform test
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        # selecting best features
        selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
        # fitting data
        selector.fit(x_train, y_train)
        # applying tranformation to data
        x_train = selector.transform(x_train)
        x_test = selector.transform(x_test)
        # transform to float
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # saving tfidf tranformer since we are going to need it
        # to compare new data with this model 
        joblib.dump(vectorizer,'{0}_tfidf.pkl'.format(mod))
        # saving transformer for best-features for the same upper reasons
        joblib.dump(selector,'{0}_selector.pkl'.format(mod))
        #saving tranform train/test data
        joblib.dump(x_train,'{0}_x_train_tfidf.pkl'.format(mod))
        joblib.dump(x_test,'{0}_x_test_tfidf.pkl'.format(mod))
        

        return (x_train, y_train), (x_test, y_test)

def seq2int(data:'tuple from getData()', mod:'concat name of model data'):
    """ 
        Assing string data to random integrals,
        takes the tuple from getData(),
        returns tranform tuple and word_index.
        input: getData() ==> (train_data, y_train):80%, (test_data, y_test):20%
        output: int(train_data, y_train), int(test_data, y_test), word_index
    """
    (x_train, y_train), (x_test, y_test) = data

    try:
        # checking if we already have the data since there are no reason to make it more than once
        x_train = joblib.load(os.path.join(pre_data_dir, mod, '{0}_x_train_sequence.pkl'.format(mod)))
        x_test  = joblib.load(os.path.join(pre_data_dir, mod, '{0}_x_test_sequence.pkl'.format(mod)))
        tokenizer = joblib.load(os.path.join(pre_data_dir, mod, '{0}_word_index.pkl'.format(mod)))

        return (x_train, y_train), (x_test, y_test), tokenizer[0]

    except:
        # seting maximum for different words
        TOP_K = 20000
        # setting maximum length for array of words
        MAX_SEQUENCE_LENGTH = 500
        # stating tranformer 
        tokenizer = text.Tokenizer(num_words=TOP_K)
        # fitting on train set
        tokenizer.fit_on_texts(x_train)
        # tranforming on train/test sets
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        # checking if the longest data is longer than the MAX_SEQUENCE_LENGTH
        # since we work with tweets, MAX_SEQUENCE_LENGTH is always greater. 
        max_length = len(max(x_train, key=len))
        if max_length > MAX_SEQUENCE_LENGTH:
            max_length = MAX_SEQUENCE_LENGTH
        # padding sequence so data base has always same length
        x_train = sequence.pad_sequences(x_train, maxlen=max_length)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
        
        # saving final database
        joblib.dump(x_train, os.path.join(pre_data_dir, mod, '{0}_x_train_sequence.pkl'.format(mod)))
        joblib.dump(x_test, os.path.join(pre_data_dir, mod, '{0}_x_test_sequence.pkl'.format(mod)))
        joblib.dump([tokenizer.word_index,max_length], os.path.join(pre_data_dir,mod, '{0}_word_index.pkl'.format(mod)))
        
        return (x_train, y_train), (x_test, y_test), tokenizer.word_index