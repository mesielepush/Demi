import os
import random
import numpy as np 
import pandas as pd

import joblib
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

current_path =  os.getcwd()
model_dir = os.path.join(current_path.split("scripts",1)[0],'models')

def logReg(data,mod,kwargs):

    if not os.path.exists(os.path.join(model_dir,mod,'logReg')):
        os.makedirs(os.path.join(model_dir,mod,'logReg'))
    

    (x_train, y_train), (x_test, y_test) = data

    
    log_reg = LogisticRegression(**kwargs)
    log_reg.fit(x_train, y_train)
    
    results = log_reg.predict(x_test)
    print('############ Model: ',mod)
    print('############ Algorithm: ', 'LogRegression')
    print(confusion_matrix(y_test, results))

    acc = []
    for i in range(len(y_test)):
        if y_test[i] == results[i]:
            acc.append(1)
        else:
            acc.append(0)
    f = pd.Series(acc)

    acc = f.value_counts()[1] /(f.value_counts()[1] + f.value_counts()[0])
    print('ACC: ', acc)
    joblib.dump(log_reg, os.path.join(model_dir,mod,'logReg',r'{0}_logReg.pkl'.format(mod)))

def randomForest(data,mod,kwargs):

    if not os.path.exists(os.path.join(model_dir,mod,'randomForest')):
        os.makedirs(os.path.join(model_dir,mod,'randomForest'))

    (x_train, y_train), (x_test, y_test) = data
    
    forest_clf = RandomForestClassifier(**kwargs)

    forest_clf.fit(x_train, y_train)
    
    results = forest_clf.predict(x_test)
    print('############ Model: ',mod)
    print('############ Algorithm: ', 'randomForest')
    print(confusion_matrix(y_test, results))

    acc = []
    for i in range(len(y_test)):
        if y_test[i] == results[i]:
            acc.append(1)
        else:
            acc.append(0)
    f = pd.Series(acc)

    acc = f.value_counts()[1] /(f.value_counts()[1] + f.value_counts()[0])
    print('ACC: ', acc)
    joblib.dump(forest_clf,os.path.join(model_dir,mod,r'randomForest\{0}_randomForest.pkl'.format(mod)))

def sgd(data,mod,kwargs):

    if not os.path.exists(os.path.join(model_dir,mod,'sgd')):
        os.makedirs(os.path.join(model_dir,mod,'sgd'))

    (x_train, y_train), (x_test, y_test) = data
    sgd_clf = SGDClassifier(**kwargs)
    sgd_clf.fit(x_train, y_train)
    
    results = sgd_clf.predict(x_test)
    print('############ Model: ',mod)
    print('############ Algorithm: ', ' SGD ')
    print(confusion_matrix(y_test, results))

    acc = []
    for i in range(len(y_test)):
        if y_test[i] == results[i]:
            acc.append(1)
        else:
            acc.append(0)
    f = pd.Series(acc)

    acc = f.value_counts()[1] /(f.value_counts()[1] + f.value_counts()[0])
    print('ACC: ', acc)
    joblib.dump(sgd_clf,os.path.join(model_dir,mod,r'sgd\{0}_sgd.pkl'.format(mod)))

def svc(data, mod, kwargs):

    if not os.path.exists(os.path.join(model_dir,mod,'svc')):
        os.makedirs(os.path.join(model_dir,mod,'svc'))
    
    (x_train, y_train), (x_test, y_test) = data

    svc = SVC(**kwargs)
    
    svc.fit(x_train, y_train)
    
    results = svc.predict(x_test)
    print('############ Model: ',mod)
    print('############ Algorithm: ', 'SVC')
    print(confusion_matrix(y_test, results))

    acc = []
    for i in range(len(y_test)):
        if y_test[i] == results[i]:
            acc.append(1)
        else:
            acc.append(0)
    f = pd.Series(acc)

    acc = f.value_counts()[1] /(f.value_counts()[1] + f.value_counts()[0])
    print('ACC: ', acc)
    joblib.dump(svc, os.path.join(model_dir,mod,r'svc\{0}_svc.pkl'.format(mod)))

def svm(data,mod, kwargs):

    if not os.path.exists(os.path.join(model_dir,mod,'svm')):
        os.makedirs(os.path.join(model_dir,mod,'svm'))

    (x_train, y_train), (x_test, y_test) = data

    svm = LinearSVC(**kwargs)
    
    svm.fit(x_train, y_train)
    
    results = svm.predict(x_test)
    print('############ Model: ',mod)
    print('############ Algorithm: ', 'SVM')
    print(confusion_matrix(y_test, results))

    acc = []
    for i in range(len(y_test)):
        if y_test[i] == results[i]:
            acc.append(1)
        else:
            acc.append(0)
    f = pd.Series(acc)

    acc = f.value_counts()[1] /(f.value_counts()[1] + f.value_counts()[0])
    print('ACC: ', acc)
    joblib.dump(svm, os.path.join(model_dir,mod,r'svm\{0}_svm.pkl'.format(mod)))

