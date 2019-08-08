import pandas as pd
import os
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from scipy.stats import mstats, wilcoxon
from inspect import getsource
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib as mpl
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

current_path = os.getcwd()
model_dict = joblib.load(os.path.join(current_path,'constants','model_dict.pkl'))

def tukey(prediction: 'array of ints', model:'array of string'):
    """ 
        This performs Tukey HSD test,
        input: array(int(predictions)), array(str(model_names))
        this test says if there are significant differences between the classes.
    """
    mc = MultiComparison(prediction,model)
    mc_results = mc.tukeyhsd()
    print(mc_results)
    return mc_results
def make2column(data):
    """
    This function takes the 'raw' results from the processed prediction class
    and transform the database to a two column one: predictions, model_names
    
    """
    dumies = {}
    for num, i in enumerate(data.keys()):
        dumies[i] = num+1
    
    model = []
    prediction = []

    for row in range(len(data)):
        for key in data.keys():
            model.append(key)
            prediction.append(data.iloc[row][key]+1)
    df = pd.DataFrame()
    df['model'] = model
    df['prediction'] = prediction
    
    return df
def target_boolean(target,tuk_text):
    lines = []
    
    for ind, word in enumerate(tuk_text):
        if word == target:
            
            lines.append([ind])
        elif word in ['True','False']:
            
            if len(lines)<1:
                continue
            else:
                lines[-1].append(word)
                if len(lines) == 2: break
    print(lines)
    for line in lines:
        if line[1] == 'False':
            print('Accounts are the SAME')
            return False
    print('Account are Different')
    return True
def makeAnalisys(pfreq, mod, target):
    
    if len(model_dict['reverse'][mod])>2:
        final_proportion = {}
        
        pfreq2= make2column(pfreq)
        tuk = tukey(pfreq2['prediction'],pfreq2['model'] )
        summary = tuk.summary().as_text().split()
        print('MODEL: ',mod)
        is_target_different = target_boolean(target,summary)
        
        total = sum([sum(pfreq[account]) for account in model_dict['reverse'][mod]])
        for account in model_dict['reverse'][mod]:
            final_proportion[account] = sum(pfreq[account])/total
        
        if final_proportion[target] == max(final_proportion.values()):
            is_target_max = True
        else: is_target_max = False
        
        return is_target_different, is_target_max, final_proportion
    
    else:
        
        final_proportion = {}
        
        accounts = [account for account in pfreq.keys()]
        _, p = wilcoxon(pfreq[accounts[0]],pfreq[accounts[1]], correction = True)
        if p < 0.05:
            is_target_different = True
        else: is_target_different = False
            
        total = sum([sum(pfreq[account]) for account in model_dict['reverse'][mod]])
        for account in model_dict['reverse'][mod]:
            final_proportion[account] = sum(pfreq[account])/total
        if target not in accounts:
            is_target_max = False
        else:
            
            if final_proportion[target] == max(final_proportion.values()):
                is_target_max = True
            else: is_target_max = False
        
        return is_target_different, is_target_max, final_proportion
def purefreq(mod,raw):
    final = {}
    dummy = {}
    counter = {}
    for num, account in enumerate(model_dict['reverse'][mod]):
        dummy[num]      = account
        counter[account]= 0
        final[account]  = []
    for ind in range(len(raw)):
        for datum in raw.iloc[ind].values:
            counter[dummy[datum]] += 1
        for account in final.keys():
            final[account].append(counter[account])
        for account in counter.keys():
            counter[account] = 0
    return pd.DataFrame(final)
def runAllAnalisys(party, target):
    
    final_result = {}
    
    for account in party.keys():
        print(account)
        final_result[account] = {}
        for model in party[account].keys():
            print(model)
            data = purefreq(model,party[account][model]['raw'])
            final_result[account][model] = makeAnalisys(data, model, target)
            print('finish: ',model)
    return final_result