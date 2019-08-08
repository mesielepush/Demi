import os
import pandas as pd
import joblib

base_accounts_dir = os.path.join(os.path.abspath('TweetSuite'),'accounts')
base_accounts     = [account for account in os.listdir(base_accounts_dir) if account != 'vault']

accounts_dir = os.path.abspath('accounts')


def getInput(base_accounts):
    for account in base_accounts:
        print('#################### ', account)
        data = joblib.load(os.path.join(base_accounts_dir,account))
        id_to_drop = []
        
        for ind, text in enumerate(data.text):
            
            if text.startswith('RT') or text.startswith('http'):
                
                id_to_drop.append(data.index[ind])

            else: pass
        if len(id_to_drop) > 1:
            data = data.drop(id_to_drop, axis=0)
        joblib.dump(data,os.path.join(accounts_dir,account[:-17] + '_tweets.pkl'))
getInput(base_accounts)