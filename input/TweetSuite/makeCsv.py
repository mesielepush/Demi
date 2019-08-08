import pandas as pd
import numpy as np
import joblib
import os

new_accounts_dir = os.path.abspath('new_tweets')
print(new_accounts_dir)
new_accounts = [x[9:-4] for x in os.listdir(new_accounts_dir) if len(x)>13]
print(new_accounts)

def makefromjson(account,data):
    dic = {'account':[],'date':[],'text':[]}
    for tweet in data:
        dic['account'].append(account)
        date = tweet['created_at'].replace('+0000','')
        print(date)
        dic['date'].append(pd.Timestamp(date))
        dic['text'].append(tweet['full_text'])
        
    dic = pd.DataFrame(dic)
    dic = dic.sort_values('date')
    dic = dic.set_index('date',drop=True)
    joblib.dump(dic,os.path.join(new_accounts_dir,'pkl','{0}_final_tweets.pkl'.format(account)))

for account in new_accounts:
    data = joblib.load(os.path.join(new_accounts_dir,'all_data_{0}.pkl'.format(account)))
    makefromjson(account,data)