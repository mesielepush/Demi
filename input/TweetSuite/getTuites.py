import os
import time

import numpy as np
import pandas as pd
import joblib
import tweepy


os.chdir('accounts')

ACCESS_TOKEN = '1070069769963954176-HpZ1PWSXBYBg8ZxydRXiyl1O3tINlH'
ACCESS_TOKEN_SECRET = '69mA8wXwZe6n5IN1BAIu2mRXN2QamAsQmjQs8D5w7GhA9'
CONSUMER_KEY = '36sx54BYaqK87yspWjxvCOa4v'
CONSUMER_SECRET = 'GlCr6ILDasundDy8f8tD1hRT4sBFcYcJ5eDmQwIXsXJ6CYaCGI'

auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

accounts = [account[:-17] for account in os.listdir(os.getcwd()) if account != 'vault']
print(accounts)

def getTuit(account, after = None):

    account = account.lower()
    
    n_tuits=0
    data = {'account':[],'date':[],'text':[]}
    
    for stat in tweepy.Cursor(api.user_timeline, id = account, tweet_mode = 'extended').items(3200):
        print('De:|| ',account,'||','---:', stat.full_text[:90],'--|| ',stat.created_at)
        
        if after is not None:
            if  pd.Timestamp(stat.created_at) <= pd.Timestamp(after):
                print('###########################################')
                print('###########################################')
                print('###########################################')
                print('No new tweets left | date reach: ',pd.Timestamp(stat.created_at), ' |')
                print('###########################################')
                print('###########################################')
                print('###########################################')
                break
        
        try:

            data['account'].append(account)
            data['date'].append(pd.Timestamp(stat.created_at))
            data['text'].append(stat.full_text)
            n_tuits+=1

            if n_tuits==3200:
                data = pd.DataFrame(data)
                data = data.sort_values('date')
                data = data.set_index('date', drop=True)
                try:
                    check = joblib.load(r'{0}_final_tweets.pkl'.format(account))
                    joblib.dump(data, r'{0}_tweets_new.pkl'.format(account))

                    print('#################### -| Last {0} tweets from: {1} |  | to pd.DataFrame ".pkl"  |'.format(len(data),account))
                    return data
                except:
                    joblib.dump(data, r'{0}_final_tweets.pkl'.format(account))
                    print('#################### -| Last 3200 tuits from: | ', account, '| to pd.DataFrame ".pkl"  |')
                    return data

                
        except:
            pass
    data = pd.DataFrame(data)
    data = data.sort_values('date')
    data = data.set_index('date', drop=True)

    try:
        check = joblib.load(r'{0}_final_tweets.pkl'.format(account))
        joblib.dump(data, r'{0}_tweets_new.pkl'.format(account))
        print('#################### -| Last {0} tweets from: {1} |  | to pd.DataFrame ".pkl"  |'.format(len(data),account))
        return data

    except:
        joblib.dump(data, r'{0}_final_tweets.pkl'.format(account))
        print('#################### -| Last 3200 tweets from: | ', account, '| to pd.DataFrame ".pkl"  |')
        return data
    
def updateTuits(account):
    try:
        print(os.getcwd())
        old_data = joblib.load(r'{0}_final_tweets.pkl'.format(account))
        old_data = old_data.sort_values(by = 'date', ascending=False)
        for stat in tweepy.Cursor(api.user_timeline, id = account, tweet_mode = 'extended').items(1):
            print('###########################################')
            print('###########################################')
            print('###########################################')
            print('Last tweet from    : ', pd.Timestamp(stat.created_at))
            print('Max old-tweet from : ',max(old_data.index))
            print('###########################################')
            print('###########################################')
            print('###########################################')

            if  pd.Timestamp(stat.created_at) <= pd.Timestamp(max(old_data.index)):
                print('###########################################')
                print('###########################################')
                print('###########################################')
                print('There is no new Tweets for: | ', account)
                print('###########################################')
                print('###########################################')
                print('###########################################')
                return old_data
        
        new_data =  getTuit(account, after = max(old_data.index))
        data = pd.concat([old_data,new_data])
        data = data.sort_values('date')
        joblib.dump(data, r'{0}_final_tweets.pkl'.format(account))
        print('###########################################')
        print('###########################################')
        print('###########################################')
        print('########################## account : ', account)
        print('########################## old_data len = ',len(old_data))
        print('########################## new_data len = ',len(new_data))
        print('########################## TOTAL new len = ',len(data))
        print('###########################################')
        print('###########################################')
        print('###########################################')
        for file in os.listdir(os.getcwd()):
            if file.endswith('_new.pkl'):
                os.remove(os.path.join(os.getcwd(),file))
        return data
    except:
        print('###########################################')
        print('###########################################')
        print('###########################################')
        print('There is no Old tweets from: ',account)
        print('Getting last 3200 tweets')
        print('###########################################')
        print('###########################################')
        print('###########################################')
        return getTuit(account)


def updateAll(accounts):
    for account in accounts:
        updateTuits(account)
getTuit('batiz_bernardo')
#updateAll(accounts)