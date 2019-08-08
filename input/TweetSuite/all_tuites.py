import os
import csv
import math
from time import sleep

import joblib
import tweepy
from tweepy import TweepError

from credentials import credentials


wich_credentials = input('Wich credentials: ')
credentials = credentials[wich_credentials]


accounts_dir = os.path.abspath('ids_{0}'.format(wich_credentials))
accounts = [x[:-8] for x in os.listdir(accounts_dir)]
output_dir = os.path.abspath('ready_{0}'.format(wich_credentials))



access_key = credentials['access_key']
access_secret = credentials['access_secret']
consumer_key = credentials['consumer_key']
consumer_secret = credentials['consumer_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


print('Credentials : ', wich_credentials)

def get_from_ids(account,ids_path):
    print('##################### With credentials: ', wich_credentials)
    print('##################### ACOUNT : ', account)
    ids = joblib.load(ids_path)

    account = account.lower()
    print('total ids: {}'.format(len(ids)))

    all_data = []
    start = 0
    end = 100
    limit = len(ids)
    i = math.ceil(limit / 100)

    for go in range(i):
        print('currently getting {} - {}'.format(start, end))
        sleep(6)  # needed to prevent hitting API rate limit
        id_batch = ids[start:end]
        start += 100
        end += 100
        tweets = api.statuses_lookup(id_batch, tweet_mode = 'extended')
        for tweet in tweets:
            
            all_data.append(dict(tweet._json))
        if go % 100 == 0:
            joblib.dump(all_data, os.path.join(output_dir, 'all_data_{0}.pkl'.format(account)))
    print('metadata collection complete')
    print('creating master json file')
    joblib.dump(all_data, os.path.join(output_dir, 'all_data_{0}.pkl'.format(account)))

def getAllids(accounts):
    for account in accounts:
        
        get_from_ids(account, os.path.join(accounts_dir, account + '_ids.pkl'))

print('### Just one account?')
print('### If you anwser is not "y" all the accounts in ids_{0} will be downloaded'.format(wich_credentials))

all_or_one = input('### Your answer please: ')
if all_or_one == 'y':
    account = input('### Ok. Wich account: ')
    get_from_ids(account, os.path.join(accounts_dir, account + '_ids.pkl'))
else:
    getAllids(accounts)
