import os
import datetime
import joblib
from time import sleep

import tweepy
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

from credentials import credentials

wich_credentials = input('Wich credentials: ')
credentials = credentials[wich_credentials]

ids_dir = os.path.abspath('ids_{0}'.format(wich_credentials))
chromedriver = r'C:\Python36\chromedriver.exe'

def getAll(account,start,end):

    account = account.lower()
    
    start = datetime.datetime(start[0],start[1],start[2])  # year, month, day
    end = datetime.datetime(end[0],end[1],end[2])  # year, month, day

    delay = 0.10  # time to wait on each page load before reading the page
    options = Options()
    options.add_argument('--disable-gpu')
    options.headless = True
    driver = webdriver.Chrome(options=options, executable_path=chromedriver)

    days = (end - start).days + 1

    id_selector = '.time a.tweet-timestamp'
    tweet_selector = 'li.js-stream-item'
    
    try:
        ids = joblib.load(os.path.join(ids_dir, '{0}_ids.pkl'.format(account)))
    except: ids = []
    


    def format_day(date):
        day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
        month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
        year = str(date.year)
        return '-'.join([year, month, day])

    def form_url(since, until):
        p1 = 'https://twitter.com/search?f=tweets&vertical=default&q=from%3A'
        p2 =  account + '%20since%3A' + since + '%20until%3A' + until + 'include%3Aretweets&src=typd'
        return p1 + p2

    def increment_day(date, i):
        return date + datetime.timedelta(days=i)

    for day in range(days):
        d1 = format_day(increment_day(start, 0))
        d2 = format_day(increment_day(start, 1))
        url = form_url(d1, d2)
        print(url)
        print(d1)
        driver.get(url)
        sleep(delay)

        try:
            found_tweets = driver.find_elements_by_css_selector(tweet_selector)
            increment = 10

            while len(found_tweets) >= increment:
                print('scrolling down to load more tweets')
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                sleep(delay)
                found_tweets = driver.find_elements_by_css_selector(tweet_selector)
                increment += 10

            print('{} tweets found, {} total'.format(len(found_tweets), len(ids)))

            for tweet in found_tweets:
                try:
                    id = tweet.find_element_by_css_selector(id_selector).get_attribute('href').split('/')[-1]
                    ids.append(id)
                    print('############################## ',id)
                except StaleElementReferenceException as e:
                    print('lost element reference', tweet)

        except NoSuchElementException:
            print('no tweets on this day')
        os.system('cls') 
        start = increment_day(start, 1)

        if day % 100 == 0:
            joblib.dump(ids,os.path.join(ids_dir,'{0}_ids.pkl'.format(account)))
        

    joblib.dump(ids,os.path.join(ids_dir,'{0}_ids.pkl'.format(account)))
       
    print('# IDS: ', len(ids))
    print('all done here')

    driver.close()




account = input('From account: ')

try:

    ids = joblib.load(os.path.join(ids_dir, '{0}_ids.pkl'.format(account)))
    start = ids[0]
    end   = ids[-1]

    access_key = credentials['access_key']
    access_secret = credentials['access_secret']
    consumer_key = credentials['consumer_key']
    consumer_secret = credentials['consumer_secret']

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    oldest_tweet = api.get_status(start).created_at
    newest_tweet = api.get_status(end).created_at
    print('###################')
    print('I already have ids for {0}'.format(account))
    print('###################')
    print('### They are from : ',oldest_tweet,'  To : ', newest_tweet)
    print('###################')
    print('The starting data is set to: ', newest_tweet)
    print('###################')
    print("If that's not ok please erase the existing ids file and try again")
    print('###################')

    from_year = newest_tweet.year
    from_month= newest_tweet.month
    from_day  = newest_tweet.day

    to_year  = input('to year : (4 digits) | ')
    print('### Year: ',to_year)
    to_month = input('to month:  | ')
    print('### Month: ',to_month)
    to_day   = input('to day  :  | ')
    print('### Day: ',to_day)

    start = int(from_year), int(from_month), int(from_day)+1
    end   = int(to_year), int(to_month), int(to_day)

    getAll(account.lower(), start, end)

except:
    
    from_year  = input('from year : (4 digits) | ')
    print('### Year: ',from_year)
    from_month = input('from month:  | ')
    print('### Month: ',from_month)
    from_day   = input('from day  :  | ')
    print('### Day: ',from_day)
    to_year  = input('to year : (4 digits) | ')
    print('### Year: ',to_year)
    to_month = input('to month:  | ')
    print('### Month: ',to_month)
    to_day   = input('to day  :  | ')
    print('### Day: ',to_day)


    start = int(from_year), int(from_month), int(from_day)+1
    end   = int(to_year), int(to_month), int(to_day)

    getAll(account.lower(), start, end)