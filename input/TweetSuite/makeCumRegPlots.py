import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

dire = os.path.abspath('accounts')
dires = [x for x in os.listdir(dire) if x.endswith('.pkl')]
def checkGram():
    
    for account in dires:
        if not os.path.exists(os.path.join(os.path.abspath('plots'),'{0}_cumreg.png'.format(account[:-17]))):
            print(account)
            data = joblib.load(os.path.join(dire,account))
            print('#######################')
            print(type(data.index[0]))
            print('#######################')
            ind = data.index
            cum = []
            counter = 0
            for i in ind:
                cum.append(counter)
                counter +=1
            plt.close('all')
            plt.scatter(ind,cum,color='black')
            plt.title(account[:-17])
            plt.xticks(orientation = 'vertical')
            plt.savefig(os.path.join(os.path.abspath('plots'),'{0}_cumreg.png'.format(account[:-17])))
        else: continue

checkGram()
