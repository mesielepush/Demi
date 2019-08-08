import pandas as pd
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

color_names =['b','r','magenta','orange','grey']
current_path = os.getcwd()
predict_dir  = os.path.join(current_path.split("scripts",1)[0],'predictions')
plot_results_dir = os.path.join(current_path.split("scripts",1)[0],'plots')

def getSumProp(prop):
    """
        This sums the proportion of every column
        and returns a dict with just the total proportion
        for each column
    """
    final_proportions = {}
    total = sum([prop[account].sum() for account in prop.keys()])
    for account in prop.keys():
        final_proportions[account] = prop[account].sum()/total
    return final_proportions

def plotResults(party: 'party name',
                target: 'target name',
                result: 'party_name_results.pkl'):
    """
        Takes the processed prediction result from a party
    """
    party_results = {}
    
    print(f'#####################  Making plots for : {party}')
    for account in result.keys():
        print(f'##################### {account}')
        party_results[account] = {}
        models  = [model for model in result[account].keys()]
        
        for model in models:
            if not os.path.exists(os.path.join(plot_results_dir,party,account,model)):
                os.makedirs(os.path.join(plot_results_dir,party,account,model))
            if not os.path.exists(os.path.join(plot_results_dir,party,'totales', account)):
                os.makedirs(os.path.join(plot_results_dir,party,'totales',account))
            
            cum = result[account][model]['cum']
            prop = result[account][model]['prop']
            
            plt.close('all')
            for num, data in enumerate(prop.keys()):
                if data == target:
                    sns.kdeplot(prop[data], color = 'g', shade = True)
                else: sns.kdeplot(prop[data],color = color_names[num])

            plt.title(account)
            plt.xlim(0.5,1.0)
            plt.ylim(0,5)
            plt.xlabel('proportion probability',fontsize=13)
            plt.savefig(os.path.join(plot_results_dir,party,account,model,'proportions.png'))

            plt.close('all')
            for num, data in enumerate(cum.keys()):
                if data == target:
                    plt.scatter(prop.index, cum[data], color = 'g', label = data)
                else: plt.scatter(prop.index, cum[data], label = data, color = color_names[num])
            
            plt.title(account,fontsize=13)
            plt.xlabel('cummulative record',fontsize=13)
            plt.legend()
            plt.savefig(os.path.join(plot_results_dir,party,account,model,'cummulative_record.png'))
            
            plt.close('all')
            for num, data in enumerate(cum.keys()):
                if data == target:
                    sns.kdeplot(cum[data], color = 'g', shade = True)
                else: sns.kdeplot(cum[data], color = color_names[num])
            
            plt.title(account,fontsize=13)
            plt.xlabel('density cummulative record',fontsize=13)
            plt.savefig(os.path.join(plot_results_dir,party,account,model,'density_cummulative_record.png'))
            
            party_results[account][model] = getSumProp(prop)
            plt.close('all')
            
            for num, data in enumerate(party_results[account][model].keys()):
                if data == target:
                    plt.bar(num, party_results[account][model][data], label = data, color = 'g')
                else:
                    plt.bar(num, party_results[account][model][data], label = data, color = color_names[num])
                plt.legend()
                plt.xticks([])

            plt.title(account + ' proportion in ' + model)
            plt.xticks([])
            plt.savefig(os.path.join(plot_results_dir, party, 'totales', account, f'{account}_{model}_total_proportion.png'))

            print(f'############## Finished Plots for {model}')

morena = joblib.load(r'C:\Users\Popotito\Desktop\DEMI5.0\results\MORENA\MORENA_results.pkl')
pan = joblib.load(r'C:\Users\Popotito\Desktop\DEMI5.0\results\PAN\PAN_results.pkl')
pri = joblib.load(r'C:\Users\Popotito\Desktop\DEMI5.0\results\PRI\PRI_results.pkl')

plotResults('morena','lopezobrador_',morena)
plotResults('pan','lopezobrador_',pan)
plotResults('pri','lopezobrador_',pri)

