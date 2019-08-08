import os
import nn
from general_ml import model_suite
from nn.mlp import train as mlp
from nn.sepCnn.train_sepcnn import train_sequence_model as sepcnn
from nn.simple_lstm.train_lstm import lstm
from load_data import *
from predict import *

current_path = os.getcwd()
accounts_dir = os.path.join(current_path.split("scripts",1)[0],'input','accounts')
accounts     = sorted([x[:-11] for x in os.listdir(accounts_dir)])
pre_pred     = os.path.join(current_path.split("scripts",1)[0],'predictions')
model_dir    = os.path.join(current_path.split("scripts",1)[0],'models')
model_dict = joblib.load(os.path.join(current_path,'constants','model_dict.pkl'))

class Mode:
    """ 
        This class creates trains the models, and saves their parameters.
    """
    
    
    def __init__(self, mode):
        
        self.mode  = sorted(mode)
        self.mod   = ''.join([letter for letter in self.mode])
        self.data  = getData(self.mode, self.mod)
        self.tfidf = tfidfvec(self.data, self.mod)
        self.seqvec= seq2int(self.data, self.mod)
    
        model_dict['reverse'][self.mod]=self.mode
        model_dict['to_mod'][str(self.mode)]=self.mod
        joblib.dump(model_dict, os.path.join(current_path,'constants','model_dict.pkl'))


    def train_logReg(self):
        kwargs = {
                'C' : 2.5,
                'tol' : 0.0001,
                'dual' : False,
                'n_jobs' : 6,
                'solver' : 'lbfgs',
                'verbose' : 2,
                'penalty' : 'l2',
                'max_iter' : 500,
                'warm_start' :  False,
                'class_weight' : 'balanced',
                'random_state' : 666,
                'fit_intercept' : False,
                'intercept_scaling' : 1.0
                }
        if len(self.mode) > 2:
            kwargs['multi_class'] = 'multinomial'
        else:
            kwargs['multi_class'] = 'ovr'

        model_suite.logReg(self.tfidf, self.mod, kwargs)

    def train_randomForest(self):
        kwargs = {
                'verbose'   : 2,
                'criterion' : 'entropy',
                'max_depth' : None,
                'bootstrap' : False,
                'random_state' : 666,
                'n_estimators' : 600,
                'max_features' : 'log2',
                'class_weight' : 'balanced',
                'min_samples_split' : 2,
                'min_samples_leaf' : 4,
                'max_leaf_nodes' : 500
                }
        model_suite.randomForest(self.tfidf, self.mod, kwargs)

    def train_sgd(self):
        kwargs = {
            'loss' : 'modified_huber',
            'eta0' : 0.0,
            'alpha' : 0.00001,
            'n_jobs' : 6,
            'average' : False,
            'epsilon' : 0.0001,
            'penalty' : 'l2',
            'max_iter' : 400,
            'l1_ratio' : 0.15,
            'warm_start': False,
            'random_state' : 666,
            'learning_rate' : 'optimal',
            'fit_intercept': False,
            'verbose': 0
            }
        model_suite.sgd(self.tfidf, self.mod, kwargs)

    def train_svc(self):
        kwargs = {
                'C' : 1.0,
                'tol' : 0.001,
                'coef0' : 0.0,
                'gamma' : 'scale',
                'kernel' : 'sigmoid',
                'degree' : 2,
                'verbose' : 2,
                'max_iter' : -1,
                'shrinking' : False,
                'probability' : False,
                'cache_size' : 2000,
                'class_weight' : 'balanced',
                'decision_function_shape' : 'ovr',
                'random_state' : 666
                }
        model_suite.svc(self.tfidf, self.mod, kwargs)

    def train_svm(self):
        kwargs = {
            'loss' :'hinge',
            'C' : 0.8,
            'dual' : True,
            'tol' : 0.0001,
            'penalty' : 'l2',
            'multi_class' : 'ovr',
            'fit_intercept' : False,
            'intercept_scaling' : 1,
            'class_weight' : 'balanced',
            'random_state' : 666,
            'max_iter' : 400,
            'verbose' : 1
            }
        model_suite.svm(self.tfidf, self.mod, kwargs)
    
    def train_mlp(self):
        kwargs = {
                'epochs' : 2000,
                'patience' : 10,
                'num_classes' : len(self.mode),
                'learning_rate' : 1e-2,
                'batch_size': 1024,
                'layers' : 2,
                'units' : 16,
                'dropout_rate' : 0.3
        }
        mlp.train_mlp_model(self.tfidf, self.mod, kwargs)
    
    def train_sepcnn(self):
        kwargs = {
            'patience' : 50,
            'learning_rate' : 1e-3,
            'epochs' : 1000,
            'batch_size' : 1024,
            'blocks' : 2,
            'filters' : 8,
            'dropout_rate' : 0.5,
            'embedding_dim' : 32,
            'kernel_size' : 3,
            'pool_size' : 3,
        }
        sepcnn(self.seqvec, len(self.mode), self.mod, **kwargs)

    def train_lstm(self):
        kwargs = {
            
                'embedding_dim' : 64,
                'dropout_rate' : 0.8
                ,
                'learning_rate' : 1e-3,
                'patience' : 50,
                'epochs' : 1000,
                'batch_size' : 256
                }
        lstm(self.seqvec,self.mod, len(self.mode),**kwargs)

    def all_models(self):
        self.train_logReg()
        self.train_randomForest()
        self.train_sgd()
        self.train_svc()
        self.train_svm()
        self.train_mlp()
        self.train_sepcnn()
        self.train_lstm()

    @classmethod
    def make_models(cls,mods):
        print(mods)
        for mode in mods:
            cls(mode).all_models()

class Preds:

    """ 
        This class manage the predictions, using the existing models and testing accounts.
    """

    def __init__(self, account, mode):

        self.account = account.lower()
        self.mode    = sorted(mode)
        self.mod     = ''.join(letter for letter in self.mode)
        self.tfidf   = load_tfidf_data(self.account, self.mod)
        self.seq2vec = load_seq2vec_data(self.account, self.mod)
    
    def get_ml_preds(self):
        predict_ml(self.account, self.tfidf, self.mod)

    def get_mlp_preds(self):
        predict_mlp(self.account, self.tfidf, self.mod)

    def get_sepcnn_preds(self):
        predict_sepcnn(self.account, self.seq2vec, self.mod)
        
    def get_lstm_preds(self):
        predict_lstm(self.account, self.seq2vec, self.mod)

    def predict_all(self):
        self.get_ml_preds()
        self.get_mlp_preds()
        self.get_sepcnn_preds()
        self.get_lstm_preds()
    
    @classmethod
    def make_predictions(cls, accounts,mods):
        """ 
            Makes several predictions at once
            takes a list  of accounts an a list of lists with the model_account_names.
            Do not return anything
        """
        for account in accounts:
            for mode in mods:
                cls(account,mode).predict_all()

    @staticmethod
    def process_pred(account,model):
        """ 
            This takes de predictions from some model
            and perform three different transformations.
            returns:
            raw: column are the algorithms and rows are the class predicted
            cum: columns are the different classes and rows are the cummulative record for each class
            porp: columns are the different classes and rows are the proportion for each class at that row
        """
        
        data = {}
        final = {}
        for machine in os.listdir(os.path.join(pre_pred,account,model)):
            # Machines endswith '.pkl' so lets clean that
            machine = machine[:-4]
            # Load the predictions into the data dict 
            data[machine] = joblib.load(os.path.join(pre_pred,account,model, f'{machine}.pkl'))
            print('data: ###############')
            print('account : ', account)
            print('model : ', model)
            print('machine : ', machine)
            print('type   :', type(data[machine]))
            
        
        dates = joblib.load(os.path.join(accounts_dir, account + '_tweets.pkl')).index
        data = pd.DataFrame(data)
        data['date'] = dates
        data = data.set_index('date')
        data = data.sort_values('date')
        final['raw'] = data
        print(f'Raw results ready for : {account}')
        

        unit_counts = {}
        data_fin = {}
        proportions = {}
        # We are going to create a new data frame with the names of the classes as columns
        for num, unit in enumerate(model_dict['reverse'][model]):
            # 'unit' is for the name of each class

            # proportion will get the proportion of votes for row
            proportions[unit]= []
            # this keeps track of the cummulative record
            unit_counts[num] = 0
            # dict for cummulative record
            data_fin[unit]   = []
        for row in range(len(data)):
            # total is the row data
            total = pd.Series(data.iloc[row])

            # loop trough each class name
            for num, unit in enumerate(model_dict['reverse'][model]):
                # in case the class has votes
                if num in total.value_counts():
                    # append the proportion of votes for the row
                    proportions[unit].append(total.value_counts()[num]/len(total))
                    # sum row result for the cummulative record
                    unit_counts[num] +=  total.value_counts()[num]/len(total)
                else:
                    # in case there is no votes for the class
                    # unit_counts[num] += 0 so there is no need to code it
                    proportions[unit].append(0)
                # append result for row to cummulative record dict.
                data_fin[unit].append(unit_counts[num])
        # create pd.DataFrame for cummulative record
        cum_result = pd.DataFrame(data_fin)
        
        cum_result['date'] = dates
        cum_result = cum_result.set_index('date')
        cum_result = cum_result.sort_values('date')
        # get cummulative record to final DataFrame
        final['cum'] = cum_result
        print(f'Cummulative record at: {account}')
        # create pd.DataFrame for cummulative record
        proportions = pd.DataFrame(proportions)
        proportions['date'] = dates
        # get proportions to final DataFrame
        proportions = proportions.set_index('date')
        proportions = proportions.sort_values('date')
        final['prop'] = proportions
        
        print(f'Finished process preds from: {account} in {model}')
        return final['raw'], final['cum'], final['prop']

    @staticmethod
    def process_party_results(accounts: 'list of party accounts',
                                mods: 'list of concated model names',
                                party: 'name for party'):
        """ 
            This takes a list of account predictions, a list of concatenated model names
            an a name for the group (party)
            It performs the process to make the three transformation mention on
            the process_pred() function
            Each account should have predictions made with the associated model
        """
        data_model = {}
        for account in accounts:
            print(account)
            print(data_model.keys())
            data_model[account] = {}
            print('account: ',account)
            for mod in mods:
                print(f'#############  Processing {account} results in {mod} model')
                data_model[account][mod] = {}
                raw_results, cum_result, proportions = Preds.process_pred(account,mod)
                data_model[account][mod]['raw'] = raw_results
                data_model[account][mod]['cum'] = cum_result
                data_model[account][mod]['prop']= proportions
                print(f'#############  Finished {account} results in {mod} model')
        if not os.path.exists(os.path.join(current_path.split("scripts",1)[0],'results',party)):
            os.makedirs(os.path.join(current_path.split("scripts",1)[0],'results',party))
        joblib.dump(data_model,os.path.join(current_path.split("scripts",1)[0],'results', party, party + '_results.pkl'))
        print(f'#############  Finished Every account in every model for {party}')
        print('#############  Results in ', os.path.join(current_path.split("scripts",1)[0],'results', party, party + '_results.pkl'))

