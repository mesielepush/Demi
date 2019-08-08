import joblib
import os

current_path = os.getcwd()
accounts_dir = os.path.join(current_path.split("scripts",1)[0],'input','accounts')
accounts     = sorted([x[:-11] for x in os.listdir(accounts_dir)])
try:
    model_dict = joblib.load(os.path.join(current_path,'constants','model_dict.pkl'))
except:
    print('creating fucking dict')
    model_dict = {}
    model_dict['reverse']={}
    model_dict['to_mod'] ={}
    print('Im about to')
    joblib.dump(model_dict, os.path.join(current_path,'constants', 'model_dict.pkl'))
    print('I just did')
from models_and_predictions import Mode, Preds

def set_parties():

    print('##################  CREATING PARTY')
    naming_party = True
    while naming_party is True:
        party_name = input('#### PARTY NAME? ')
        is_party_name_ok = input(f'#### Is {party_name} ok? n to try again, enter for yes ')
        if is_party_name_ok != 'n':
            naming_party = False    
    print('################### available accounts:')
    initial_set = sorted(list(set([account[0] for account in accounts])))
    for letter in initial_set:
        print('########')
        print(f'######## {letter.upper()}:')
        print('########')
        begins_with = [account for account in accounts if account.startswith(letter)]
        for account in begins_with:
            print(f'#### {account}')
    party_members = []
    getting = True
    while getting:
        member = input('member: ')
        if member not in accounts:
            print(f'### {member} is not a correct account')
        else:
            is_ok = input(f'### is {member} ok? n for try again, enter for yes: ')
            if is_ok != 'n':
                party_members.append(member)
                more = input(f'### end to close party, enter to next account: ')
                if more == 'end':
                    getting = False
    parties[party_name] = [party_members]
    joblib.dump(parties, os.path.join(current_path,'constants','parties_dict.pkl'))
    print('########')
    print(f'######## created {party_name.upper()} :')
    print('########')
    for member in party_members:
        print(f'#### {member}')


try:
    model_dict = joblib.load(os.path.join(current_path,'constants','model_dict.pkl'))
except:
    print('creating fucking dict')
    model_dict = {}
    model_dict['reverse']={}
    model_dict['to_mod'] ={}
    print('Im about to')
    joblib.dump(model_dict, os.path.join(current_path,'constants','model_dict.pkl'))
    print('I just did')
try: 
    parties = joblib.load(os.path.join(current_path,'constants','parties_dict.pkl'))
except:
    parties = {}

modes = [['lopezobrador_','joseameadek'],['lopezobrador_','joseameadek','ricardoanayac'],['lopezobrador_','ricardoanayac'],['ricardoanayac', 'joseameadek']]
mods = []
for i in modes:
    mod = ''.join([x for x in sorted(i)])
    mods.append(mod)

#general production line
#Mode(['ricardoanayac', 'joseameadek']).all_models()
#Preds.make_predictions(accounts,[['joseameadek','ricardoanayac']])


pan = ['accionnacional','damianzepeda','diputadospan','ernestocordero','felipecalderon','larioshector',
        'markocortes','mzavalagc','vicentefoxque']
pri = ['arturozamora','carlosaceves_','enriqueochoar','epn','gppridiputados','hernandezderas','juarezcisneros',
        'lvidegaray','osoriochong','pri_nacional','ruizmassieu']
morena = ['batiz_bernardo','beatrizgmuller','berthalujanu','claudiashein','delfinagomeza','diputadosmorena',
        'froylanyescas','jesusrcuevas','jimenezespriu','m_ebrard','marco_medinap','mario_delgado','martibatres',
        'mexteki','ricardomonreala','rosariopiedraib','taibo2','tatclouthier','yeidckol']

Preds.process_party_results(pan,mods,'PAN')
Preds.process_party_results(morena,mods,'MORENA')
Preds.process_party_results(pri,mods,'PRI')


#set_parties()

#makeOne('milenio', to_make)

#try:
#
#    playsound(r'C:\Users\Popotito\Desktop\init.mp3')
#    Preds.process_party_results(parties['morena'],to_make,'morena')
#    playsound(r'C:\Users\Popotito\Desktop\end_fine.mp3')
#    Preds.process_party_results(parties['pri'],to_make,'pri')
#    playsound(r'C:\Users\Popotito\Desktop\end_fine.mp3')
#    Preds.process_party_results(parties['pan'],to_make,'pan')
#except:
#    playsound(r'C:\Users\Popotito\Desktop\fatal_error.mp3')
#

#Mode(['joseameadek','ricardoanayac']).train_lstm()





#

#Preds.process_party_results(parties['morena'],['lopezobrador_ricardoanayac'],'morena')






#party = 'pri'

#
##Mode.make_models(to_make)
#model_dict =joblib.load(os.path.join(model_dir,'model_dict.pkl'))
##
#for i in model_dict.keys():
#    print(i)
#    for h in model_dict[i].keys():
#        print(h)
#print(model_dict.keys())
#Preds.process_party_results(parties[party], [model_dict['to_mod'][str(sorted(model))] for model in to_make],party)




#Mode(['joseameadek','lopezobrador_']).all_models()
#Preds.make_predictions(parties['pri'],[['joseameadek','lopezobrador_']])
#for i in parties['pri']:
#    Preds.process_pred(i,['joseameadek','lopezobrador_'])
#for i in parties['pri']:
#    Preds(i,['epn','ricardoanayac']).get_ml_preds()


#Preds.make_predictions(accounts,[['accionnacional','diputadosmorena','joseameadek']])
#Preds.get_cum_record(parties['morena'], reverse_mod.keys(),'morena')
#Preds.get_cum_record(parties['pan'], reverse_mod.keys(),'pan')
#Preds.get_cum_record(parties['pri'], reverse_mod.keys(),'pri')

#print(accounts)
#os.system('shutdown /s /f')
#Mode.make_models([['pri_nacional','mario_delgado'],
#                    ['pri_nacional','accionnacional']])
#os.system('shutdown /p /f')

