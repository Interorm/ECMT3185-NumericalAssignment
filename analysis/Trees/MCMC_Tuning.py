from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
from tqdm import tqdm

import pickle

def PrepX(X:pd.DataFrame):

        X_prep = pd.DataFrame(index=X.index)
        X_types = X.dtypes.to_dict()

        #Iterates over X and applies transformations based on column type; Adds transformed columns 
        new_cols = []
        for col, t in X_types.items():

            #Leaves numeric columns unchanged, no scaling needed for tree-based methods, scale invariant
            if t in ['int', 'float']: 
                new_cols.append(X[col].copy())
                continue

            #Create dummies for categorical (str) columns
            if t == 'object': 
                new_col = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
                new_cols.append(new_col)
                continue

            print(f'Type \"{t}\" not in standard types!')

        if new_cols: X_prep = pd.concat([X_prep] + new_cols, axis=1)

        return X_prep

def Split(X:pd.DataFrame, Y:pd.DataFrame, TestSize:float, random_state:int=42):

        split_index = np.random.RandomState(random_state) \
                            .choice([True, False], size=len(Y), p=[TestSize,1-TestSize])
        
        X_test = X[split_index]
        X_train = X[~split_index]
        Y_test = Y[split_index]
        Y_train = Y[~split_index]

        return X_test, X_train, Y_test, Y_train

#Pertubate model to randomly explore parameter space
def Pertubate(params:dict, params_info:dict, step_scaler:float) -> dict:

    new_params = {}

    for param, old_value in params.items():
        #Perbutation Type, Steps and Boundaries based on given Parameter Info
        (param_min, param_max) = params_info[param]['range']
        param_type = params_info[param]['type']
        step_size = params_info[param]['step_size'] * step_scaler

        #Change float values with normal distribution, sigma = Stepsize*Scaler
        if param_type == 'float':
            new_value = old_value + np.random.normal(0, step_size) 

        #Change int values by one with the chance of StepSize*Scaler
        if param_type == 'int':
            if np.random.uniform(0,1) <= step_size:
                new_value = old_value + np.random.choice([-1,1]) 
            else:
                new_value = old_value

        if new_value < param_min:
            new_value = param_min + (param_min - new_value)
        if new_value > param_max:
            new_value = param_max - (new_value - param_max)
        new_value = np.clip(new_value, param_min, param_max)
        
        new_params[param] = new_value

    return new_params

#Calculate Acceptance; Lower RSME = Always chosen, Higher RSME sometimes chosen with prob based on deviation
def Acceptance(old_proposal, new_proposal, temperature):
    
    accepting_probability = min(1, np.exp( (old_proposal['rsme'] - new_proposal['rsme']) / temperature ))

    accept = np.random.random() <= accepting_probability

    return accept

#Function to fit proposed models and record parameters and RSME-Score
def TestModel(X, Y, model, fixed_params:dict, test_params:dict):
    
    TestModel = model(**fixed_params, **test_params)
    rsme = np.mean( -cross_val_score(TestModel, X, Y, n_jobs=-1,
                                     scoring='neg_root_mean_squared_error', cv=KFold(3, shuffle=True)))
    
    out = test_params | {'rsme':rsme, 'params':test_params}

    return out


def MCMC_Tuning(X, Y, model,
                fixed_params:dict, test_params:dict, 
                burnin:int, chain_length:int, chain_number:int,
                starting_temperature:float, temperature_decay:float=0.998, starting_step_scaler:float=1,
                target_acceptance:float=0.3, adaption_schedule:int=10, 
                autosave:bool=False, debug:bool=False, load=None, load_Burnin=None):
    
    #Save Parameter Info and split off grid for random search
    initial_params = {param: info['range'] for param, info in test_params.items()}
    info_params = {param: {'type':info['type'],
                           'range':tuple((min(info['range']), max(info['range']))),
                           'step_size':info['step_size']} 
                   for param, info in test_params.items()}

    if load is None:
        print('Generating Burn-In Samples...')
        BaseModel = model(**fixed_params)

        #Instead of Classic Burnin, choose #burnin random parameter combinations with RandomizedSearchCV and choose best for chain starts 
        BurnIn_Samples = RandomizedSearchCV(
            estimator=BaseModel, param_distributions=initial_params, n_iter=burnin,
            scoring='neg_root_mean_squared_error', cv=KFold(shuffle=True),
            refit=False, random_state=1, n_jobs=-1, verbose=0)
        BurnIn_Samples.fit(X, Y)

        print('Choosing Chain-Starts...')
        #Choose best randomly selected Parameter Combinations to start Chains
        BurnIn_Samples = pd.DataFrame(BurnIn_Samples.cv_results_)
        BurnIn_Samples = BurnIn_Samples.sort_values(by='mean_test_score', ascending=False)\
                                    .reset_index().drop(columns=['index'])
        BurnIn_Samples = BurnIn_Samples.loc[0:chain_number-1, [col for col in BurnIn_Samples.columns if col.startswith('param_')]]
        BurnIn_Samples.columns = BurnIn_Samples.columns.str.replace('param_','')

        chain_starting_params = BurnIn_Samples.to_dict('records')

        print('Starting MCMC Algorithm...')
        chains = [None] * chain_number

        #Initialize Chains with starting parameters, best parameters from RandomSearch
        for i, chain_start in enumerate(chain_starting_params):
            chains[i] = {
                'samples':[],
                'temperature':starting_temperature,
                'step_scaler':starting_step_scaler,
                'acceptance_history':[]
            }
            chains[i]['samples'].append(TestModel(X, Y, model, fixed_params, chain_start))

        start = 1
    else:
        with open(load, 'rb') as f: chains = pickle.load(f)
        start = len( chains[0]['samples'] )


    #Heart of MCMC Algorithm; Slightly alter parameters every iteration randomly to successively explore feature space
    for iteration in tqdm(range(start,chain_length), total=chain_length-start+1,
                          desc='Sampling Parameter-Space...', unit='Iterations', position=1):

        for chain in tqdm(chains,
                          desc='Chains', unit='chain', position=2, leave=False):

            old_proposal = chain['samples'][-1]
            old_proposal_params = old_proposal['params']
            #Propose new Parameters through Pertubation and Build Model
            new_proposal_params = Pertubate( old_proposal_params, info_params, chain['step_scaler'] )
            new_proposal = TestModel(X,Y,model,fixed_params,new_proposal_params)

            #Decide if New Proposal should be accepted
            accept = Acceptance(old_proposal, new_proposal, chain['temperature'])
            chain['acceptance_history'].append(int(accept))
            if accept:
                chain['samples'].append(new_proposal)
            else:
                chain['samples'].append(old_proposal)

            #Debug to show current stats
            if debug:
                print(f'Iteration: {iteration}')
                print(f'Old Params: {old_proposal_params}')
                print(f'New Params: {new_proposal_params}')
                old_rmse = old_proposal['rsme']
                new_rmse = new_proposal['rsme']
                print(f'Old RMSE: {old_rmse}')
                print(f'New RMSE: {new_rmse}')
                print(f'Accepted: {accept}')
                print('-'*20)

            if iteration % adaption_schedule == 0 and iteration > 0:

                #Reduce Temperature and Stepsize by given Factor every adaption
                chain['temperature'] *= temperature_decay

                #Modify Step Size and Temperature based on acceptance Rate (Annealing)
                accepance_rate = np.mean(chain['acceptance_history'][-adaption_schedule:])

                if accepance_rate < target_acceptance - 0.1666:
                    chain['temperature'] *= 1.05
                    chain['step_scaler'] *= (1 / 1.05)
                if accepance_rate > target_acceptance + 0.1666:
                    chain['temperature'] *= (1 / 1.05)
                    chain['step_scaler'] *= 1.05

                #Autosave for on-th-fly changes and tests
                if autosave:
                    with open(f'models/MCMC_{model.__name__}_Autosave.sav', 'wb') as f: pickle.dump(chains, f)

    with open(f'models/MCMC_{model.__name__}.sav', 'wb') as f: pickle.dump(chains, f)

    return chains

if __name__ == '__main__':


   # Load Data and Prepare for Processesing (PrepX)
    with open('../../data/DatasetCleaned.csv', 'r') as f:
        DATA = pd.read_csv(f)

    Y_DATA = np.log(DATA['SalePrice'].copy().values)
    VAR_DEPENDENT = 'SalePrice'

    X_DATA = DATA.copy().drop(columns=['SalePrice','Unnamed: 0'])
    X_DATA = PrepX(X_DATA)
    VAR_NAMES = X_DATA.columns

    #Create Validation Set 
    VALIDATION_SET = True
    if VALIDATION_SET: 
        X_VAL, X_DATA, Y_VAL, Y_DATA = Split(X_DATA, Y_DATA, 0.05, 1)
        X_VAL = X_VAL.values.reshape(-1, X_VAL.shape[1])

    X_DATA = X_DATA.values.reshape(-1,X_DATA.shape[1])


    #Full XGBRegressor MCMC Model
    XGBBoost = False
    if XGBBoost:

        #Parameters for each Regressor
        fixed_params = {
            'objective':'reg:squarederror',
            'booster':'gbtree',
            'eval_metric':'rmse',
            'tree_method':'hist',
            'n_estimators':200,
            'verbosity':0,
            'n_jobs':-1,
            'random_state':42
        }

        #Parameter Space for BurnIn, Chains based on best Models from BurnIn, includes arguments for pertubation (type, stepsize)
        test_params = {
            'eta':               {'type':'float', 'range': np.linspace(0.001, 0.33, 10).tolist(), 'step_size':0.01},
            'max_depth':         {'type':'int',   'range': list(range(3, 20)), 'step_size':0.1},
            'min_child_weight':  {'type':'float', 'range': np.linspace(1, 100, 10).tolist(), 'step_size':3},
            'min_split_loss':    {'type':'float', 'range': np.linspace(0, 2, 5).tolist(), 'step_size':0.1},
            'subsample':         {'type':'float', 'range': np.linspace(0.5, 1.0, 5).tolist(), 'step_size':0.025},
            'colsample_bylevel': {'type':'float', 'range': np.linspace(0.3, 1.0, 5).tolist(), 'step_size':0.033},
            'colsample_bynode':  {'type':'float', 'range': np.linspace(0.3, 1.0, 5).tolist(), 'step_size':0.033},
            'reg_alpha':         {'type':'float', 'range': np.linspace(0.0, 5.0, 10).tolist(), 'step_size':0.33},
            'reg_lambda':        {'type':'float', 'range': np.linspace(0.0, 100.0, 10).tolist(), 'step_size':5},
            'max_delta_step':    {'type':'float', 'range': np.linspace(0.0, 10.0, 10).tolist(), 'step_size':0.5}
            }
        
        MCMC_XGBoost = MCMC_Tuning(X_DATA, Y_DATA, xgb.XGBRegressor, fixed_params, test_params,
                                   2000, 2500, 10, 0.025, autosave=True, load='models/MCMC_XGBoost_Autosave.sav')
    
    #Full RandomForest MCMC Model
    RandomForest = True
    if RandomForest:

        #Parameters for each Regressor
        fixed_params = {
            'n_estimators':100,


            'verbose':0,
            'n_jobs':-1,
            'random_state':42
        }

        #Parameter Space for BurnIn, Chains based on best Models from BurnIn, includes arguments for pertubation (type, stepsize)
        test_params ={
            'ccp_alpha':            {'type':'float', 'range':np.logspace(-6,-2,50), 'step_size':1e-4},
            'max_depth':            {'type':'int',   'range': list(range(3, 20)), 'step_size':0.1},
            'max_features':         {'type':'float', 'range':np.linspace(0.01,1,20), 'step_size':0.05},
            'min_samples_leaf':     {'type':'float', 'range':np.linspace(0.01,0.25,20), 'step_size':0.025},
            'min_samples_split':    {'type':'float', 'range':np.linspace(0.01,0.33,20), 'step_size':0.025},
            'min_impurity_decrease':{'type':'float', 'range':np.logspace(-7,-5,50), 'step_size':1e-6},
        }

        MCMC_XGBoost = MCMC_Tuning(X_DATA, Y_DATA, RandomForestRegressor, fixed_params, test_params,
                                   500, 2500, 10, 0.01, autosave=True, load='models/MCMC_RandomForestRegressor_Autosave.sav')
