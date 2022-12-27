'''
Hyptertuning models
'''

from src import feat, mod
import optuna

res_results = {}

##############################################
# Setting up data and variable sets

train_dat = feat.get_data(split_pred=True)

# Model fit with weights
weight_cats = (True,False)

# Model fit with categorical variables
cat_cats = (True,False)

# Lat/Lon included in model
xy_cats = {'no': [],
           'both': ['latitude','longitude'],
           'lat': ['latitude'],
           'lon': ['longitude']}

xy_keys = tuple(xy_cats.keys())

# Region variables
region_cats = {'both': ['region','cluster'],
               'reg':  ['region'],
               'clust':  ['cluster']}

reg_keys = tuple(region_cats.keys())

# Elevation Variables
ele_cats = {'max_dif':['maxe','dife'],
            'all_var':['maxe','dife','elevation','stde'],
            'ele_std':['elevation','stde'],
            'ele_dif':['elevation','dife'],
            'max_std':['maxe','stde'] }

ele_keys = tuple(ele_cats.keys())

##############################################


##############################################
# LightBoost hyperparameter

def objective_lgb(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 600, 10),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "ele_set": trial.suggest_categorical("ele_vars", ele_keys),
        "xy_set": trial.suggest_categorical("xy_set", xy_keys),
        "reg_set": trial.suggest_categorical("reg_set", reg_keys),
        "weight": trial.suggest_categorical("weight", weight_cats),
        "cat_type": trial.suggest_categorical("cat_type", cat_cats)
    }
    # Setting the different variables
    ov = region_cats[param['reg_set']]
    cv = ele_cats[param['ele_set']] + xy_cats[param['xy_set']]
    rm = mod.RegMod(ord_vars=ov,
                    dum_vars=None,
                    dat_vars=['date'],
                    ide_vars=cv,
                    weight = 'split_pred',
                    y='severity',
                    mod = mod.LGBMRegressor(n_estimators=round(param['n_estimators']),
                                            max_depth=param['max_depth']))
    avg_rmse = rm.met_eval(train_dat,ret=True,weight=param['weight'],cat=param['cat_type'])
    return avg_rmse

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=150)
trial_lgb = study_lgb.best_trial
res_results['lgb'] = trial_lgb

#print(f"Best Average RMSE LightBoost {trial_lgb.value}")
# Best Average RMSE LightBoost 0.7550269937373841
#print("Best Params")
#print(trial_lgb.params)
# {'n_estimators': 380, 'max_depth': 5, 'ele_vars': 'ele_std', 'xy_set': 'both', 'reg_set': 'reg', 'weight': True, 'cat_type': True}

##############################################


##############################################
# XGBoost hyperparameter

def objective_xgb(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 600, 10),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "ele_set": trial.suggest_categorical("ele_vars", ele_keys),
        "xy_set": trial.suggest_categorical("xy_set", xy_keys),
        "reg_set": trial.suggest_categorical("reg_set", reg_keys),
        "weight": trial.suggest_categorical("weight", weight_cats)
    }
    # Setting the different variables
    ov = region_cats[param['reg_set']]
    cv = ele_cats[param['ele_set']] + xy_cats[param['xy_set']]
    rm = mod.RegMod(ord_vars=ov,
                    dum_vars=None,
                    dat_vars=['date'],
                    ide_vars=cv,
                    weight = 'split_pred',
                    y='severity',
                    mod = mod.XGBRegressor(n_estimators=round(param['n_estimators']),
                                           max_depth=param['max_depth'])
                )
    avg_rmse = rm.met_eval(train_dat,ret=True,weight=param['weight'])
    return avg_rmse

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=100)
trial_xgb = study_xgb.best_trial
res_results['xgb'] = trial_xgb

#print(f"Best Average RMSE XGBoost {trial_xgb.value}")
# Best Average RMSE XGBoost 0.7703229517195069
#print("Best Params")
#print(trial_xgb.params)
# {'n_estimators': 220, 'max_depth': 9, 'ele_vars': 'ele_std', 'xy_set': 'both', 'reg_set': 'both', 'weight': True}

##############################################

##############################################
# CatBoost hyperparameter

def objective_cat(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 600, 10),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "ele_set": trial.suggest_categorical("ele_vars", ele_keys),
        "xy_set": trial.suggest_categorical("xy_set", xy_keys),
        "reg_set": trial.suggest_categorical("reg_set", reg_keys),
        "weight": trial.suggest_categorical("weight", weight_cats),
        "cat_type": trial.suggest_categorical("cat_type", cat_cats)
    }
    # Setting the different variables
    ov = region_cats[param['reg_set']]
    cv = ele_cats[param['ele_set']] + xy_cats[param['xy_set']]
    rm = mod.RegMod(ord_vars=ov,
                    dum_vars=None,
                    dat_vars=['date'],
                    ide_vars=cv,
                    weight = 'split_pred',
                    y='severity',
                    mod = mod.CatBoostRegressor(iterations=round(param['n_estimators']),
                                                depth=param['max_depth'],
                                                allow_writing_files=False,
                                                verbose=False)
                )
    avg_rmse = rm.met_eval(train_dat,ret=True,weight=param['weight'],cat=param['cat_type'])
    return avg_rmse

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=150)
trial_cat = study_cat.best_trial
res_results['cat'] = trial_cat

#print(f"Best Average RMSE CatBoost {trial_cat.value}")
#Best Average RMSE CatBoost 0.7526218612441193
#print("Best Params")
#print(trial_cat.params)
# {'n_estimators': 550, 'max_depth': 9, 'ele_vars': 'ele_std', 'xy_set': 'both', 'reg_set': 'reg', 'weight': False, 'cat_type': False}


##############################################


# Printing Results
print('\n\nTRIAL RESULTS\n\n')

for m,t in res_results.items():
    print(f"Best Average RMSE {m} {t.value}")
    print("Best Params")
    print(t.params)
