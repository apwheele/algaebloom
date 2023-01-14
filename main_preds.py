'''
Main script to fit model
and generate predictions

Andy Wheeler
'''

from src import feat, mod

today = feat.today_str()
train_dat = feat.get_data(split_pred=True)

###################################
# Example just predicting severity directly

lat_lon = ['latitude','longitude'] # these appear to overfit quite a bit
cv2 = ['elevation','dife'] #'maxe','stde']
sat_vars = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500', 'prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']
#sat_vars = ['prop_lake_500', 'r_500', 'g_500', 'b_500']
cv2 += sat_vars
cv2 += lat_lon
#cv2 = ['meanlogDensity300']

# reset to missing
#train_mis = train_dat.copy()
#for v in sat_vars:
#    train_dat[v].replace({-1: mod.np.NAN}, inplace=True)

ov = ['region','cluster']

# RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=20)
# CatBoostRegressor(iterations=500,depth=5,allow_writing_files=False,verbose=False)
# LGBMRegressor(n_estimators=500,max_depth=5)
# XGBRegressor(n_estimators=500, max_depth=5)

#Best Average RMSE cat 0.7505640632697526
#Best Params
#{'n_estimators': 500, 'max_depth': 9, 'ele_vars': 'all_var', 'xy_set': 'both', 'sl_set': 'lag1000', 'reg_set': 'both', 'weight': True, 'cat_type': False, 'sat_set': 'sat500'}

# Prediction 15, 1/1/2023, LightBoost [n_estimators=400, max_depth=8, cat_vars=True, +weights, +missing data], uses month/weekday/days + region/cluster/latlon + elevation vars ('elevation','dife') + sat(1000/2500). Score 0.8190 [Weighted Valid 0.792]
# Prediction 4, 12/21/2022, XGBoost [n_estimators=100, max_depth=3], uses month/weekday/days + region/cluster. Score 0.7870 [Insample Validation 0.7903]
# Prediction 8, 12/25/2022, Catboost [n_estimators=500, max_depth=5], uses month/weekday/days + region/cluster/lat/lon + elevation vars ('maxe','dife'). Score .8152 [Weighted Valid 0.8015]
# Prediction 17, Ensemble [Catboost 12/25/22, LightBoost 1/1/23, XGBoost 12/21/22]

sat_500 = ['prop_lake_500', 'r_500', 'g_500', 'b_500']
sat_1000 = ['prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']
sat_2500 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500']
sat_1025 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500', 
           'prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']

cat = mod.RegMod(ord_vars=['region','cluster'],
                dum_vars=None,
                dat_vars=['date'],
                ide_vars=['latitude','longitude','maxe','dife'],
                weight = 'split_pred',
                y='severity',
                mod = mod.CatBoostRegressor(iterations=500,depth=5,
                   allow_writing_files=False,verbose=False)
                )
cat.fit(train_dat,weight=False,cat=False)


lig = mod.RegMod(ord_vars=['region','cluster','imtype'],
                dum_vars=None,
                dat_vars=['date'],
                ide_vars=['latitude','longitude','elevation','dife'] + sat_1025,
                weight = 'split_pred',
                y='severity',
                mod = mod.LGBMRegressor(n_estimators=400,max_depth=8)
                )
lig.fit(train_dat,weight=False,cat=True)


xgb = mod.RegMod(ord_vars=['region','cluster'],
                 dat_vars=['date'],
                 y='severity',
                 mod = mod.XGBRegressor(n_estimators=100, max_depth=3))
xgb.fit(train_dat,weight=False,cat=False)


rm = mod.EnsMod(mods={'xgb': xgb, 'cat': cat, 'lig': lig})



#rm = mod.RegMod(ord_vars=ov,
#                dum_vars=None,
#                dat_vars=['date'],
#                ide_vars=cv2,
#                weight = 'split_pred',
#                y='severity',
#                mod = mod.CatBoostRegressor(iterations=500,depth=5,
#                   allow_writing_files=False,verbose=False)
#                )

#rm.met_eval(train_dat,pr=True,ret=True,weight=True,cat=False,full_train=True)

# Now getting files for out of sample data
test = feat.get_data(data_type='test')
test['pred'] = rm.predict_int(test)

form_dat = feat.sub_format(test)
print(form_dat['severity'].value_counts())
form_dat.to_csv(f'sub_{today}.csv',index=False)

# Saving the model
mod.save_model(rm,f'mod_{today}')
###################################