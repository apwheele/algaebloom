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

#cv2 = ['latitude','longitude'] # these appear to overfit quite a bit
cv2 = ['latitude','longitude','elevation','stde'] #['maxe','dife']
#cv2 = ['meanlogDensity300']

ov = ['region'] #'cluster'

# RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=20)
# CatBoostRegressor(iterations=500,depth=5,allow_writing_files=False,verbose=False)
# LGBMRegressor(iterations=500,max_depth=5)
# XGBRegressor(n_estimators=500, max_depth=5)

#{'n_estimators': 550, 'max_depth': 9, 'ele_vars': 'ele_std', 'xy_set': 'both', 'reg_set': 'reg', 'weight': False, 'cat_type': False}

rm = mod.RegMod(ord_vars=ov,
                dum_vars=None,
                dat_vars=['date'],
                ide_vars=cv2,
                weight = 'split_pred',
                y='severity',
                mod = mod.CatBoostRegressor(n_estimators=550,
                                              max_depth=9,
                                              allow_writing_files=False,
                                              verbose=False)
                )



rm.met_eval(train_dat,pr=True,ret=True,weight=False,cat=False,full_train=True)

# Now getting files for out of sample data
test = feat.get_data(data_type='test')
test['pred'] = rm.predict_int(test)

form_dat = feat.sub_format(test)
print(form_dat['severity'].value_counts())
form_dat.to_csv(f'sub_{today}.csv',index=False)

# Saving the model
mod.save_model(rm,f'mod_{today}')
###################################