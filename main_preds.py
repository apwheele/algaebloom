'''
Main script to fit model
and generate predictions

Andy Wheeler
'''

from src import feat, mod

today = feat.today_str()
train_dat = feat.get_data(split_pred=True)
#train, val = mod.split(train_dat)
train, val = mod.split_weight(train_dat)

###################################
# Example just predicting severity directly

#cv2 = ['latitude','longitude'] # these appear to overfit quite a bit
cv2 = ['maxe','dife','latitude','longitude'] #,'elevation','stde'] #['wpsd','pres']


#cv2 = ['meanlogDensity300']

# ord_vars=['cluster','region']

# mod = mod.XGBRegressor(n_estimators=500, max_depth=5)

#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor

# RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=20)
# CatBoostRegressor(iterations=500,depth=5,allow_writing_files=False,verbose=False)
# LGBMRegressor(iterations=500,max_depth=5)
# XGBRegressor(n_estimators=500, max_depth=5)

rm = mod.RegMod(ord_vars=['cluster','region'],
                dum_vars=None,
                dat_vars=['date'],
                ide_vars=cv2,
                y='severity',
                mod = CatBoostRegressor(iterations=500,depth=5,allow_writing_files=False,verbose=False)
                )

rm.fit(train)  #sample_weight=train['split_pred']

train['pred'] = rm.predict_int(train)
val['pred'] = rm.predict_int(val)

mod.rmse_region(train)
mod.rmse_region(val)

# Now retraining on the full dataset
rm.fit(train_dat)  #sample_weight=train_dat['split_pred']

# Now getting files for out of sample data
test = feat.get_data(data_type='test')
test['pred'] = rm.predict_int(test)

form_dat = feat.sub_format(test)
print(form_dat['severity'].value_counts())
form_dat.to_csv(f'sub_{today}.csv',index=False)

# Saving the model
mod.save_model(rm,f'mod_{today}')
###################################

####################################
## Example predicting log Density and back transforming
#
#rm = RegMod(ord_vars=['FCODE','region'],dat_vars=['date'],ide_vars=cont_vars,y='density',transform=safelog,inv_trans=np.exp)
#rm.fit(train)
#
#train['pred2'] = rm.predict(train)
#val['pred2'] = rm.predict(val)
#
#rmse_region(val,'pred2',scale=True)
#rmse_region(train,'pred2',scale=True)
#
## predicting severity directly has better results than
## predicting log severity and back transforming
####################################