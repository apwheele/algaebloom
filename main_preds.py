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
                mod = mod.CatBoostRegressor(iterations=380,depth=6,
                   allow_writing_files=False,verbose=False)
                )
cat.fit(train_dat,weight=False,cat=False)


lig = mod.RegMod(ord_vars=['region','cluster','imtype'],
                dum_vars=None,
                dat_vars=['date'],
                ide_vars=['latitude','longitude','elevation','dife'] + sat_1025,
                weight = 'split_pred',
                y='severity',
                mod = mod.LGBMRegressor(n_estimators=500,max_depth=8)
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


# function to check if similar to any past submissions
def check_similar(current):
    import os
    import pandas as pd
    import numpy as np
    files = os.listdir("./submissions")
    for fi in files:
        old = pd.read_csv(f"./submissions/{fi}")
        dif = np.abs(current['severity'] - old['severity']).sum()
        if dif == 0:
            print(f'Date {fi} same as current')

check_similar(form_dat)


# Saving the data and model
form_dat.to_csv(f'sub_{today}.csv',index=False)
mod.save_model(rm,f'mod_{today}')
###################################