
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

def today_str():
    now = datetime.now()
    return now.strftime('%Y_%m_%d')


#weath_vars = ['tavg','tmin','tmax','prcp','wpsd','pres','tsun']

#for w in weath_vars:
#    subw = train_dat[w] == -1
#    sub_data = train_dat[~subw].copy()
#    print(sub_data[['logDensity',w]].corr())


# 'tavg','tmin','tmax','prcp','wpsd','pres','tsun'
#cont_vars = ['sqrtSQKM','latitude','longitude','tavg','tmin','tmax','prcp','wpsd','pres','tsun','elevation']
#dum_vars = ['FCODE','region']
#dat_vars = ['date']

# Modifies FCODE string in place
def mod_fcode(data,fstr='FCODE'):
    data[fstr] = data['FCODE'].astype(str).str[0:2]
    data[fstr] = data[fstr].replace({'37':'-1'})

# ordinal encoding for region
def org_reg(data,rstr='region'):
    data[rstr] = data[rstr].replace({'west':4,
                                     'midwest':3,
                                     'south':2,
                                     'northeast':1})

def safesqrt(values):
    return np.sqrt(values.clip(0))

def safelog(x):
    return np.log(x.clip(1))

def strat(values):
    edges = [np.NINF,20000,1e6,1e7,1e8,np.inf]
    labs = [1,2,3,4,5]
    res = pd.cut(values,bins=edges,labels=labs,right=False)
    return res.astype(int)

# Looking at train/test
# there are a few clusters, want
# to make sure to predict these well
def cluster(x):
    lat, lon = x[0], x[1]
    if (lat < 41) & (lon < -116):
        # cali
        return 7
    elif (lat < 41) & (lat > 36.29) & (lon < -92.9) & (lon > -102.2):
        # midwest
        return 6
    elif (lat < 38.14) & (lat > 33.26) & (lon < -74.8) & (lon > -85.52):
        # carolina
        return 2
    elif (lat < 43) & (lat > 38.7) & (lon < -75.4) & (lon > -83.55):
        # erie
        return 3
    elif (lat < 43.1) & (lat > 40.7) & (lon < -69.5) & (lon > -74.6):
        # mass
        return 4
    elif (lat < 49.6) & (lat > 41.5) & (lon < -83.55) & (lon > -104.56):
        # dakota
        return 1
    else:
        # other
        return 5

#                  1   2      2   3    3     4   4    5   5
#te_st = pd.Series([1,20000,30000,1e6,1e6+1,1e7,1e7+1,1e8,1e9])
#print(strat(te_st))

db = './data/data.sqlite'

train_query = """
SELECT 
  m.uid,
  l.region,
  l.severity,
  l.density,
  m.latitude,
  m.longitude,
  m.date,
  e.elevation,
  e.mine,
  e.maxe,
  e.dife,
  e.avge,
  e.stde,
  we.tavg,
  we.tmin,
  we.tmax,
  we.prcp,
  we.wpsd,
  we.pres,
  we.tsun,
  wa.FCODE,
  wa.SQKM,
  sl.meanlogDensity300
FROM meta AS m
LEFT JOIN elevation_dem AS e
  ON m.uid = e.uid
LEFT JOIN weather AS we
  ON m.uid = we.uid
LEFT JOIN water AS wa
  ON m.uid = wa.uid
LEFT JOIN spat_lag300 AS sl
  ON m.uid = sl.uid
LEFT JOIN labels AS l
  ON m.uid = l.uid
WHERE
  m.split = 'train'
"""

#, sp.pred AS pred_split
# LEFT JOIN split_pred AS sp
#  ON m.uid = sp.uid

test_query = """
SELECT 
  m.uid,
  l.region,
  m.latitude,
  m.longitude,
  m.date,
  e.elevation,
  e.mine,
  e.maxe,
  e.dife,
  e.avge,
  e.stde,
  we.tavg,
  we.tmin,
  we.tmax,
  we.prcp,
  we.wpsd,
  we.pres,
  we.tsun,
  wa.FCODE,
  wa.SQKM,
  sl.meanlogDensity300
FROM meta AS m
LEFT JOIN elevation_dem AS e
  ON m.uid = e.uid
LEFT JOIN weather AS we
  ON m.uid = we.uid
LEFT JOIN water AS wa
  ON m.uid = wa.uid
LEFT JOIN spat_lag300 AS sl
  ON m.uid = sl.uid
LEFT JOIN format AS l
  ON m.uid = l.uid
WHERE
  m.split = 'test'
"""

def add_table(data,tab_name,db_str=db):
    db_con = sqlite3.connect(db_str)
    dn = data.copy()
    dn['DateTime'] = pd.to_datetime('now',utc=True)
    dn.to_sql(tab_name,index=False,if_exists='replace',con=db_con)


def get_both(db_str=db,split_pred=False):
    r1 = get_data('train',db_str,split_pred)
    r1['test'] = 0
    r1.drop(columns=['severity','density','logDensity'],inplace=True)
    r2 = get_data('test',db_str,split_pred)
    r2['test'] = 1
    res_df = pd.concat([r1,r2],axis=0)
    return res_df.reset_index(drop=True)

def get_data(data_type='train',db_str=db,split_pred=False):
    db_con = sqlite3.connect(db_str)
    if data_type == 'train':
        sql = train_query
    elif data_type == 'test':
        sql = test_query
    dat = pd.read_sql(sql,con=db_con)
    mod_fcode(dat) # FCODE recode
    org_reg(dat) # Region ordinal encode
    dat['sqrtSQKM'] = safesqrt(dat['SQKM'])
    dat['cluster'] = dat[['latitude','longitude']].apply(cluster,axis=1)
    if data_type == 'train':
        dat['logDensity'] = safelog(dat['density'])
    if split_pred:
        pred_test = pd.read_sql('SELECT uid, pred AS split_pred FROM split_pred',con=db_con)
        dat = dat.merge(pred_test,on='uid')
    return dat


# Need logic to take predictions and get them in the right order
def sub_format(data,pred='pred'):
    form = pd.read_csv('./data/submission_format.csv')
    # some logic to transform predictions via Duan
    # smearing
    dp = data[[pred,'uid']].copy()
    dp[pred] = dp[pred].round().astype(int).clip(1,5)
    mf = form.merge(dp,on='uid')
    mf['severity'] = mf['pred']
    return mf[['uid','region','severity']]