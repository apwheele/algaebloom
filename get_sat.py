'''
Getting satellite imagery
'''

from src import get_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


from datetime import datetime, timedelta
import pandas as pd
#import cv2
import odc.stac

# Establish a connection to the STAC API
import planetary_computer as pc # sign up for API key
from pystac_client import Client
import geopy.distance as distance

import rioxarray
from PIL import Image as PILImage

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)


# "sentinel-2-l2a", "landsat-c2-l2"

# Function to search for imagery data
# change to search for lat/lon, then query out items inside of bbox
def cat_search(lat,lon,date,time_buffer_days=30,collections=["sentinel-2-l2a","landsat-c2-l2"],meter_buffer=1000):
    dr = get_data.get_date_range(date,time_buffer_days)
    bbox = get_data.get_bounding_box(lat,lon,meter_buffer=meter_buffer)
    # limits cloud cover to images less than 5%
    search = catalog.search(collections=collections,
                            bbox=bbox,
                            datetime=dr,
                            query={"eo:cloud_cover": {"lt": 5}})
    items = [item for item in search.get_all_items()]
    item_details = pd.DataFrame(
    [{
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "platform": item.properties["platform"],
            "min_long": item.bbox[0],
            "max_long": item.bbox[2],
            "min_lat": item.bbox[1],
            "max_lat": item.bbox[3],
            "bbox": item.bbox,
            "item_obj": item,
     }
        for item in items
    ])
    item_details.sort_values(by='datetime',inplace=True,ascending=False)
    item_details['lat'] = lat
    item_details['lon'] = lon
    return item_details


# If Sentinel available, use that, if not use LandSat

# get_data.crop_sentinel_image
# get_data.crop_landsat_image

def get_image(data, meter_buffer=1000):
    sent = data['platform'] == 'Sentinel-2A'
    lat = data['lat'][0]
    lon = data['lon'][0]
    bbox = get_data.get_bounding_box(lat,lon,meter_buffer=meter_buffer)
    # If sentinel available, use that
    if sent.sum() > 0:
        #print('Grabbing Sentinel image')
        sd = data[sent]
        item = sd['item_obj'].tolist()[0]
        cf = get_data.crop_sentinel_image(item,bbox)
        im_type = 'sentinel'
    else:
        #print('Grabbing LandSat image')
        sd = data[data['platform'] == 'landsat-8']
        item = sd['item_obj'].tolist()[0]
        cf = get_data.crop_landsat_image(item,bbox)
        im_type = 'land_sat'
    return cf, im_type


def k_mean_image(image_np,k=2, seed=10):
    red = image_np[0].flatten()
    green = image_np[1].flatten()
    blue = image_np[2].flatten()
    im_df = pd.DataFrame(zip(red,green,blue),columns=['r','g','b'])
    # eliminate black bands in image
    black = im_df.sum(axis=1)
    im_df = im_df[black > 0].reset_index(drop=True)
    # k-means cluster
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(im_df)
    # figure out cluster that is the most blue
    lab_blue = np.argmax(kmeans.cluster_centers_[:,2])
    lake = kmeans.labels_ == lab_blue
    # calculate proportion of image
    prop_lake = lake.mean()
    # calculate red/green/blue average inside that
    lake_im = im_df[lake].copy()
    red_lake = lake_im['r'].mean()
    green_lake = lake_im['g'].mean()
    blue_lake = lake_im['b'].mean()
    dat = {'prop_lake': prop_lake,
           'r': red_lake,
           'g': green_lake,
           'b': blue_lake}
    return dat


def get_image_data_ll(lat,lon,date,meter_buffer=[500,1000,2500]):
    # Get the images
    res_df = cat_search(lat,lon,date)
    res_di = {}
    for m in meter_buffer:
        # This could be made more efficient
        # only grab image once, then do multiple crops
        ri, im_type = get_image(res_df, meter_buffer=m)
        res_di['imtype'] = im_type
        ri_dat = k_mean_image(ri)
        for d in ri_dat.keys():
            nk = f'{d}_{str(m)}'
            res_di[nk] = ri_dat[d]
    return res_di


def loop_image_data(data,con=get_data.db_con,table_name='sat'):
    # if table exists, check to make sure data is not already included
    if get_data.tab_exists(table_name,con):
        dc = get_data.get_update(data,table_name)
        print(f'Updating {dc.shape[0]} records in {table_name}')
    else:
        dc = data.copy()
        print(f'Making {dc.shape[0]} new records in {table_name}')
    uidl = dc['uid'].tolist()
    latl = dc['latitude'].tolist()
    lonl = dc['longitude'].tolist()
    datl = pd.to_datetime(dc['date']).tolist()
    it = 0
    for uid,lat,lon,dat in zip(uidl,latl,lonl,datl):
        it += 1
        print(f'Grabbing image {it} out of {dc.shape[0]} @ {datetime.now()}')
        try:
            res_im = get_image_data_ll(lat,lon,dat)
            res_df = pd.DataFrame([res_im])
            res_df['uid'] = uid
            get_data.add_table(res_df,table_name,con)
        except Exception:
            print(f'Unable to query image data for UID: {uid}')
    res = pd.read_sql(f'SELECT * FROM {table_name}',con)
    return res

#get_data.drop_table('sat')

meta = pd.read_csv('./data/metadata.csv')
res_df = loop_image_data(meta) # meta.sample(100) # to check
print(res_df.shape)
res_df.to_csv('./data/sat_stats.csv',index=False)



