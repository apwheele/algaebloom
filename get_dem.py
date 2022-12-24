'''
This script generates the DEM data
run on planetary computer
have trouble running on personal
machine and hitting rate limiting
'''

from src import get_data
from src.get_data import db_con
from datetime import datetime
import pandas as pd
import pystac_client
import rioxarray
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def elevation_point(lat,lon,box=1000):
    # returns point elevation, min/max/dif/mean within box meters
    try:
        ll = [lon,lat]
        search = catalog.search(
           collections=["cop-dem-glo-30"],
           intersects={"type": "Point", "coordinates": ll},)
        items = list(search.get_items())
        if len(items) == 0:
            md = -99999
            dat = {'latitude': lat, 'longitude': lon, 'box': box, 'elevation':md, 'mine':md, 'maxe':md, 'dife':md, 'avge': md, 'stde': md}
            return dat
        signed_asset = planetary_computer.sign(items[0].assets["data"])
        ro = rioxarray.open_rasterio(signed_asset.href)
        #ro.x.values
        #ro.y.values
        #ro.values # need to flatten
        ele = ro.sel(x=lon, y=lat, method="nearest").values[0]
        bbox = get_data.get_bounding_box(lat,lon,box)
        ro_clip = ro.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        min_ele = ro_clip.values.min()
        max_ele = ro_clip.values.max()
        dif_ele = max_ele - min_ele
        avg_ele = ro_clip.values.mean()
        std_ele = ro_clip.values.std()
        dat = {'latitude': lat,
               'longitude': lon,
               'box': box,
               'elevation': ele, 
               'mine': min_ele, 
               'maxe': max_ele, 
               'dife': dif_ele, 
               'avge': avg_ele, 
               'stde': std_ele}
        ro.close()
        return dat
    except Exception:
        print(f'Query failed for {lat},{lon}')
        md = -99999
        dat = {'latitude': lat, 'longitude': lon, 'box': box, 'elevation':md, 'mine':md, 'maxe':md, 'dife':md, 'avge': md, 'stde': md}
        time.sleep(10)
        return dat

def add_elevation(data,lat='latitude',lon='longitude',con=db_con,name='elevation_dem'):
    uid = data['uid'].tolist()
    lat_l = data[lat].tolist()
    lon_l = data[lon].tolist()
    iv = 0
    tot_n = data.shape[0]
    res = []
    for u,la,lo in zip(uid,lat_l,lon_l):
        iv += 1
        #print(f'Getting {iv} out of {tot_n} @ {datetime.now()}')
        ele_dat = elevation_point(la,lo,1000)
        ele_dat['uid'] = u
        res.append(ele_dat.copy())
    res_df = pd.DataFrame(res)
    get_data.add_table(res_df,name,con)
    return res_df


def get_elevation(data,con=db_con,name='elevation_dem',chunk=100):
    # If table exists, only worry about getting new information
    if get_data.tab_exists(name,con):
        upd = get_data.get_update(data,name,con)
        if upd.shape[0] > 0:
            print(f'Updating {upd.shape[0]} records')
            upd_chunks = get_data.chunk_pd(upd,chunk)
            print(f'Chunk size is {upd_chunks[0].shape[0]}')
            for i,ud in enumerate(upd_chunks):
                print(f'Getting chunk {i+1} out of {len(upd_chunks)} @ {datetime.now()}')
                ele_data = add_elevation(data=ud,name=name)
        else:
            print('No new records to append to elevation dem table')
    else:
        print('elevation_dem table does not exist, add in stats')
        upd_chunks = get_data.chunk_pd(data,chunk)
        print(f'Chunk size is {upd_chunks[0].shape[0]}')
        for i,ud in enumerate(upd_chunks):
            print(f'Getting chunk {i+1} out of {len(upd_chunks)} @ {datetime.now()}')
            ele_data = add_elevation(data=ud,name=name)
    return None

# sometimes missing data, can delete out
if get_data.tab_exists('elevation_dem'):
    db_con.execute('DELETE FROM elevation_dem WHERE elevation = -99999')

meta = pd.read_csv('./data/metadata.csv')
get_elevation(meta)
res = pd.read_sql('SELECT * FROM elevation_dem',db_con)
res.to_csv('./data/elevation_dem.csv',index=False)