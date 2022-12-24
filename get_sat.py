

from datetime import datetime, timedelta
import pandas as pd
import cv2
import odc.stac

# Establish a connection to the STAC API
import planetary_computer as pc # sign up for API key
from pystac_client import Client
import geopy.distance as distance

import rioxarray
#from IPython.display import Image
from PIL import Image as PILImage

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)

# get our date range to search, and format correctly for query
def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"
    return date_range

# "sentinel-2-l2a", "landsat-c2-l2"

# Function to search for imagery data
def cat_search(bbox,date,time_buffer_days=10,collections=["sentinel-2-l2a"]):
    dr = get_date_range(date,time_buffer_days)
    # limits cloud cover to images less than 10%
    search = catalog.search(collections=collections,
                            bbox=bbox,
                            datetime=dr,
                            query={"eo:cloud_cover": {"lt": 10}})
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
    return items, item_details



bbox = [-111.26063646639783,
 41.52988747516146,
 -110.05404353360218,
 42.43019710235757]

date = '2021-09-27'

res, res_df = cat_search(bbox,date)

ls_res, ls_df = cat_search(bbox,date,collections=["landsat-c2-l2"])


# Get most recent image + image 6 months prior
# green current, veg current, ?blue current?
# green old, veg old
# dif green/veg 
# Maybe new meta-data2vec reduction
def get_image():
    # This will need to cache the image files somewhere
    return None

# https://planetarycomputer.microsoft.com/catalog


# DEM data
# https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30#Example-Notebook



# Weather https://planetarycomputer.microsoft.com/dataset/daymet-daily-na

# Vegitation https://planetarycomputer.microsoft.com/dataset/modis-13Q1-061#Example-Notebook [2000 to present]
#def get_veg(bbox,date):
#    dr = get_date_range(date,15)
#    search = catalog.search(
#        collections=["modis-13Q1-061"],
#        bbox=bbox,
#        datetime=dr,
#    )
#    items = [item for item in search.get_all_items()]
#    return items
#
#res_veg = get_veg(bbox,date)
#
#data = odc.stac.load(
#    res_veg,
#    bands="250m_16_days_EVI",
#    resolution=250,
#    bbox=bbox,
#)