# Predicting Algae Bloom's

This is repo associated with user `apwheele` in the [DrivenData Algae Bloom prediction competition](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/649/).

## Python Environment

To set up the python environment, I use Anaconda. In particular here is my initial set up (I have a difficult time with geopandas, so make them go first):

    conda create --name bloom python=3.9 pip geopandas
    conda activate bloom
    pip install -r requirements.txt

Where `requirements.txt` has the necessary libraries for data download/manipulation and statistical modeling.

I saved the final built versions via `pip list > final_env.txt`, which is also uploaded to the repo.

## Downloading Data

I have saved the final files I used in the competition as CSV files in the `./data` folder. These include:

 - `elevation_dem.csv`, data obtained from Planetary computers DEM source, see `get_dem.py`
 - ???????
 - ???????

Note that for each of these scripts, they involved downloading (and caching) the data in a local sqllite database. On my personal machine they could take a very long time, and if your internet goes out could result in errors. The scripts are written so you can just "rerun" them again, and it will attempt to fill in the missing information and add in more data. E.g., if you are in the root of the project, you can run:

    python get_dem.py

See the output, and then if some data is missing, rerun the exact same script:

    python get_dem.py

To attempt to download more data. In the end I signed up for the Planetary Computer Hub, running the scripts on their machines went a bit faster than on my local machine.

## Running Models

ToDo, notes on building selection model weights and then final model.