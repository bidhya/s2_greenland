#!usr/bin/env python

""" Processing of Sentinel Images for Greenland Glacier Velocity

    1. Download Data by Regions
    2. For each region, merge tiles (using MGRS grids) for the same day
    3. Project if required


    Functions:
        merge_tifs(download_folder, subset): merge all tiffs from aoi and same date_hour    
    Created : Jan 18, 2021 
    Bidhya N Yadav

    Overall strategy is to download files per each region [because sat search only works for one polygon at a time]
    Then process its
    And repeat for other regions
    save downloaded files in the downloads folder, so that for intersecting regions, the file need not be downloaded again

    Issues
    ------
    skipping clipping error where there is no/minimum overlap with AOI polygon with try/except
    Memory Error during parallel processing: 
        - joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could 
        be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.
        The exit codes of the workers are {SIGKILL(-9)}

    References
    ----------
    https://tc.copernicus.org/articles/16/2629/2022/
    Chudley, T. R., Howat, I. M., Yadav, B., & Noh, M. J. (2022). Empirical correction of systematic orthorectification error in Sentinel-2 velocity fields for Greenlandic outlet glaciers. The Cryosphere, 16(6), 2629-2642.

    NB:
    ---
    Make sure to run download for a subset of regions separately, else will take weeks for all the glaciers, even for 1 year!
"""
# Author: Bidhya N. Yadav <yadav.111@osu.edu>

import os
from pathlib import Path
import datetime
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays

from satsearch import Search
import json
from joblib import Parallel, delayed
import logging
# to extract boundary for raster foreground
from shapely.geometry import Polygon
from rasterio import features  # used to contruct raster boundary by ignoring nodata pixels
from rasterio.errors import RasterioIOError  # Raised when a dataset cannot be opened using one of the registered format drivers.
from rioxarray.exceptions import NoDataInBounds
from rasterio.enums import Resampling  # for cubic, bilinear resampling etc during reprojection or reprojection match
import matplotlib.pyplot as plt
import argparse


def download_sentinel(download_folder, geom, start_date='2021-10-01', end_date='2021-02-28', collection_name='sentinel-s2-l2a-cogs'):
    """ Download Sentinel2 COGs from AWS Cloud using satsearch api.

    Parameters
    ----------
    download_folder : Full path to download Sentinel-2 TIFFs
        Where that data will be staged by stac-api.
    geom : single polygon AOI
        Shapely geometry extracted from Geopandas DataFrame.
    start_data : starting data to search for data

    end_date : last day to search data

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.

    """
    logging.info(f'Download data: {start_date} to {end_date}')
    stac_url = 'https://earth-search.aws.element84.com/v0'
    # collection_name = 'sentinel-s2-l2a-cogs'  # sentinel-s2-l1c landsat-8-l1-c1
    search = Search.search(url=stac_url,
                        datetime=f'{start_date}/{end_date}',
                        # query=["eo:cloud_cover<50"],
                        collections=[collection_name],
                        intersects=geom
                        )
    print(f'intersects search: {search.found()} items')
    if search.found() > 0:
        items = search.items()
        # TODO: Maybe this will solve following warning: WARNING:There are more items found (11702) than the limit (10000) provided.; only for 191_Hagen_Brae so far
        # items = search.items(limit=20000)
        logging.info(f'{len(items)} items found')
        filenames = items.download('B08', filename_template=f'{download_folder}/${{year}}/${{sentinel:product_id}}')  # nir = B08 
        logging.info("Finished download assets")


def region_metadata_csv(region):
    """ Create metdata csv for each region
        Get bounding coordinates, extent, and resolution of each raster
        We will use this csv later to troubleshoot.
    Parameters
    ----------
    region : The glacier region to create metadata csv for
    Notes
    -----
        base_folder is hardcoded here
        No verified either
    """
    base_folder = '/fs/project/howat.4/sentinel2'
    # Get Clipped TIFFs for a region
    tifs = os.listdir(f'/{base_folder}/clipped/{region}') 
    tifs.sort()
    print(region, len(tifs))
    # Get some shape attributes from each raster to analyze what is going on
    coord_list = []
    tif_list = []
    shape_list = []
    res_list = []  # resolution; if all 10m or not
    for tif in tifs:
        ds = rioxarray.open_rasterio(f'{base_folder}/clipped/{region}/{tif}')
        ds = ds.sel(band=1)
        coord_list.append(ds.rio.bounds())
        shape_list.append(ds.shape)
        res_list.append(ds.rio.resolution())
        tif_list.append(tif.split('.tif')[0])
    coord_list = np.array(coord_list)
    shape_list = np.array(shape_list)
    res_list = np.array(res_list)
    coord_arr = np.hstack((coord_list, shape_list, res_list))
    # Convert to Dataframe
    # df = pd.DataFrame([shape_list, coord_list], index=tif_list)
    df = pd.DataFrame(coord_arr, index=tif_list)
    df.columns = ['x0', 'y0', 'x1', 'y1', 'sizey', 'sizex', 'resx', 'resy']
    df.index.name = 'Tile'
    df.to_csv(f'{base_folder}/analysis/csv/{region}.csv')


def get_template_tif(clip_folder, download_folder, aoi_tifs, tif_prefix, aoi, epsg_str):
    """ Merge one or more rasters and clip the merged product by AOI
        And save the clipped product.
    Parameters
    ----------
    clip_folder : path to folder for saving clipped TIFFs
    download_folder : path to folder for previously downloaded/staged TIFFs
    aoi_tifs : List of all the TIFFs within AOI
    tif_prefix : Filename substring upto the hour of TIFF acquisition
        Unique filename truncated to hour of filename, example: TODO. This was extracted from aoi_tifs
    epsg_str : EPSG code to reproject from native UTM zome to Polar Stereographic North (Greenland for example)
    Notes
    -----
    To get tif_prefix, we first extract the sub-string for aoi_tifs list;
    then convert it to set.
    The idea is to get a unique names that correpond to same satellite-day-hour.
    Merge these tiffs to get seamless coverage of Tiles for same day and same satellite (A or B)
    """
    clipped_tif = f'/{clip_folder}/{tif_prefix}.tif'
    subset = [f'{x}_B08.tif' for x in aoi_tifs if tif_prefix in x]
    try:
        merged = rioxarray.open_rasterio(f'{download_folder}/{subset[0]}')  # chunks={'x': 1200, 'y': 1200}
        # Reproject to common CRS (Polar Stereographic North for example)
        # Required for merging TIFFs that span different UTM zones
        merged = merged.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)  # or as tuple (10,10)
        if len(subset) > 1:
            # even though this for loop will be ignored when there is just 1 item in list, it is better to put this guard
            for tif in subset[1:]:
                # read data; reproject it, then merge
                tmp_ds = rioxarray.open_rasterio(f'{download_folder}/{tif}')  #, chunks={'x': 1200, 'y': 1200}
                tmp_ds = tmp_ds.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)
                merged = merge_arrays([merged, tmp_ds], res = 10)
        clipped = merged.rio.clip(aoi.geometry.apply(mapping), aoi.crs) ## A list of geojson geometry dicts.
        # Determine what fraction of glacier is covered by clipped raster
        aoi_area = aoi.Area.item()
        clipped_area = xr.where(clipped > 0, 1, 0).sum().item()*100/1e6  # assume anything above 0 is foreground
        coverage_fraction = clipped_area/aoi_area
        if coverage_fraction > 0.9:  # try to generate template from large glacier; but smaller one should work too
            # Save the Clipped TIFF only if it covers at least 30% of glacier (AOI)
            clipped.rio.to_raster(clipped_tif)
            template_tif = clipped_tif
            return template_tif
    except RasterioIOError:
        # File does not exist probably
        logging.info('RasterioIOError')
        print('RasterioIOError')
    except NoDataInBounds:
        # NoDataInBounds: No data found in bounds; for cases where overlap with aoi polygon is very small (and perhaps only with area with nodata)
        print('NoDataInBounds error ')
        logging.info('NoDataInBounds error')


def merge_and_clip_tifs(clip_folder, metadata_folder, download_folder, aoi_tifs, tif_prefix, aoi, template_tif, tile_ids, epsg_str):
    """ Merge one or more rasters and clip the merged product by AOI
        And save the clipped product.

    Parameters
    ----------
    clip_folder : path to folder for saving clipped TIFFs
    download_folder : path to folder for previously downloaded/staged TIFFs
    aoi_tifs : List of all the TIFFs within AOI
    tif_prefix : Filename substring upto the hour of TIFF acquisition
        Unique filename truncated to hour of filename, example: TODO. This was extracted from aoi_tifs
    epsg_str : EPSG code to reproject from native UTM zome to Polar Stereographic North (Greenland for example)
    Notes
    ----- 
    To get tif_prefix, we first extract the sub-string for aoi_tifs list;
    then convert it to set.
    The idea is to get a unique names that correpond to same satellite-day-hour.
    Merge these tiffs to get seamless coverage of Tiles for same day and same satellite (A or B)
    """
    clipped_tif = f'/{clip_folder}/{tif_prefix}.tif'
    metadata_file = f'/{metadata_folder}/{tif_prefix}.csv'
    if not os.path.exists(clipped_tif):
        """ No processing if clipped TIFF already exist 
            But this check is not required anymore as we update the codes to check for existing clipped tiffs
        """
        subset = [f'{x}_B08.tif' for x in aoi_tifs if tif_prefix in x]
        # Sort tiles in order of Sentinel mgrs grid coverage that was determined apriori and populated into aoi shapefile
        # with higest coverage to lowest; for regions with more than 1 mrgs_tile we corrected this manually sometimes
        order = {key: i for i, key in enumerate(tile_ids)}  # https://stackoverflow.com/questions/34308635/sort-elements-with-specific-order-in-python
        subset = sorted(subset, key=lambda x: order[x.split('_')[5][1:]])
        # check if all tiles come from same UTM zone (only 190_Academy and 122_upernavik_central has data from two UTM zones )
        utm_zones = [f.split('_')[5][1:3] for f in subset]
        utm_set = set(utm_zones)

        tif_dict = {}
        # tif_dict[tif_prefix] = ','.join(subset)  # for saving to csv join with comma
        tif_dict[tif_prefix] = subset  # for yaml use original subset list
        try:
            if len(utm_set) == 1:
                # if all tiffs are from same utm zone, use this one liner for merging
                merged = merge_arrays([rioxarray.open_rasterio(f'{download_folder}/{tif}') for tif in subset])
            elif len(utm_set) > 1:  # actually just two here
                # Older: if all tiffs come from Different utm zones, use this approach. That is, reproject to common crs before mergins
                # TODO: Mar 22, 2022: We've harded each AOI to have data from no more than 2 utm zones
                subset1 = [f for f in subset if utm_zones[0] in f.split('_')[-3]]
                if len(subset1) > 0:
                    merged = merge_arrays([rioxarray.open_rasterio(f'{download_folder}/{tif}') for tif in subset1])
                    merged = merged.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)
                    # We will only process remaining tifs if some files from subset1 is processed
                    subset2 = [f for f in subset if not f in subset1]  # process any remaining tiffs from different UTM zones
                    if len(subset2) > 0:
                        merged2 = merge_arrays([rioxarray.open_rasterio(f'{download_folder}/{tif}') for tif in subset2])
                        merged2 = merged2.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)
                        merged = merge_arrays([merged, merged2])
            dst10m = rioxarray.open_rasterio(f'{template_tif}', chunks=True)  #template raster
            clipped = merged.rio.reproject_match(dst10m)  # this should take care of size, shape, resolution, clips etc.
            # Determine what fraction of glacier is covered by clipped raster
            aoi_area = aoi.Area.item()
            clipped_area = xr.where(clipped > 0, 1, 0).sum().item()*100/1e6  # assume anything above 0 is foreground
            coverage_fraction = clipped_area/aoi_area
            if coverage_fraction > 0.5:  # we'll save tiffs that cover more than 50% of glacier
                clipped.rio.to_raster(clipped_tif)  # Save the Clipped TIFF only if it covers at least 50% of glacier AOI
                # tif_dict['coverage'] = coverage_fraction
                with open(metadata_file, 'w') as outfile:
                    for key in tif_dict.keys():
                        vals = ",".join(tif_dict[key])
                        outfile.write(f'{key},{vals}')
                # # To save as yaml file
                # with open(f'{metadata_file}.yml', 'w') as outfile:
                #     yaml.dump(tif_dict, outfile, default_flow_style=False)
                return tif_prefix, subset  # list of tiffs used to mosaic then clip; we'll use this for metadata for nsidc
        except RasterioIOError:
            # File does not exist probably
            logging.info(f'RasterioIOError for tif_prefix: {tif_prefix}')
            print(f'RasterioIOError for tif_prefix: {tif_prefix}')
        except NoDataInBounds:
            # NoDataInBounds: No data found in bounds; for cases where overlap with aoi polygon is very small (and perhaps only with area with nodata)
            print('NoDataInBounds error ')
            logging.info('NoDataInBounds error')


def concat_csv_files(base_metadata_folder, region):
    """ Merge individual csv files for each region into a master combined file
        Individual csv files are just one row corresponding to the clipped tiff
        The row consist of clipped tiff name and actual sentinel2 files that were merged prior to clipping
        The goal here is to track the constituent tiffs for each clipped file

        Run this each time a new clipped tiff is created.
    """
    metadata_folder = f'{base_metadata_folder}/individual_csv/{region}'
    csv_files = [f for f in os.listdir(f'{metadata_folder}') if f.endswith('.csv')]
    csv_files = sorted(csv_files, key=lambda x: pd.to_datetime(x.split('_')[2]))
    df1 = pd.read_csv(f'{metadata_folder}/{csv_files[0]}', index_col=0, header=None, delimiter=',')
    for csv_file in csv_files[1:]:
        tmp_df = pd.read_csv(f'{metadata_folder}/{csv_file}', index_col=0, header=None, delimiter=',')
        df1 = pd.concat([df1, tmp_df])
    # Save to combined dataframe to csv file
    combined_csv_folder = f'{base_metadata_folder}/combined_csv'
    os.makedirs(combined_csv_folder, exist_ok = True)
    df1.to_csv(f'{combined_csv_folder}/{region}.csv')


def main():
    """ Change/Update the following parameters
    Parameters
    ----------
    download_folder : where to download data
    clip_folder     : save the merged and clipped TIFFs
    epsg_str        : [OPTIONAL] if reprojecting TIFFs from UTM to something different
    njobs           : [OPTIONAL] [Default -1] number of cores inside the Parrallel/delayed line for processing in parallel
    
    Notes
    -----
    Reduce number of jobs/cores if you get memory error during parallel merge/clip operation
    Downloads:
        Downloads is done in serial, so if only downloading data, less than 1GB of memory should be enough
        Comment everything below this line of code: download_sentinel(download_folder, geom, start_date=start_date, end_date=end_date)    
    However, to merge and clip we will need significantly more memory (~8GB per core/process). Increase if if you suspect memory error
        For processing of TIFFs only, comment the download line of code: download_sentinel(download_folder ...................

    Intial download and processing was time-consuming forall Greenlands glaciers
        - 12 hours for download only 30 glaciers
        - 24 hours to for all process (download and clipping) 50 glaciers with 186gb/20 cores  

    Hard-Coded Parameters
    ---------------------
    Not used anymore: Cloud fraction < 50% is hard-coded in the download code suite

    """
    parser = argparse.ArgumentParser(description='Process Sentinel-2 from AWS.')
    parser.add_argument('--start_date', help='First day of Sentinel2 data', type=str)
    parser.add_argument('-ed','--end_date', help='Last day of Sentinel2 data', type=str)
    parser.add_argument('-r','--region', help='Region to process', type=str)
    parser.add_argument('--min_area', help='Subset glaciers greater than or equal to', type=float)
    parser.add_argument('--max_area', help='Subset glaciers lesser than', type=float)
    parser.add_argument('--cores', help='Number of cores to use for multiprocessing', type=int, default=1)
    parser.add_argument('--log_name', help='Name of Log file', type=str, default='sentinel_glacier.log')
    parser.add_argument('--start_end_index', help='Start and End index of regions to process with colon', type=str)

    parser.add_argument('--ignore_region', help='Skip processing this region', type=str)
    parser.add_argument('--collection_name', help='Name of satellite data', type=str, default='sentinel-s2-l2a-cogs')   # sentinel-s2-l1c landsat-8-l1-c1
    parser.add_argument('--download_flag', help='Whether to download data (1=yes, 0=no (default)', type=int, default=0)

    args = parser.parse_args()
    download_flag = args.download_flag
    start_date = args.start_date
    end_date = args.end_date
    if not start_date:
        start_date = '2021-07-01'  # 23 June 2015 (S2A) and 7 March 2017 (S2B)
    if not end_date:
        end_date = datetime.date.today() # end_date='2021-01-31'
        end_date = end_date.strftime('%Y-%m-%d')
    epsg_str="EPSG:3413"  # Required and Fixed for Greenland; project to polar stereographic north before merging adjascent tiffs
    # Set some optional parameters at the start
    cores = args.cores
    min_area = args.min_area
    max_area = args.max_area
    log_name = args.log_name
    start_end_index = args.start_end_index
    ignore_region = args.ignore_region
    collection_name = args.collection_name

    logging.basicConfig(filename=f'{log_name}', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info('  ')
    logging.info('-------------------------START LOGGING--------------------------------------')   
    logging.info(f'cores: {cores}  ')

    root_folder = '/fs/project/howat.4/sentinel2'
    # base_folder = f'{root_folder}/prototyping/2022/110_store'
    base_folder = f'{root_folder}'
    download_parent_folder = f'{base_folder}/downloads' # Location to download sentinel TIFFs; will be created if not exist

    # regions = gpd.read_file(f'{root_folder}/ancillary/glaciers_roi_proj.shp') # Shapefile with AOI for glaciers
    regions = gpd.read_file(f'{root_folder}/ancillary/glacier_roi_v2/glaciers_roi_proj_v3_300m.gpkg') # Shapefile with AOI for glaciers
    # regions = gpd.read_file(f'{root_folder}/ancillary/glacier_roi_v2/glaciers_roi_geog_v2_300m.gpkg') # Shapefile with AOI for glaciers
    regions.index = regions.region
    # To subset based on glacier area
    if min_area:
        regions = regions[regions.Area>=min_area]
        logging.info(f'min_area: {min_area}  ')
    if max_area:
        regions = regions[regions.Area < max_area]
        logging.info(f'max_area: {max_area}  ')
        #regions_subset = regions[np.logical_and(regions.Area>=min_area, regions.Area<max_area)]

    # Sort by Area so parallel job fails towards end rather than the beginning of script
    regions.sort_values(by='Area', inplace=True)
    if start_end_index:
        # to subset based on start and end index
        start, end = start_end_index.split(':')  # : required even if end index is not given!
        start = int(start)
        if end == '':  # if start is specified but end is not given, this implies everything till the last item in index
            end = len(regions)
        end = int(end)
        #idx_list = idx_list[start:end]
        regions = regions.iloc[start:end]

    # Create a list of regions to process
    regions_list = list(regions.region)
    # regions_list = ['128_vestifjord'] #['022_dietrichsons','077_mogens_south']  # or manually
    if ignore_region:
        regions_list.remove(ignore_region)
        logging.info(f'Forced skipping region: {ignore_region}')

    if args.region:
        # overwrite regions_list if only one region supplied
        regions_list = [args.region]
        logging.info(f'regions_list: {regions_list}')

    Total_Regions = len(regions_list)
    logging.info(f'Dates: {start_date} to {end_date} on {cores} cores with Starting region index = {start_end_index}')
    logging.info(f'Total Regions = {Total_Regions}\n  regions_list: {regions_list}  \n')

    region_count = 0
    for region in regions_list:
        logging.info(f'Processing {region} : {region_count}/{Total_Regions}.......')
        # Create Output folder to save processed data
        clip_folder = f'{base_folder}/clipped2/{region}'
        template_folder = f'{base_folder}/template'  # Template raster for reprojection_match
        base_metadata_folder = f'{base_folder}/metadata'
        metadata_folder = f'{base_metadata_folder}/individual_csv/{region}'
        os.makedirs(metadata_folder, exist_ok = True)

        # Get geometry for 1 area to pass to sat-search
        aoi = regions[regions.region == region].copy() #use this for clipping the merged TIFFs
        aoi_lat_lon = aoi.to_crs('EPSG:4326') # only if aoi not in lat/lon geographic coordinates
        # logging.info(f'Feature count in AOI = {len(aoi_lat_lon)} and its CRS = {aoi_lat_lon.crs}, original CRS = {aoi.crs}')
        shp_json = aoi_lat_lon.to_json()
        geom = json.loads(shp_json)['features'][0]['geometry']
        # -----------------------------------------------------------------------------------------

        # 1. Download TIFFs using stac frow AWS Cloud ----------------------------------------------
        # NB: Comment this line if data download is not required
        if download_flag == 1:
            download_sentinel(download_parent_folder, geom, start_date=start_date, end_date=end_date, collection_name = collection_name)
            time.sleep(1) # Optional : delay next request to server if only downloading data

        # # Comment everything below this line if ONLY download of data is required
        # # # 2. To subset, merge and clip TIFFs by AOI ------------------------------------------------
        # # # i. Intersect Sentinel Tile Grids with AOI to get subset of mgrs-style tile names  [No more required as regions shapefile has all this list pre-populated with manual editing as well]. 
        # # gdf = gpd.read_file(f'{root_folder}/ancillary/g_land_tiles2.shp')
        # # res = gpd.overlay(aoi.to_crs(gdf.crs), gdf, how='intersection') #res = result
        # # tile_ids = list(res.Name) ## Names of tiles; example, ['24WVU', '24WVV', '24WWU', '24WWV']
        # # Mar 15, 2022: New approach; get tile_ids with pre-selected tile_ids from geopandas dataframe which was manually edited
        # tile_ids = aoi['utm_grid'].values[0]
        # tile_ids = tile_ids.split(',')

        # # res = gdf[gdf.Name.isin(tile_ids)] # or this one to leave the intersecting polygons intact
        # # Given this tiles, select correponding images from the region
        # # 12/12/2021: Now files are downloaded separately inside year, modify script to match this pattern
        # years = ['2022'] # ['2020', '2019'] #os.listdir(download_parent_folder) # #  #TODO hardcoded for 2021 (perhaps pass as paramter)
        # for yr in years:
        #     # TODO : subset also based on months, perhaps easiest will be to use another nested for loop based on month/s to process
        #     logging.info(f'Processing year: {yr} with tile_ids: {tile_ids}')
        #     download_folder = f'{download_parent_folder}/{yr}'
        #     all_tifs = os.listdir(download_folder)
        #     all_tifs = [tif.split('_B08.tif')[0] for tif in all_tifs if tif.endswith('_B08.tif')] #only required if we have blue, green bands etc as well
        #     # Get all (universal set) TIFF for AOI
        #     aoi_tifs = []
        #     for tile_id in tile_ids:
        #         """ Get all TIFFs that are only within AOI """
        #         tifs1 = [tif for tif in all_tifs if tif.split('_')[5][1:]==tile_id]  # here tile_id as: '24WWU', '24WWV' etc.
        #         aoi_tifs.extend(tifs1)
        #     logging.info(f'Total Number of TIFFs corresponding to AOI = {len(aoi_tifs)}')
        #     # list files in sorted order before clipping  (Mar 16, 2021)
        #     # aoi_tifs = sorted(aoi_tifs, key=lambda x: pd.to_datetime(x.split('_')[2]))  #better to sort after converting to set

        #     # ii. Make a unique filename substring for same day and same satellite TIFFs
        #     # tifs_set = list(set([x[:22] for x in aoi_tifs]))  # truncate filename string after hour of acquisition
        #     tifs_set = set([x[:37] for x in aoi_tifs])  # To include (relative) orbit number
        #     logging.info(f'Unique (tifs_set) TIFFs by same date time and orbit : {len(tifs_set)}')

        #     if not os.path.exists(clip_folder):
        #         os.makedirs(clip_folder)
        #     else:
        #         # Check how many tiffs were processed already and remove these from tifs_set (to save on processing time)
        #         logging.info(f"Existing {clip_folder}")
        #         processed_files = [f.split('.tif')[0] for f in os.listdir(clip_folder)]
        #         processed_files = set(processed_files)
        #         tifs_set = tifs_set.difference(processed_files)
        #         # logging.info(f'Already processed: {len(processed_files)}')  #TODO: misleading when processing by year, revamp it better
        #         logging.info(f'tifs_set remaining to process: : {len(tifs_set)}')
        #         if len(tifs_set)<1:
        #             logging.info("Continue to next region as all files already processed")
        #             continue
        #     tifs_set = list(tifs_set)  # Though not required, it is flexible to work with list than set object
        #     # list files in sorted order before clipping  (Mar 20, 2021)
        #     tifs_set = sorted(tifs_set, key=lambda x: pd.to_datetime(x.split('_')[2]))
        #     # Generate a template raster for clipping (saved inside clipped folder for now)
        #     if not os.path.exists(template_folder):
        #         os.makedirs(template_folder)
        #     template_tif = f'/{template_folder}/{region}.tif'
        #     if not os.path.exists(template_tif):
        #         for tif_prefix in tifs_set:
        #             subset = [f'{x}_B08.tif' for x in aoi_tifs if tif_prefix in x]
        #             merged = rioxarray.open_rasterio(f'{download_folder}/{subset[0]}')  # chunks={'x': 1200, 'y': 1200}
        #             merged = merged.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)  # or as tuple (10,10)
        #             if len(subset) > 1:
        #                 for tif in subset[1:]:
        #                     tmp_ds = rioxarray.open_rasterio(f'{download_folder}/{tif}')  #, chunks={'x': 1200, 'y': 1200}
        #                     tmp_ds = tmp_ds.rio.reproject(epsg_str, resolution=10, resampling=Resampling.cubic)
        #                     merged = merge_arrays([merged, tmp_ds], res = 10)
        #             clipped = merged.rio.clip(aoi.geometry.apply(mapping), aoi.crs) ## A list of geojson geometry dicts.
        #             aoi_area = aoi.Area.item()
        #             clipped_area = xr.where(clipped > 0, 1, 0).sum().item()*100/1e6  # assume anything above 0 is foreground
        #             coverage_fraction = clipped_area/aoi_area
        #             if coverage_fraction > 0.95:  # we are assuming this is 100% coverage
        #                 # Save the Clipped TIFF only if it covers at least 95% of glacier (AOI)
        #                 # Apply Translation for x and y coordinates, or here change to integer values
        #                 clipped['x'] = clipped['x'].astype(int)
        #                 clipped['y'] = clipped['y'].astype(int)
        #                 # Perform exact match with bounding box
        #                 # Check discrepancy between bottom left corrdinates of vector and raster and add this offset to x, y coords of raster
        #                 minx, miny, maxx, maxy = aoi.bounds.values[0]    # vector bounds
        #                 left, bottom, right, top = clipped.rio.bounds()  # raster bounds
        #                 xoff = minx-left
        #                 yoff = miny-bottom
        #                 clipped['x'] = clipped['x'] + xoff
        #                 clipped['y'] = clipped['y'] + yoff
        #                 clipped.rio.to_raster(template_tif) #Save
        #                 break
        #     if cores == 1:
        #         logging.info(f'Serial processing because number of cores = {cores}')
        #         for tif_prefix in tifs_set:
        #             merge_and_clip_tifs(clip_folder, metadata_folder, download_folder, aoi_tifs, tif_prefix, aoi, template_tif, tile_ids, epsg_str=epsg_str)
        #     else:
        #         result = Parallel(n_jobs=cores)(delayed(merge_and_clip_tifs) (clip_folder, metadata_folder, download_folder, aoi_tifs, tif_prefix, aoi, template_tif, tile_ids, epsg_str=epsg_str) for tif_prefix in tifs_set)
        # # Combine all indivisual csv files corresponding to clipped_tif into single csv file
        # # This will include all the csv files that were generated anytime (ie, just not from this script)
        # concat_csv_files(base_metadata_folder, region)
        # region_count += 1
        # logging.info(f'Finished all processing for: {region}--------------------------------------------------------------\n')
    logging.info('-------------------------------------------END LOGGING -------------------------------------------------------')
    print('Finished Main')

if __name__ == "__main__":
    """ Call the main function
    """
    main()
