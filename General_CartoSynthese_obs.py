#!/usr/bin/env python
# coding: utf-8

## Synthese Température -
#
#Base sur un notebook de Travis Logan
#
#Liste des ajouts pour les figures de la synthèse Températures
#    - Possibilité de spécifier une liste de contours au lieu de la gérérer automatiquement
#    - Océan et lacs en gris très pâle au lieu de bleu
#    - Parallèles et méridiens à tous les 5 degrès
#    - Affichage des unités au sommet de la légende de couleurs
#
#  Adaptation aux données observées avec superposition des stations
# Panneaux 3 col (Tmin, Tmoy, Tmax) X 4 lignes pour le cas saisonnier
# Panneaux 3 col (Tmin, Tmoy, Tmax) X 1 ligne pour le cas annuel
#)
# ### Imports and sub-functions : Normally no need to change  

import argparse
from xclim.core.units import convert_units_to
import xarray as xr
from pathlib import Path
import geopandas as gpd
from clisops.core import subset
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
#import hvplot.pandas
#import panel as pn
from IPython.display import clear_output
#import threddsclient as tds
import matplotlib.colors
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.geoaxes import GeoAxes
import sys

try:
    import yaml
except:
    get_ipython().system('pip install pyyaml')
    import yaml

try:
    import figanos.matplotlib as fg
except:
    get_ipython().system('pip install figanos')
    import figanos.matplotlib as fg

from shapely.geometry import Point
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
import matplotlib.font_manager as fm

#
# Forcer l'utilisation de la police Arial conformément à la
# ligne graphique d'Ouranos
#
font_path = '/usr/share/fonts/msttcore/arial.ttf'
fm.fontManager.addfont(font_path)
rcParams['font.family'] = fm.FontProperties(fname='/usr/share/fonts/msttcore/arial.ttf').get_name()
plt.rcParams['font.family'] = 'Arial'
print("Font being used by Matplotlib:")
print(fm.FontProperties().get_name())

crs = ccrs.LambertConformal()
data_crs = ccrs.PlateCarree()

#======================================================================
# Fonctions
#======================================================================
def _get_cmap(cmap_str):             
    cmap = fg.utils.create_cmap(filename=cmap_str)
    return cmap

#-----------------------------------
def _create_fig_agencement(agencement, variables, seasons, figsize=(7,7)):
    if agencement == 'solo_saisonnier':
        ncols=2
        nrows=2
        nbmaps=len(variables)*len(seasons)
        if  nbmaps != ncols*nrows:
            raise ValueError(f"Agencement 'solo_saisonnier' is only suitable for four maps, but got {nbmaps}.")
    else:
        cols=variables
        rows=seasons
        ncols=len(cols)
        nrows=len(rows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection': crs}, figsize=figsize)
    
    if nrows*ncols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for a in axs:
        a.set_aspect('equal')
    if agencement == 'solo_saisonnier':
        axs_keys = [f"{variables[0]}_{s}" for s in seasons ]    
    else:
        axs_keys = [f"{v}_{s}" for s in rows for v in cols ]
    axs = {axs_keys[ii]:ax for ii,ax in enumerate(axs) }
    return fig, axs

#-----------------------------------
def _get_recondata(vars=None, shape=None, recon='ECMWF_ERA5-Land', xfreq='YS-JAN', climdatadir=None):
    # 
    # Obtenir les donnees de reconstruction
    # Pour un fichier YS-JAN, on ajoute une dimension 'season' avec la valeur 'annual'
    # pour faciliter la sélection ultérieure dans le code principal.
    #  
    if shape is None:
        shape = '/exec/cote/obsflow/gis/regions_simplified_qc.geojson'

    if climdatadir is None: 
        raise ValueError("The 'climdatadir' parameter must be provided.")
    
    reconfile = Path(f'{climdatadir}/{recon}_NAM_reconstruction_{xfreq}_climatology.zarr')
    if not reconfile.exists():
        print(f"Warning: The file '{reconfile}' does not exist.")

    ds_list = []
    for vv in vars:   
        ds_list.append(xr.open_zarr(reconfile, chunks='auto'))
    
    ds = xr.merge(ds_list)
    
    for crd in ds.coords:
        if ds[crd].dtype == '|S64':
            ds[crd] = ds[crd].astype(str)
    if xfreq == 'YS-JAN':
        season= ['annual']
        ds = ds.expand_dims(dim={'season': season})
        ds = ds.assign_coords(season=season)

        for var in ds.data_vars:
            print(f"Variable: {var}, Dimensions: {ds[var].dims}")

        # Broadcast the 'season' coordinate to all variables
        for var in ds.data_vars:
            if 'season' not in ds[var].dims:
                print(f"Adding 'season' dimension to variable: {var}")
                ds[var] = ds[var].expand_dims(dim={'season': season})
                # Align dimensions explicitly
                ds[var] = ds[var].transpose('season', *[dim for dim in ds[var].dims if dim != 'season'])

    return subset.subset_shape(ds, shape)

#-----------------------------------
def _plot_data(fig, axs_dict, ds, carto_dict, extent, agencement, add_contours=False, add_contour_lab=False):
    draw_gridlines= carto_dict.get('draw_gridlines', True)
    if "levels" in carto_dict.keys():
            levels=carto_dict["levels"]
    else:
        v0 =carto_dict["colorlim"][0]
        v1 = carto_dict["colorlim"][-1]
        interv = carto_dict["c_interv"]
        levels = np.arange(v0, v1+interv, interv)
    cmap = _get_cmap(carto_dict["cmap"])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    id = carto_dict.get("asymmetric", False)
    if id:
        vmin = min(levels)
        vmax = max(levels)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    if 'extend' in carto_dict.keys():
        extend =  carto_dict["extend"]
    else:
        extend = None

    #
    # Certains réglages dépendent de l'agencement choisi
    # - cbar_height est la hauteur de la barre de couleurs, 
    # - latlonsize est la taille des étiquettes de latitude et longitude, 
    # - Ajustements pour la position des sous-figures:
    #       adj_right/left/top/bottom/wspace/hspace
    #       definies par une valeur ou par défaut à partir de mpl.rcParams
    # 
    if agencement == 'trio_saisonnier':
        cbar_height=0.6
        latlonsize=4
        adj_right=0.85
        adj_left=mpl.rcParams['figure.subplot.left']
        adj_top=mpl.rcParams['figure.subplot.top']
        adj_bottom=mpl.rcParams['figure.subplot.bottom']
        adj_wspace=0.05
        adj_hspace=0.1
    elif agencement == 'duo_saisonnier':
        cbar_height=0.6
        latlonsize=4
        adj_right=0.75
        adj_left=0.25
        adj_bottom=mpl.rcParams['figure.subplot.bottom']
        adj_top=mpl.rcParams['figure.subplot.top']
        adj_wspace=0.02
        adj_hspace=0.1
    elif agencement == 'solo_saisonnier':
        cbar_height=0.5
        latlonsize=6
        adj_right=0.85
        adj_left=mpl.rcParams['figure.subplot.left']
        adj_top=mpl.rcParams['figure.subplot.top']  
        adj_bottom=0.2
        adj_wspace=0.15
        adj_hspace=0.02
    elif agencement == 'solo_annuel':
        cbar_height=0.5
        latlonsize=8
        adj_right=0.85
        adj_left=mpl.rcParams['figure.subplot.left']
        adj_top=mpl.rcParams['figure.subplot.top']
        adj_bottom=mpl.rcParams['figure.subplot.bottom']
        adj_wspace=0.05
        adj_hspace=0.1
    elif agencement == 'trio_annuel':
        cbar_height=0.175
        latlonsize=4
        adj_right=0.85
        adj_left=mpl.rcParams['figure.subplot.left']
        adj_top=mpl.rcParams['figure.subplot.top']
        adj_bottom=mpl.rcParams['figure.subplot.bottom']
        adj_wspace=0.05
        adj_hspace=0.1
    else:
        cbar_height=0.5
        latlonsize=4
        adj_right=0.85
        adj_left=mpl.rcParams['figure.subplot.left']
        adj_top=mpl.rcParams['figure.subplot.top']
        adj_bottom=mpl.rcParams['figure.subplot.bottom']        
        adj_wspace=0.05
        adj_hspace=0.1


    for key in axs_dict.keys():
        #get variable and season from axs key
        str1 = key.split('_')
        seas=str1[-1]
        vari="_".join(str1[0:-1])
        print(f"Plotting variable: {vari}, season: {seas} on axis: {key}")
        print (f"Data coordinates: {ds[vari].coords}")

        # Ensure 'season' is properly aligned
        if 'season' not in ds[vari].dims:
            raise ValueError(f"'season' is not a dimension in variable {vari}. Check the dataset structure.")

        # Get the index of the desired season
        season_index = int(np.where(ds['season'].values == seas)[0][0])
        print("-----season_index dans _PLOT_DATA-------")
        print(f"Season '{seas}' found at index: {season_index}")

        #  Use isel to select the season using the index,
        #  and squeeze to remove the season dimension
        #  Ensure data is 2D
        #
        data = ds[vari].isel(season=season_index).squeeze()
        if len(data.shape) != 2:
            raise ValueError(f"Data for {col} is not 2D after squeezing: {data.shape}")

        axs[key].set_extent(extent, crs=data_crs)
        axs[key].set_aspect('equal')

        im = axs[key].contourf(data.lon.values, data.lat.values, data.values, levels=levels, cmap=cmap, norm=norm, transform=data_crs, extend=extend)
        
        if draw_gridlines:
            gl=axs[key].gridlines(
                draw_labels=True, 
                x_inline=False, 
                y_inline=False, 
                rotate_labels=False, 
                xpadding=-1.0, 
                ypadding=-1.0,
                linestyle='dotted', 
                color='grey', 
                alpha=0.6, 
                zorder=1
            )
            gl.top_labels = False
            gl.right_labels = True
            gl.left_labels = False
            gl.bottom_labels = True

            gl.xlabel_style={'size': latlonsize }
            gl.ylabel_style={'size': latlonsize }
            gl.xlocator = MultipleLocator(5)  # Longitude gridlines every 5 degrees
            gl.ylocator = MultipleLocator(5)   # Latitude gridlines every 5 degrees
        axs[key].coastlines(linewidth=0.1)
        axs[key].add_feature(cfeature.OCEAN, facecolor='#e0e0e0')
        axs[key].add_feature(cfeature.LAKES, facecolor='#e0e0e0')
        #ax.add_feature(cfeature.RIVERS,linewidth=0.1)
        if add_contours:
            CS = axs.contour(data.lon.values, data.lat.values, data.values, levels=levels, linewidths=[i*.0 for i in levels], transform=data_crs)
            if add_contour_lab:
                axs.clabel(CS, CS.levels, inline=True, colors='k', fontsize=3)

        if agencement == 'solo_saisonnier':
            axs[key].set_title(carto_dict["titles_saisons"]["seasonal"][seas])
            axs[key].set_ylabel("")
        elif agencement == 'solo_annuel':
            axs[key].set_title("")
            axs[key].set_ylabel("")
        else:
            if seas == seasons[0]:
                axs[key].set_title(carto_dict["titles_variables"][vari])
            else:
                axs[key].set_title("")
            if vari == variables[0]:
                if seas=='annual':
                    axs[key].set_ylabel("") 
                else:
                    axs[key].set_ylabel(carto_dict["titles_saisons"]["seasonal"][seas])
            else:
                axs[key].set_ylabel("")
    
    #
    # Ajustement de la position des sous-figures selon l'agencement choisi
    #
    fig.subplots_adjust(right=adj_right, left=adj_left, top=adj_top, bottom=adj_bottom, wspace=adj_wspace, hspace=adj_hspace)   
    
    #
    # Ajustements de la barre de couleurs
    #  - taille et position selon l'agencement choisi
    #  - ticks selon les niveaux ou un pas spécifié dans le YAML
    #  - Traduction des unités
    #  - par décennies si tendances
    #
    cbar_y0 = 0.5 - (cbar_height / 2)  # Center the colorbar vertically
    cbar_x0 = adj_right + 0.05  # Position the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([cbar_x0, cbar_y0, 0.03, cbar_height])  # [x0, y0, width, height]
    ticks_step = carto_dict.get('ticks_step', None)
    if ticks_step:
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=levels[::ticks_step])
    else:
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=levels)

    if "units" not in data.attrs:
        print("Warning: 'units' attribute is missing. Setting to 'unknown'.")
        data.attrs["units"] = 'unknown'
        unites= carto_dict.get("units")
    else:
        unites = data.attrs['units']

    clab = unites
    clab= clab.replace('degC', '°C')
    clab= clab.replace('K', '°C')
    clab= clab.replace('/year', '/décennie')
    # Add horizontal label above the colorbar using fig.text
    x = cbar_ax.get_position().x0 + cbar_ax.get_position().width / 2
    y = cbar_ax.get_position().y1 + 0.01  # a bit above the colorbar
    fig.text(x, y, clab, ha='center', va='bottom', fontsize=12)
    
#==============FIN DES FONCTIONS =============================

############## CODE PRINCIPAL ###############################
# Le code principal est organisé en trois sections :
# 1) Chargement du fichier de configuration YAML et définition des paramètres
# 2) Acquisition des données de réanalyses et des stations selon les paramètres définis
# 3) Code de tracé des cartes : 
#   - Organisation du panneau de figures selon les paramètres définis
#   - Ajouts des données aux stations si requis
#   - Ajouts des zones hachurées associées si les tendances non-signicatives
#   - Ajout d'une insertion pour les Iles-de-la-Madeleine si requis
#     (Le code est basé sur l'hypothèse qu'il n'y a qu'une seule station pour les 
#     Iles-de-la-Madeleine)
#
#   ATTENION: le nom de certaines stations peut différer d'une source de données à l'autre.
#
#############################################################
#
# Parse command-line arguments to provide YAML file path.
# Then access variables and parameters from the YAML file

parser = argparse.ArgumentParser(description="Run General_CartoSynthese with a configurable YAML file.")
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    help="Path to the YAML configuration file."
)
args = parser.parse_args()
config_file = Path(args.config)
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

add_contours = config["add_contours"]
add_contour_lab = config["add_contour_lab"] # whether to add contour labels
climdatadir = Path(config["climdatadir"])   # directory for climatology data
outfolder = Path(config["outfolder"])   # output folder for figures
outfolder.mkdir(exist_ok=True)
season_dict = config["season_dict"]
noms_variables_dict = config["noms_variables_dict"]
panno_dict = config["panno_dict"]


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.geoaxes import GeoAxes

#
#Initialisation
#
station_IDM_all_nan = True  #mettre à True si les données des stations contiennent des valeurs pour les Iles-de-la-Madeleine

province_shape = Path(f'/exec/cote/obsflow/gis/regions_simplified_qc.geojson')

#
#Etendue désirée pour la province de Québec
#
prov_lon_min=-80.8
prov_lon_max=-59.0
prov_lat_min=44.65
prov_lat_max=60.95   
province_extent = [prov_lon_min, prov_lon_max, prov_lat_min, prov_lat_max]

regions = gpd.read_file('/exec/cote/obsflow/gis/region_mrc.zip')

#
# Reglages relatifs a l'affichage des données des Iles-de-la-Madeleine
# Valeurs possibles pour le IDM.buffer: 0.1 ou 0.08
#  
IDM = regions.loc[regions.name == "Communauté maritime des Îles-de-la-Madeleine"]
IDM=IDM.to_crs(crs=data_crs)
IDM['geometry'] = IDM.buffer(0.08)

for panno in panno_dict.keys():
    print(f"Processing PANNO:  {panno}")

    #
    # identification des donnees 
    #
    
    horizon = panno_dict[panno]['horizon']
    recon= panno_dict[panno]['recon']
    affiche_IDM = panno_dict[panno]['affiche_IDM']
    affiche_stations = panno_dict[panno]['affiche_stations']
    stationsprefix = panno_dict[panno]['stationsprefix']
    tendances = panno_dict[panno]['tendances']
    absvalues = panno_dict[panno]['absvalues']
    xfreq = panno_dict[panno]['xfreq']
    variables = panno_dict[panno]['variables']
    print(f"LE TYPE Variables to plot: {type(variables)}")
    
    if stationsprefix == "GovCan_CanHomT-daily_CAN_station-tas":
        id_station_IDM = ['ILES DE LA MADELEINE']
    elif stationsprefix =='GovCan_AHCCD_CAN_station-tas':
        id_station_IDM = ['ILES_MADELEINE']
    else:
        print (f"Warning: Unrecognized station prefix '{stationsprefix}'. No specific station IDs for Îles-de-la-Madeleine will be set.")
    
    if tendances:
        scale_factor=10.0  # pour convertir en décénnie
        mpl.rcParams['hatch.linewidth'] = 0.3  # Set default hatch linewidth                                     
        mpl.rcParams['hatch.color'] = "dimgray"  # Set default hatch color 
    #
    # organisation du panneau de figures
    #
    agencement = panno_dict[panno]['agencement']
    seasons = panno_dict[panno]['seasons']
    figsize = panno_dict[panno]['figsize']

    #
    # parametres d'affichage communs a toutes les cartes du panneau
    # certains sont optionnels tandis que d'autres seront définis par défaut 
    # 
       
    #Taille des symboles pour les données des stations selon les agencements 
    if agencement == 'solo_annuel':
        marker_size = 20 
    else:
        marker_size = 8  
    
    carto_dict= panno_dict[panno]['carto_options']
    cmap= str(carto_dict['cmap'])
    colorlim= carto_dict.get('colorlim', None)      
    c_interv= carto_dict.get('c_interv', None)
    extend= carto_dict['extend']
    edgecolor= carto_dict['edgecolor']
    suptitle= carto_dict.get('suptitle', "") 
    ticks_step= carto_dict.get('ticks_step', None)
    levels= carto_dict.get('levels', None)
    asymmetric= carto_dict.get('asymmetric', False)
    divergent= carto_dict.get('divergent', False)
    units= carto_dict.get('units', 'unknown')    
    draw_gridlines= carto_dict.get('draw_gridlines', True)
    
    #
    # Acquisition des donnees
    #

    ds = _get_recondata(vars=variables, shape=province_shape, recon = recon, xfreq = xfreq, climdatadir=climdatadir) 
 
    if affiche_IDM:
        ds_IDM = _get_recondata(vars=variables, shape=IDM, recon = recon, xfreq = xfreq, climdatadir=climdatadir)
        # Check if all variables in ds_IDM are NaN
        recon_IDM_all_nan = True  # Flag to track if all variables are NaN

        for var in ds_IDM.data_vars:
            if not ds_IDM[var].isnull().all():
                print(f"IDM: Variable '{var}' is not entirely NaN.")
                recon_IDM_all_nan = False
            else:
                print(f"IDM: Variable '{var}' is entirely NaN.")

            if recon_IDM_all_nan:
                    print("IDM: All variables in ds_IDM are entirely NaN.")
            else:
                print("IDM: Not all variables in ds_IDM are NaN.")
        
        # Get the bounds of the ds_IDM subregion
        IDM_lon_min = ds_IDM.lon.min().compute().item()
        IDM_lon_max = ds_IDM.lon.max().compute().item()
        IDM_lat_min = ds_IDM.lat.min().compute().item()
        IDM_lat_max = ds_IDM.lat.max().compute().item()
        IDM_extent = [IDM_lon_min, IDM_lon_max, IDM_lat_min, IDM_lat_max]

    # Extraire les tendances et p-value
    if tendances:
        dstrend = ds[variables].sel(linreg_param = 'slope').squeeze(drop=False)
        dstrend = dstrend * scale_factor
        dspvalue = ds[variables].sel(linreg_param = 'pvalue').squeeze(drop=False)
        
        # Ensure 'season' is a proper dimension
        if 'season' not in dstrend.dims and seasons == ['annual']:
            dstrend = dstrend.expand_dims(dim={'season': seasons})
            dstrend = dstrend.assign_coords(season=seasons)
        # Set 'season' as an index
        if 'season' not in dstrend.indexes:
            dstrend = dstrend.set_index({'season': 'season'})
        
        if 'season' not in dspvalue.dims and seasons == ['annual']:
            dspvalue = dspvalue.expand_dims(dim={'season': seasons})
            dspvalue = dspvalue.assign_coords(season=seasons)
        # Set 'season' as an index
        if 'season' not in dspvalue.indexes:
            dspvalue = dspvalue.set_index({'season': 'season'})
    
    #if tendances and affiche_IDM and (not recon_IDM_all_nan):
    if tendances and affiche_IDM:
        # Habituellement, il n'y a pas de valeurs valides pour les Iles-de-la-Madeleine
        # dans les données de réanalyses. On les extrait quand même au cas où.
        dstrend_IDM = ds_IDM[variables].sel(linreg_param = 'slope')
        dstrend_IDM = dstrend_IDM * scale_factor
        dspvalue_IDM = ds_IDM[variables].sel(linreg_param = 'pvalue')

    if affiche_stations:
        # Donnees des stations                                                                                           
        stationsfile = Path(f'{climdatadir}/{stationsprefix}_{xfreq}_climatology.zarr')
        if not stationsfile.exists():
            print(f"Warning: The file '{stationsfile}' does not exist.")
        ds_stations = xr.open_zarr(stationsfile, chunks='auto')
        if xfreq == 'YS-JAN':
            season= ['annual']
            ds_stations = ds_stations.expand_dims(dim={'season': season})
            ds_stations = ds_stations.assign_coords(season=season)
            for var in ds_stations.data_vars:
                if 'season' not in ds_stations[var].dims:
                    print(f"Adding 'season' dimension to variable: {var}")
                    ds_stations[var] = ds_stations[var].expand_dims(dim={'season': season})
                    # Align dimensions explicitly
                    ds_stations[var] = ds_stations[var].transpose('season', *[dim for dim in ds_stations[var].dims if dim != 'season'])
        # Set 'season' as an index
        if 'season' not in ds_stations.indexes:
            ds_stations = ds_stations.set_index({'season': 'season'})

        print(f"Available variables in ds_stations: {list(ds_stations.data_vars)}")
        ds_stations = ds_stations[variables]

        if affiche_IDM:
            # Vérifier si les données des stations pour les Iles-de-la-Madeleine sont toutes NaN
            mask = (ds_stations['station_name'] == id_station_IDM).compute()
            ds_station_IDM = ds_stations.where(mask, drop=True)
            station_IDM_all_nan = True  # Flag to track if all variables are NaN for IDM station
            station_IDM_all_nan = ds_station_IDM.isnull().to_array().all().compute().item()
            if station_IDM_all_nan:
                print("All station data for Îles-de-la-Madeleine are entirely NaN.")
            else:
                print("Not all station data for Îles-de-la-Madeleine are NaN.")
        
        if tendances:
            dstrend_stations = ds_stations.sel(linreg_param = 'slope')
            dstrend_stations = dstrend_stations * scale_factor
            dspvalue_stations = ds_stations.sel(linreg_param = 'pvalue')
        
        if tendances and affiche_IDM and (not station_IDM_all_nan):
            #Chercher la station des iles-de-la-madeleine                                                                   
            mask = (ds_stations['station_name'] == id_station_IDM).compute()
            dstrend_station_IDM = dstrend_stations.where(mask, drop=True)
            dstrend_station_IDM = dstrend_station_IDM # deja multiplié par scale_factor
            dspvalue_station_IDM = dspvalue_stations.where(mask, drop=True)
    
    for c in ds.coords:
        print(ds[c].dtype)
        if ds[c].dtype == '|S64':
            ds[c] = ds[c].astype(str)
    for c in ds_IDM.coords:
        print(ds_IDM[c].dtype)
        if ds_IDM[c].dtype == '|S64':
            ds_IDM[c] = ds_IDM[c].astype(str)
    
    if tendances:
        dsin = dstrend.sel(season=seasons, horizon=horizon)
        dsinpvalue = dspvalue.sel(season=seasons, horizon=horizon)
    else:
        dsin = ds.sel(season=seasons, horizon=horizon)
    
    # Ensure the 'season' dimension is preserved
    if 'season' not in dsin.dims:
        dsin = dsin.expand_dims(dim={'season': seasons})
        dsin = dsin.assign_coords(season=seasons)
    # Align dimensions explicitly
    for var in dsin.data_vars:
        if 'season' not in dsin[var].dims:
            dsin[var] = dsin[var].expand_dims(dim={'season': seasons})
            dsin[var] = dsin[var].transpose('season', *[dim for dim in dsin[var].dims if dim != 'season'])    
    if tendances:
        if 'season' not in dsinpvalue.dims:
            dsinpvalue = dsinpvalue.expand_dims(dim={'season': seasons})
            dsinpvalue = dsinpvalue.assign_coords(season=seasons)
        for var in dsinpvalue.data_vars:
            if 'season' not in dsinpvalue[var].dims:
                dsinpvalue[var] = dsinpvalue[var].expand_dims(dim={'season': seasons})
                dsinpvalue[var] = dsinpvalue[var].transpose('season', *[dim for dim in dsinpvalue[var].dims if dim != 'season'])

    if affiche_IDM and (not recon_IDM_all_nan):    
        if tendances:
            dsin_IDM = dstrend_IDM.sel(season=seasons,horizon=horizon)
            dsinpvalue_IDM = dspvalue_IDM.sel(season=seasons,horizon=horizon)
        else:
            dsin_IDM = ds_IDM.sel(season=seasons,horizon=horizon)

    if affiche_stations:
        if tendances:
            data_stations = dstrend_stations.sel(season=seasons, horizon=horizon)
            datapvalue_stations = dspvalue_stations.sel(season=seasons, horizon=horizon)
            datapvalue_stations = datapvalue_stations
        else:
            data_stations = ds_stations.sel(season=seasons,horizon=horizon) # Garder la dimension 'season' pour le moment (annual)

        if affiche_IDM and (not station_IDM_all_nan):
            if tendances:
                data_station_IDM = dstrend_station_IDM.sel(season=seasons, horizon=horizon)
                datapvalue_station_IDM = dspvalue_station_IDM.sel(season=seasons, horizon=horizon)
                datapvalue_station_IDM = datapvalue_station_IDM
            else:
                data_station_IDM = ds_station_IDM.sel(season=seasons,horizon=horizon)

            data_station_IDM = data_station_IDM
            if 'station' in data_station_IDM.dims:
                data_station_IDM = data_station_IDM.dropna(dim='station', how='any')
            else:
                print("Warning: 'station' dimension not found in data_station_IDM.")
                print(data_station_IDM.values)
    #
    # Affichage des cartes de chaque panneau    
    # Create subplot axes : axs is a dictionary 
    #
    fig, axs= _create_fig_agencement(agencement, variables, seasons, figsize=figsize)
    print(f" axs keys: {axs.keys()}")
    print(f" axes axs: {len(axs)}")
    
    # Affichage des données de réanalyses (tendances ou autres) de tout le Quebec
    # pour toutes les variables et saisons du panneau de figures
    #
    _plot_data(fig, axs, dsin, carto_dict, province_extent, agencement, add_contours=add_contours, add_contour_lab=add_contour_lab)

    # Création du titre global du panneau de figures
    # Le réglage de la hauteur du titre est ajusté en fonction de l'agencement
    # pour éviter les chevauchements
    #
    if agencement in ['trio_annuel'] :
        y_suptitle=0.66
    elif agencement in ['solo_saisonnier']:
        y_suptitle=0.9
    elif agencement in ['solo_annuel']:
        y_suptitle=0.83
    else:
        y_suptitle=0.94
    
    first_var=variables[0]
    if 'mean_clim_mean' in first_var:
        if seasons== ['annual']: 
            statis="normales annuelles"
        else:
            statis="normales saisonnières"
    elif 'std_clim_mean' in first_var:
        if seasons == ['annual']:   
            statis="écart-types intra-annuels"
        else:
            statis="écart-types intrasaisonniers"
    elif 'mean_clim_std' in first_var:
        statis="écart-types interannuels"
    elif 'clim_linregress' in first_var:
        if seasons == ['annual']:
            statis="tendances annuelles"
        else:           
            statis="tendances saisonnières"
    else:
        statis=""          
    if 'solo' in agencement:
        vv = first_var.split("_")[0]
        if not any(substring in first_var for substring in ['max_clim_mean', 'min_clim_mean']):
            aliasvv = noms_variables_dict[vv]
            prefixe = f"{aliasvv}: " 
        else:
            prefixe = ""     
    else:
        prefixe = ""
    if suptitle:
        if suptitle == "Default":
            title = f"{prefixe} {statis} {horizon}"
        else:
            #title = f"{prefixe} {suptitle} {horizon}"
            title = f"{suptitle} {horizon}"
        plt.suptitle(title, y=y_suptitle)
    else:
        print("Warning: 'suptitle' is not defined in carto_dict. No global title will be added to the figure.")
    
    # Ajouts sur chacune des cartes du panneau
    # Pour chaque axe, ajouter le hatching, les stations et les insets
    # selon les options

    for key, ax in axs.items():
        str1 = key.split('_')
        row  = str1[-1]
        col  = "_".join(str1[0:-1])

        # Hatching p-value significance pour les tendances de la réanalyse
        # sur tout le Québec  
        if tendances:
            datapvalue= dsinpvalue[col].sel(season=row).squeeze()

            axs[key].contourf(
                datapvalue.lon.values,
                datapvalue.lat.values,
                np.where(datapvalue.values > 0.05, 1, np.nan),  # Mask areas where pvalue <= 0.05                      
                levels=[0.5, 1.5],  # Define levels for hatching                                                 
                colors='none',  # No fill color                                                                  
                hatches=['///'],  # Hatching pattern                                                             
                transform=data_crs,
                zorder=4
                )
        #
        # Gestion des carto_options communes à toutes les cartes du panneau
        #
        if isinstance(carto_dict['cmap'], str):
            cmap = _get_cmap(carto_dict['cmap'])
        else:
            raise ValueError(f"Invalid cmap value: {carto_dict['cmap']}. Expected a string.")

        if levels is not None:  
            print(f"Using predefined levels for {col}: {levels}")
        else:
            vmin = colorlim[0]
            vmax = colorlim[-1]
            interv = c_interv
            levels = np.arange(vmin, vmax + interv, interv)
            print(f"Computed levels for {col}: {levels}")
        nlevels=len(levels)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        if asymmetric:
            vmin = min(levels)
            vmax = max(levels)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # Ajout des stations de tout le Québec sur chaque 
        # carte du panneau
        print(f"----ROW {row} COL {col}----")

        if affiche_stations:                            
            station_data = data_stations[col].sel(season=row)

            if tendances:
                # Filter datapvalue_stations by the current season
                if 'season' not in datapvalue_stations.dims:
                    print("Warning: 'season' is not a dimension. Expanding it to match the 'station' dimension.")
                    datapvalue_stations = datapvalue_stations.expand_dims(dim={'season': [row]})
                    datapvalue_stations = datapvalue_stations.assign_coords(season=[row])

                datapvalue_stations = datapvalue_stations.sel(season=row)

                # Ensure alignment between station_data and datapvalue_stations
                datapvalue_stations = datapvalue_stations.sel(station=station_data.station)

                # Create masks
                mask_significant = datapvalue_stations[col].values <= 0.05
                mask_insignificant = datapvalue_stations[col].values > 0.05
                        
                # Scatter plot for significant stations
                axs[key].scatter(
                    station_data.lon.values[mask_significant],
                    station_data.lat.values[mask_significant],
                    s=marker_size,
                    c=station_data.squeeze().values[mask_significant],
                    cmap=cmap,
                    norm=norm,
                    marker='o',
                    transform=data_crs,
                    edgecolor=edgecolor,
                    linewidth=0.5,
                    zorder=10,
                )
                # Scatter plot for insignificant stations   
                axs[key].scatter(
                    station_data.lon.values[mask_insignificant],
                    station_data.lat.values[mask_insignificant],
                    s=marker_size,
                    c=station_data.squeeze().values[mask_insignificant],    
                    cmap=cmap,
                    norm=norm,
                    marker='s',
                    transform=data_crs,
                    edgecolor=edgecolor,
                    linewidth=0.5,
                    zorder=10,
                )
                print(f"station data values: {col} {row} {station_data.squeeze().values}")
            else:
                # Scatter plot for all stations (when not tendances)
                axs[key].scatter(
                    station_data.lon.values,
                    station_data.lat.values,
                    s=marker_size,
                    c=station_data.squeeze().values,
                    cmap=cmap,
                    norm=norm,
                    marker='o',
                    transform=data_crs,
                    edgecolor=edgecolor,
                    linewidth=0.5,
                    zorder=10,
                )
                print(f"station data values: {col} {row} {station_data.squeeze().values}")
        #
        # Ajout des insets pour les Iles-de-la-Madeleine si requis
        # sur chaque carte du panneau.
        # Les insets ne sont ajoutés que si au moins un des deux
        # jeux de données IDM (réanalyse ou stations) contient des valeurs
        # 
        #if affiche_IDM and ((not recon_IDM_all_nan) or (not station_IDM_all_nan)):
        if affiche_IDM:
            inset_ax = inset_axes(ax, width="18%", height="32%", loc='lower left',
                bbox_to_anchor=(0.74, 0.09, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
                axes_class=GeoAxes,
                axes_kwargs={'projection': crs})

            inset_ax.set_extent(IDM_extent, crs=data_crs)

            if not(recon_IDM_all_nan) and not(tendances):
                # si données de réanalyse disponibles sur IDM
                # Pas de tendances car trop petit pour hachurer si non significatif
                # (Alternative a developper: changer le motif de hachures pour les insets 
                # afin de pouvoir afficher les tendances meme si non significatives)

                data = dsin_IDM[col].sel(season=row)
                data= data.squeeze()
                if len(data.shape) != 2:   # Ensure data is 2D
                    raise ValueError(f"Data is not 2D after squeezing: {data.shape}")

                im = inset_ax.contourf(
                    data.lon.values,
                    data.lat.values,
                    data.values,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=data_crs
                )
                        
            # Optional: add region boundary
            inset_ax.add_geometries(IDM.geometry, crs=data_crs, facecolor='none', edgecolor='none')
            inset_ax.coastlines(linewidth=0.1)
            inset_ax.add_feature(cfeature.OCEAN,facecolor='#e0e0e0', zorder=3)
            gl_inset = inset_ax.gridlines(draw_labels=False, color="none")
            gl_inset.top_labels=False
            gl_inset.bottom_labels=False
            gl_inset.left_labels=False
            gl_inset.right_labels=False

            if affiche_stations and (not station_IDM_all_nan): 
                # station IDM
                station_data_IDM = data_station_IDM[col].sel(season=row)
                #station_data_IDM = station_data_IDM.squeeze()
                print (f"station_data_IDM size: {station_data_IDM.size}")
                # Check if 'season' is a dimension in datapvalue_station_IDM
                if tendances:
                    if 'season' not in datapvalue_station_IDM.dims:
                        print("Warning: 'season' is not a dimension in datapvalue_station_IDM. Expanding it to match the 'station' dimension.")
                        datapvalue_station_IDM = datapvalue_station_IDM.expand_dims(dim={'season': [row]})
                        datapvalue_station_IDM = datapvalue_station_IDM.assign_coords(season=[row])
                    # Verify the updated structure
                    print(f"Updated dimensions for datapvalue_station_IDM: {datapvalue_station_IDM.dims}")
                    print(f"Updated coordinates for datapvalue_station_IDM: {datapvalue_station_IDM.coords}")
                    datapvalue_station_IDM = datapvalue_station_IDM.sel(season=row) 

                    # creation des masks IDM selon la p-value
                    mask_IDM_significant = datapvalue_station_IDM[col].values <= 0.05
                    mask_IDM_insignificant = datapvalue_station_IDM[col].values > 0.05
                    print(f"mask_IDM_significant size : {mask_IDM_significant.size}")
                    print(f"mask_IDM_insignificant size : {mask_IDM_insignificant.size}")
                            
                    if mask_IDM_significant:
                        inset_ax.scatter(
                            station_data_IDM.lon.values,  
                            station_data_IDM.lat.values,              
                            s=15,
                            c=station_data_IDM.squeeze().values,
                            cmap=cmap,
                            norm=norm,
                            marker='o',
                            transform=data_crs,
                            edgecolor='dimgray',
                            linewidth=0.5,
                            zorder=10,
                        )
                    if mask_IDM_insignificant:
                        inset_ax.scatter(
                            station_data_IDM.lon.values,  
                            station_data_IDM.lat.values,
                            s=15,
                            c=station_data_IDM.squeeze().values,
                            cmap=cmap,
                            norm=norm,
                            marker='s',
                            transform=data_crs,
                            edgecolor='dimgray',
                            linewidth=0.5,
                            zorder=10,
                        )
                else:
                    inset_ax.scatter(
                        station_data_IDM.lon.values,
                        station_data_IDM.lat.values,
                        s=15,
                        c=station_data_IDM.squeeze().values,
                        cmap=cmap,
                        norm=norm,
                        marker='o',
                        transform=data_crs,
                        edgecolor='dimgray',
                        linewidth=0.5,
                        zorder=10,
                    )
    outfig = outfolder.joinpath(f"{panno}.png")
    print(outfig)
    fig.savefig(outfig, bbox_inches='tight', dpi=300)                

print('Finished.')
