# -*- coding: utf-8 -*-
"""
Created on Apr 29, 2025
@author: Sergi Gonzalez Fajardo
"""
#%% INITIAL STUDY OF THE AREA FOR THE RESEARCH OF SPRING BLOOMS ON ANTARCTICA (CHL)  AND CHL OVERALL DATA %%#
#%% LIBRARIES
import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from datetime import datetime, timedelta

import os
#%% LOAD DATA, EXTRACT and ADAPT VARIABLES

chl_path = "C:/Users/sergo/OneDrive - Danmarks Tekniske Universitet/MCS Ocean Engineering/2nd semester/25340 Digital Ocean/Final Project/DATA/erdMH1chlamday_873e_e104_a9e7.nc"

# Output of figures
output_dir = "C:/Users/sergo/OneDrive - Danmarks Tekniske Universitet/MCS Ocean Engineering/2nd semester/25340 Digital Ocean/Final Project/output_visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# EXTRACT and ADAPT VARIABLES 

ds_chl = nc.Dataset(chl_path)

time_chl = ds_chl.variables['time'][:]
lon = ds_chl.variables['longitude'][:]
lat = ds_chl.variables['latitude'][:]
chl = ds_chl.variables['chlorophyll'][:]  # (time, lat, lon)


# reduce resolution but having better wuality generation
chl= chl.astype(np.float16)

# Time adaptation seconds to month and year
dates_chl = [datetime(1970,1,1) + timedelta(seconds=t) for t in time_chl]
dates_chl_num = mdates.date2num(dates_chl)
# Meshgrids

projection_in = ccrs.PlateCarree()
lon2d, lat2d = np.meshgrid(lon, lat)

#%% MAXIMUM CONCENTRATION


vmin_chl, vmax_chl = 0, 60
chl_max = np.nanmax(chl, axis=0)
norm_chl = colors.Normalize(vmin_chl,vmax_chl)


fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
ax.set_title('Maximum Chlorophyll Concentration [mg/m³] (2017–2022)', fontsize=14)
ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
ax.coastlines(resolution='50m', color='black')
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


max_plot = ax.pcolormesh(lon2d, lat2d, chl_max, transform=ccrs.PlateCarree(),
                         cmap='YlGn', norm=norm_chl, shading='auto')


bloom_zones = {
    "SB 1: Weddell Sea": {"N": -62.08, "S": -65.32, "W": -58.2,  "E": -43.97, "color": "red"},
    "SB 2: Ross Sea": {"N": -74.11, "S": -78.12, "W": 161.24, "E": 179.85, "color": "blue"},
    "SB 3: East Antarctica": {"N": -67.18, "S": -70.42, "W": 14.526, "E": 28.820, "color": "green"},
}

legend_patches = []

for label, coords in bloom_zones.items():
    rect_lons = [coords["W"], coords["E"], coords["E"], coords["W"], coords["W"]]
    rect_lats = [coords["S"], coords["S"], coords["N"], coords["N"], coords["S"]]
    ax.plot(rect_lons, rect_lats, transform=ccrs.PlateCarree(),
            color=coords["color"], linewidth=2, linestyle='-', zorder=10)
    legend_patches.append(Patch(color=coords["color"], label=label))


cb = plt.colorbar(max_plot, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
cb.set_label('mg/m³')


ax.legend(handles=legend_patches, loc='lower left', fontsize=9, title="Spring Bloom (SB) Zones")



proj = ccrs.PlateCarree()


bloom_regions = {
    'Spring Bloom 1': [-58.2, -43.97, -65.32, -62.08],
    'Spring Bloom 2': [161.24, 179.85, -78.12, -74.11],
    'Spring Bloom 3': [14.526, 28.820, -70.42, -67.18]
}


positions = ['upper left', 'lower right', 'upper right', ]



for (name, bounds), loc in zip(bloom_regions.items(), positions):
    inset_ax = inset_axes(ax, width="35%", height="35%", loc=loc,
                          axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
    
    w, e, s, n = bounds
    inset_ax.set_extent([w, e, s, n], crs=proj)
    inset_ax.coastlines(resolution='50m', color='black')
    inset_ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    inset_ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

   
    inset_ax.pcolormesh(lon2d, lat2d, chl_max, transform=proj,
                        cmap='YlGn', norm=norm_chl, shading='auto')

    
    rect_lons = [w, e, e, w, w]
    rect_lats = [s, s, n, n, s]
    inset_ax.plot(rect_lons, rect_lats, transform=proj,
                  color='black', linewidth=1.2, linestyle='-', zorder=10)

    inset_ax.set_title(name, fontsize=8)





plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(f"{output_dir}/chl_maximum_map.png", dpi=300)
plt.close()
print("PLOT SAVED! :)")

    
      
#%% DISPLAY OF ALL DATA (chl) with log-scaled colorbar but original data

# Choose year by year to waste less memory!! :)
start_month = 24   
end_month = 36   

# Clean values for log scale: avoid zeros or negatives
chl[chl <= 0] = np.nan  # log scale cannot handle 0 or negative values

num_plots = end_month - start_month
cols = 3
rows = int(np.ceil(num_plots / cols))

fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                        subplot_kw={'projection': ccrs.SouthPolarStereo()})

axs = axs.flatten()


norm_chl = colors.LogNorm(vmin=0.01, vmax=40)

for plot_idx, data_idx in enumerate(range(start_month, end_month)):
    ax = axs[plot_idx]
    date_str = dates_chl[data_idx].strftime('%Y-%m')

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.set_title(f'{date_str}', fontsize=9)
    ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    ax.coastlines(resolution='50m', color='black')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    chl_plot = ax.pcolormesh(lon2d, lat2d, chl[data_idx, :, :], transform=ccrs.PlateCarree(),
                             cmap='YlGn', norm=norm_chl, shading='auto')


for k in range(num_plots, len(axs)):
    fig.delaxes(axs[k])


cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(chl_plot, cax=cbar_ax)
cbar.set_label('Chlorophyll [mg/m³]')
cbar.set_ticks([0.01, 0.1, 1, 10])
cbar.set_ticklabels(['0.01', '0.1', '1', '10'])

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
plt.savefig(f"{output_dir}/chl__logscale_monthly_grid_{start_month}-{end_month}.png", dpi=200)
plt.show()   
print("PLOT SAVED! :)")

# -*- coding: utf-8 -*-
"""
Created on Apr 29, 2025
@author: Sergi Gonzalez Fajardo
"""
#%% ------ INITIAL STUDY OF THE AREA FOR THE RESEARCH OF SPRING BLOOMS ON ANTARCTICA (SST)----- %%#
#%% LIBRARIES
import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from datetime import datetime, timedelta

import os



#%% LOAD DATA, EXTRACT and ADAPT VARIABLES

sst_path = "C:/Users/sergo/OneDrive - Danmarks Tekniske Universitet/MCS Ocean Engineering/2nd semester/25340 Digital Ocean/Final Project/DATA/sst_2021_0601_2022.nc"

output_dir = "C:/Users/sergo/OneDrive - Danmarks Tekniske Universitet/MCS Ocean Engineering/2nd semester/25340 Digital Ocean/Final Project/output_visualizations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


ds_sst = nc.Dataset(sst_path)


time_sst = ds_sst.variables['time'][:]
lon = ds_sst.variables['longitude'][:]
lat = ds_sst.variables['latitude'][:]
sst = ds_sst.variables['sstAnom'][:]  # (time, lat, lon)


# chatgpt help to reduce the resolution of data to dont waste ram on plot generation
step = 4  # Increase this for lower resolution, e.g., 2, 4, 8...

dates_sst = [datetime(1970,1,1) + timedelta(seconds=t) for t in time_sst]
dates_sst_num = mdates.date2num(dates_sst)

sst_lowres = sst[:, ::step, ::step]
lat_lowres = lat[::step]
lon_lowres = lon[::step]
lon2d_lowres, lat2d_lowres = np.meshgrid(lon_lowres, lat_lowres)

#%%

start_month = 0   
end_month = 6  

num_plots = end_month - start_month
cols = 3
rows = int(np.ceil(num_plots / cols))

fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                        subplot_kw={'projection': ccrs.SouthPolarStereo()})

axs = axs.flatten()
norm_sst = colors.Normalize(vmin=-2, vmax=5) 


for plot_idx, data_idx in enumerate(range(start_month, end_month)):
    ax = axs[plot_idx]
    date_str = dates_sst[data_idx].strftime('%Y-%m')

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.set_title(f'{date_str}', fontsize=9)
    ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    ax.coastlines(resolution='50m', color='black')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    sst_plot = ax.pcolormesh(lon2d_lowres, lat2d_lowres, sst_lowres[data_idx, :, :],
                             transform=ccrs.PlateCarree(),
                             cmap='turbo', norm=norm_sst, shading='auto')

for k in range(num_plots, len(axs)):
    fig.delaxes(axs[k])



cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(sst_plot, cax=cbar_ax)
cbar.set_label('Sea Surface Temperature [ºC]')



plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
plt.savefig(f"{output_dir}/sst_monthly_grid_1-6_2021.png", dpi=150)
plt.show()
print("PLOT SAVED! :)")
