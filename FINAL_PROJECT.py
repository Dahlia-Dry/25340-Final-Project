# -*- coding: utf-8 -*-
"""
Created on Apr 29, 2025
@author: Sergi Gonzalez Fajardo
"""
#%% ------ INITIAL STUDY OF THE AREA FOR THE RESEARCH OF SPRING BLOOMS ON ANTARCTICA ----- %%#
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

# CHL

ds_chl = nc.Dataset(chl_path)


time_chl = ds_chl.variables['time'][:]
lon = ds_chl.variables['longitude'][:]
lat = ds_chl.variables['latitude'][:]
chl = ds_chl.variables['chlorophyll'][:]  # (time, lat, lon)

chl_log = np.log10(chl)
# reduce resolution but having better wuality generation
chl= chl.astype(np.float16)

# Time adaptation seconds to month and year
dates_chl = [datetime(1970,1,1) + timedelta(seconds=t) for t in time_chl]
dates_chl_num = mdates.date2num(dates_chl)
# Meshgrids

projection_in = ccrs.PlateCarree()
lon2d, lat2d = np.meshgrid(lon, lat)

#%% MAXIMUM CONCENTRATION

# Define vmin and vmax
vmin_chl, vmax_chl = 0, 60
chl_max = np.nanmax(chl, axis=0)
norm_chl = colors.Normalize(vmin_chl,vmax_chl)

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
ax.set_title('Maximum Chlorophyll Concentration [mg/m³] (2017–2022)', fontsize=14)
ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
ax.coastlines(resolution='50m', color='black')
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# Plot the chlorophyll max values
max_plot = ax.pcolormesh(lon2d, lat2d, chl_max, transform=ccrs.PlateCarree(),
                         cmap='YlGn', norm=norm_chl, shading='auto')

# Add rectangles for the spring bloom zones
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

# Add colorbar
cb = plt.colorbar(max_plot, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
cb.set_label('mg/m³')

# Add legend for rectangles
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, title="Spring Bloom (SB) Zones")


# Define projection for inset
proj = ccrs.PlateCarree()

# Define bloom regions as [W, E, S, N]
bloom_regions = {
    'Spring Bloom 1': [-58.2, -43.97, -65.32, -62.08],
    'Spring Bloom 2': [161.24, 179.85, -78.12, -74.11],
    'Spring Bloom 3': [14.526, 28.820, -70.42, -67.18]
}

# Set inset positions
positions = ['upper left', 'lower right', 'upper right', ]

# Plot each inset
for (name, bounds), loc in zip(bloom_regions.items(), positions):
    inset_ax = inset_axes(ax, width="35%", height="35%", loc=loc,
                          axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
    
    w, e, s, n = bounds
    inset_ax.set_extent([w, e, s, n], crs=proj)
    inset_ax.coastlines(resolution='50m', color='black')
    inset_ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    inset_ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

    # Plot chlorophyll
    inset_ax.pcolormesh(lon2d, lat2d, chl_max, transform=proj,
                        cmap='YlGn', norm=norm_chl, shading='auto')

    # Draw rectangle for clarity
    rect_lons = [w, e, e, w, w]
    rect_lats = [s, s, n, n, s]
    inset_ax.plot(rect_lons, rect_lats, transform=proj,
                  color='black', linewidth=1.2, linestyle='-', zorder=10)

    inset_ax.set_title(name, fontsize=8)




# Save and close
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(f"{output_dir}/chl_maximum_map.png", dpi=300)
plt.close()
print("PLOT SAVED! :)")

#%% MAXIMUM  LOG CONCENTRATION
# Define vmin and vmax

chl_log = np.log10(chl)
vmin_chl, vmax_chl = 0, 60
chl_max = np.nanmax(chl_log, axis=0)
norm_chl = colors.Normalize(vmin_chl,vmax_chl)

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
ax.set_title('Maximum log10 Chlorophyll Concentration [mg/m³] (2017–2022)', fontsize=14)
ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
ax.coastlines(resolution='50m', color='black')
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# Plot the chlorophyll max values
max_plot = ax.pcolormesh(lon2d, lat2d, chl_max, transform=ccrs.PlateCarree(),
                         cmap='YlGn', norm=norm_chl, shading='auto')

# Add rectangles for the spring bloom zones
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

# Add colorbar
cb = plt.colorbar(max_plot, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
cb.set_label('mg/m³')

# Add legend for rectangles
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, title="Spring Bloom (SB) Zones")


# Define projection for inset
proj = ccrs.PlateCarree()

# Define bloom regions as [W, E, S, N]
bloom_regions = {
    'Spring Bloom 1': [-58.2, -43.97, -65.32, -62.08],
    'Spring Bloom 2': [161.24, 179.85, -78.12, -74.11],
    'Spring Bloom 3': [14.526, 28.820, -70.42, -67.18]
}

# Set inset positions
positions = ['upper left', 'lower right', 'upper right', ]

# Plot each inset
for (name, bounds), loc in zip(bloom_regions.items(), positions):
    inset_ax = inset_axes(ax, width="35%", height="35%", loc=loc,
                          axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
    
    w, e, s, n = bounds
    inset_ax.set_extent([w, e, s, n], crs=proj)
    inset_ax.coastlines(resolution='50m', color='black')
    inset_ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    inset_ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

    # Plot chlorophyll
    inset_ax.pcolormesh(lon2d, lat2d, chl_max, transform=proj,
                        cmap='YlGn', norm=norm_chl, shading='auto')

    # Draw rectangle for clarity
    rect_lons = [w, e, e, w, w]
    rect_lats = [s, s, n, n, s]
    inset_ax.plot(rect_lons, rect_lats, transform=proj,
                  color='black', linewidth=1.2, linestyle='-', zorder=10)

    inset_ax.set_title(name, fontsize=8)




# Save and close
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(f"{output_dir}/chl_log_maximum_map.png", dpi=300)
plt.close()
print("PLOT SAVED! :)")





# we apply log to have better VISIBLE results
chl_log = np.log10(chl)

chl_max = np.nanmax(chl_log, axis=0)
norm_chl = colors.Normalize()

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
ax.set_title('Maximum Log10 Chlorophyll Concentration [mg/m³] (2017–2022)', fontsize=14)
ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
ax.coastlines(resolution='50m', color='black')
ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# Plot the chlorophyll max values
max_plot = ax.pcolormesh(lon2d, lat2d, chl_max, transform=ccrs.PlateCarree(),
                         cmap='YlGn', norm=norm_chl, shading='auto')

# Add rectangles for the spring bloom zones
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

# Add colorbar
cb = plt.colorbar(max_plot, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
cb.set_label('mg/m³')

# Add legend for rectangles
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, title="Spring Bloom (SB) Zones")


# Define projection for inset
proj = ccrs.PlateCarree()

# Define bloom regions as [W, E, S, N]
bloom_regions = {
    'Spring Bloom 1': [-58.2, -43.97, -65.32, -62.08],
    'Spring Bloom 2': [161.24, 179.85, -78.12, -74.11],
    'Spring Bloom 3': [14.526, 28.820, -70.42, -67.18]
}

# Set inset positions
positions = ['upper left', 'lower right', 'upper right', ]

# Plot each inset
for (name, bounds), loc in zip(bloom_regions.items(), positions):
    inset_ax = inset_axes(ax, width="35%", height="35%", loc=loc,
                          axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
    
    w, e, s, n = bounds
    inset_ax.set_extent([w, e, s, n], crs=proj)
    inset_ax.coastlines(resolution='50m', color='black')
    inset_ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    inset_ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3, linestyle='--')

    # Plot chlorophyll
    inset_ax.pcolormesh(lon2d, lat2d, chl_max, transform=proj,
                        cmap='YlGn', norm=norm_chl, shading='auto')

    # Draw rectangle for clarity
    rect_lons = [w, e, e, w, w]
    rect_lats = [s, s, n, n, s]
    inset_ax.plot(rect_lons, rect_lats, transform=proj,
                  color='black', linewidth=1.2, linestyle='-', zorder=10)

    inset_ax.set_title(name, fontsize=8)




# Save and close
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(f"{output_dir}/chl__log_maximum_map.png", dpi=300)
plt.close()
print("PLOT SAVED! :)")    
      
      
#%% DISPLAY OF ALL DATA

start_month = 48   
end_month = 60   


num_plots = end_month - start_month
cols = 3
rows = int(np.ceil(num_plots / cols))

fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                        subplot_kw={'projection': ccrs.SouthPolarStereo()})

axs = axs.flatten()
norm_chl = colors.Normalize(vmin=-2, vmax=1) 

for plot_idx, data_idx in enumerate(range(start_month, end_month)):
    ax = axs[plot_idx]
    date_str = dates_chl[data_idx].strftime('%Y-%m')

    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax.set_title(f'{date_str}', fontsize=9)
    ax.add_feature(cfeature.LAND, facecolor='darkgray', edgecolor='black')
    ax.coastlines(resolution='50m', color='black')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    chl_plot = ax.pcolormesh(lon2d, lat2d, chl_log[data_idx, :, :], transform=ccrs.PlateCarree(),
                             cmap='YlGn', norm=norm_chl, shading='auto')


for k in range(num_plots, len(axs)):
    fig.delaxes(axs[k])


cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(chl_plot, cax=cbar_ax)
cbar.set_label('log₁₀(Chlorophyll) [mg/m³]')
cbar.set_ticks([-2, -1, 0, 1])
cbar.set_ticklabels(['0.01', '0.1', '1', '10'])

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
plt.savefig(f"{output_dir}/chl__log10_monthly_grid_{start_month}-{end_month}.png", dpi=200)
plt.show()

    
#%% HISTOGRAM CHL SB

spring_bloom_areas = [
    {"name": "Spring Bloom 1", "N": -62.08, "S": -65.32, "W": -58.2, "E": -43.97},
    {"name": "Spring Bloom 2", "N": -74.11, "S": -78.12, "W": 161.24, "E": 179.85},
    {"name": "Spring Bloom 3", "N": -67.18, "S": -70.42, "W": 14.526, "E": 28.820}
]

for bloom in spring_bloom_areas:
    N, S, W, E = bloom["N"], bloom["S"], bloom["W"], bloom["E"]

    # Máscara espacial
    mask = (lat2d >= S) & (lat2d <= N) & (lon2d >= W) & (lon2d <= E)

    # Extraemos los datos dentro de la región para cada tiempo
    chl_bloom = chl[:, mask]  # shape (time, n_pixels)

    # Convertimos a escala logarítmica evitando ceros o negativos
    chl_bloom = np.clip(chl_bloom, 1e-3, None)
    chl_log = np.log10(chl_bloom)

    # Creamos arrays planos para plotting
    chl_values = chl_log.flatten()
    time_values = np.repeat(dates_chl_num, chl_log.shape[1])

    # Filtramos nan
    valid = ~np.isnan(chl_values)
    chl_values = chl_values[valid]
    time_values = time_values[valid]

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(12, 6))
    hb = ax.hexbin(time_values, chl_values, gridsize=100, cmap='YlGn', mincnt=1)

    # Ejes
    ax.set_xlabel("Year")
    ax.set_ylabel("log10(Chlorophyll) [mg/m³]")
    ax.set_title(f"Chlorophyll vs Time in {bloom['name']}")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Colorbar
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Pixel count")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chl_time_series{bloom['name'].replace(' ', '_')}.png", dpi=300)
    plt.show()



    


