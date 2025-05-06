import netCDF4 as netcdf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

nc   = netcdf.Dataset("bloom1_final2.nc", "r")


plotacf=False
plotPred=False
plotccf=False
multiplot=True
model2=False


print(nc.dimensions)


chl = nc.variables['chlorophyll'][:]  # shape: (time, lat, lon)
time = nc.variables['time'][:]        # shape: (time,)
lat = nc.variables['latitude'][:]  # shape: (time, lat, lon)
lon = nc.variables['longitude'][:]   


print(time)



# Convert time units if needed
# For example: convert to datetime if time units are "days since 2003-01-01"
from netCDF4 import num2date

time_units = nc.variables['time'].units
time_calendar = nc.variables['time'].calendar if 'calendar' in nc.variables['time'].ncattrs() else 'standard'
dates =[datetime.datetime.fromtimestamp(ts) for ts in time]

print(dates)
# Average chlorophyll over lat/lon to get time series
chl_mean = np.nanmean(chl, axis=(1, 2))  # shape: (time,)


"""gaps = [dates[i+1] - dates[i] for i in range(len(dates)-1)]

delta_seconds = [diff.total_seconds() for diff in gaps]
plt.plot(delta_seconds)
plt.ylabel("Seconds between timestamps")
plt.title("Time Gaps Between Points")
plt.show()



expected_step = gaps[0]  # assumes first step is expected
for i, gap in enumerate(gaps):
    if gap != expected_step:
        print(f"Gap at index {i}: {gap} instead of {expected_step}")
# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot( chl_mean, label='Mean Chlorophyll')
plt.xlabel('Time')
plt.ylabel('Chlorophyll Concentration')
plt.title('Time Series of Mean Chlorophyll (2003–2022)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()"""

"""
years = np.array([d.year for d in dates])

unique_years = np.unique(years)
max_lats = []
max_lons = []
max_years = []

for y in unique_years:
    indices = np.where(years == y)[0]
    if len(indices) == 0:
        continue
    chl_year = chl[indices, :, :]  # shape: (T, lat, lon)
    max_idx = np.unravel_index(np.argmax(chl_year), chl_year.shape)
    _, lat_idx, lon_idx = max_idx
    max_lats.append(lat[lat_idx])
    max_lons.append(lon[lon_idx])
    max_years.append(y)

plt.figure(figsize=(10, 6))
plt.scatter(max_lons, max_lats, c=max_years, cmap='viridis', s=100, edgecolor='k')
plt.colorbar(label='Year')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Annual Maximum Chlorophyll Locations")
plt.grid(True)
plt.show()

weeks = np.array([d.isocalendar()[:2] for d in dates])  # shape (N, 2)

# Unique year-week pairs
_, unique_indices = np.unique(weeks, axis=0, return_index=True)
unique_weeks = weeks[np.sort(unique_indices)]

max_lats = []
max_lons = []
week_labels = []

target_year = 2016
weeks = np.array([d.isocalendar()[:2] for d in dates])  # [(year, week), ...]

# Mask for just the target year
indices = np.where(years == target_year)[0]
chl = chl[indices, :, :]
weeks = weeks[indices]

# Unique (year, week) pairs in target year
_, unique_indices = np.unique(weeks, axis=0, return_index=True)
unique_weeks = weeks[np.sort(unique_indices)]


for y, w in unique_weeks:
    indices = [i for i, (yy, ww) in enumerate(weeks) if yy == y and ww == w]
    if not indices:
        continue
    chl_week = chl[indices, :, :]  # shape (T, lat, lon)
    max_idx = np.unravel_index(np.argmax(chl_week), chl_week.shape)
    _, lat_idx, lon_idx = max_idx
    max_lats.append(lat[lat_idx])
    max_lons.append(lon[lon_idx])
    week_labels.append(f"{y}-W{w:02d}")

plt.figure(figsize=(10, 6))
sc = plt.scatter(max_lons, max_lats, c=range(len(week_labels)), cmap='plasma', s=60)
plt.plot(max_lons, max_lats, color='gray', linestyle='--', alpha=0.5)  # optional trajectory line
plt.colorbar(sc, label='Week Index')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Weekly Max Chlorophyll Locations")
plt.grid(True)
plt.show()"""
plt.rcParams.update({'font.size': 20})  # Set global font size to 14
df=pd.DataFrame({'X': chl_mean})

split_point = -24

df['X_lag_1'] = df['X'].shift(1)

df['X_lag_2'] = df['X'].shift(2)
df['X_lag_3'] = df['X'].shift(3)
df['X_lag_365'] = df['X'].shift(12)

df['X_lag_13'] = df['X'].shift(13)
#dffilled=df.interpolate(method='linear')  # or use df.fillna(method='ffill'), etc.

df = df.iloc[13:]


dffilled=df.fillna(0)
dffilled=dffilled.dropna()
assert not dffilled.isnull().any().any(), "There are still NaNs in the DataFrame!"
X = dffilled[['X_lag_1','X_lag_365','X_lag_13']]
y = dffilled['X']
print(y)
print(len(dates))



from sklearn.metrics import mean_squared_error


X_train = X.iloc[:split_point]
y_train = y.iloc[:split_point]
X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

from sklearn.linear_model import LinearRegression
from scipy.stats import t

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

alpha = 0.05
n = len(X_train)
p = X_train.shape[1]
y_pred_train = model.predict(X_train)
residuals = y_train - y_pred_train
mse = mean_squared_error(y_train, y_pred_train)

X_test_np = np.hstack((np.ones((len(X_test), 1)), X_test.values))  # add intercept
X_train_np = np.hstack((np.ones((len(X_train), 1)), X_train.values))
XtX_inv = np.linalg.inv(X_train_np.T @ X_train_np)
se = np.sqrt(np.sum((X_test_np @ XtX_inv) * X_test_np, axis=1) * mse)
tval = t.ppf(1 - alpha/2, df=n - p - 1)

# Confidence intervals
lower = y_pred_test - tval * se
upper = y_pred_test + tval * se



if plotPred:

    # Predict
    plt.figure(figsize=(12, 6))
    """
    plt.plot(y.index,y_pred, label='Mean Chlorophyll predicted')
    plt.plot(y.index,y, label='Mean Chlorophyll')"""
    plt.plot(dates[13:split_point], y_train, label="Training data", color='gray')

    plt.plot(dates[13:split_point],y_pred_train,label='Predicted (train)',color='orange')
    # Plot actual test values
    plt.plot(dates[split_point:], y_test, label="Actual (test)", color='blue')

    # Plot predicted test values
    plt.plot(dates[split_point:], y_pred_test, label="Predicted (test)", color='red')

    # Fill uncertainty
    plt.fill_between(dates[split_point:], lower, upper, color='red', alpha=0.3, label="95% Confidance Interval")
    for dt in dates:
        if dt.month == 1 and dt.day == 1:
            plt.axvline(x=dt, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Chlorophyll Concentration')
    plt.title('Time Series of Mean Chlorophyll (2003–2022)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if plotacf:
    from statsmodels.graphics.tsaplots import plot_acf
    max_lag=50
    plot_acf(y, lags=max_lag)  # Change lags to desired max lag
    for index in range(max_lag):
        if index%12==0 :
            plt.axvline(x=index, color='black', linestyle='--', alpha=0.3)
            plt.text(index, 0.90, f'{index//12}y', ha='center', va='bottom', fontsize=15, rotation=0)
    plt.xlabel("Lag (in months) ")
    plt.show()

if multiplot:
    from statsmodels.tsa.stattools import ccf
    nc2  = netcdf.Dataset("temp1.nc", "r")
    print("iitit")
    temp = nc2.variables['sst'][:]  # shape: (time, lat, lon)
    tempMean = np.nanmean(temp, axis=(1, 2))  # shape: (time,)
    df2=pd.DataFrame({'X': tempMean})
    df2['temp13'] = df2['X'].shift(13)
    df2['temp1'] = df2['X'].shift(1)

    dffilled2=df2.ffill()  # or use df.fillna(method='ffill'), etc.

    dffilled2=dffilled2.dropna()  
    assert not dffilled2.isnull().any().any(), "There are still NaNs in the DataFrame!"

    temp=dffilled2['temp13'].values

    time_units = nc2.variables['time'].units
    time_calendar = nc2.variables['time'].calendar if 'calendar' in nc.variables['time'].ncattrs() else 'standard'
    dates =[datetime.datetime.fromtimestamp(ts) for ts in time]

    if plotccf:
        ccf_values = ccf(y, temp)
        plt.stem(range(len(ccf_values)), ccf_values)
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation between sea-surface temperature and Chlorophyl Concentration')
        for lag in range(12, len(ccf_values), 12):
            plt.axvline(x=lag, color='gray', linestyle='--', alpha=0.5)
            plt.text(lag, max(ccf_values)*0.95, f'{lag//12}y', ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.show()




    if model2:
        dffilled['temp']=dffilled2['temp1']
        X2 = dffilled[['X_lag_1','X_lag_365','X_lag_13','temp']]
        y2 = dffilled['X']

        X_train2 = X2.iloc[:split_point]
        y_train2 = y2.iloc[:split_point]
        X_test2 = X2.iloc[split_point:]
        y_test2 = y2.iloc[split_point:]
        

        model2 = LinearRegression() 
        model2.fit(X_train2, y_train2)
        print(model2.coef_)         # prints learned coefficients (for linear models)

        y_pred_test2 = model2.predict(X_test2)
        alpha = 0.05
        n = len(X_train2)
        p = X_train2.shape[1]
        y_pred_train2 = model2.predict(X_train2)
        residuals = y_train2 - y_pred_train2
        mse = mean_squared_error(y_train2, y_pred_train2)

        X_test_np = np.hstack((np.ones((len(X_test2), 1)), X_test2.values))  # add intercept
        X_train_np = np.hstack((np.ones((len(X_train2), 1)), X_train2.values))
        XtX_inv = np.linalg.inv(X_train_np.T @ X_train_np)
        se = np.sqrt(np.sum((X_test_np @ XtX_inv) * X_test_np, axis=1) * mse)
        tval = t.ppf(1 - alpha/2, df=n - p - 1)

        # Confidence intervals
        lower = y_pred_test - tval * se
        upper = y_pred_test + tval * se

  
        plt.figure(figsize=(12, 6))
        plt.plot(dates[13:split_point], y_train, label="Training data", color='gray')

        plt.plot(dates[13:split_point],y_pred_train2,label='Predicted with temperature(train)',color='purple')
        plt.plot(dates[13:split_point],y_pred_train,label='Predicted without temperature(train)',color='orange',linestyle='-.')

        # Plot actual test values
        plt.plot(dates[split_point:], y_test, label="Actual (test)", color='blue')

        # Plot predicted test values
        plt.plot(dates[split_point:], y_pred_test2, label="Predicted with temperature (test)", color='green')
        plt.plot(dates[split_point:], y_pred_test, label="Predicted without temperature (test)", color='red', linestyle='-.')


        # Fill uncertainty
        plt.fill_between(dates[split_point:], lower, upper, color='red', alpha=0.3, label="95% Confidance Interval")
        for dt in dates:
            if dt.month == 1 and dt.day == 1:
                plt.axvline(x=dt, color='black', linestyle='--', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Chlorophyll Concentration')
        plt.title('Time Series of Mean Chlorophyll (2003–2022)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    nc3   = netcdf.Dataset("seaice_concentration_2003_2024_antarctica.nc", "r")
    print(nc3)
    print(nc3.variables)


    time_units = nc.variables['time'].units
    time_calendar = nc.variables['time'].calendar if 'calendar' in nc.variables['time'].ncattrs() else 'standard'
    datesseaIce =[datetime.datetime.fromtimestamp(ts) for ts in time]

    sea_ice_concentration = nc3.variables['cdr_seaice_conc_monthly'][:]  # shape (264, 332, 316)    #print(dates[:20])
    print(sea_ice_concentration)
    sea_ice_concentration=sea_ice_concentration[:len(dates)][:][:]

    lat_bounds = (-78.12, -74.11)  
    lon_bounds = (161.24, 179.85)
    x = nc3.variables['xgrid'][:]  # Replace with the correct variable name
    y2 = nc3.variables['ygrid'][:]
    from pyproj import Proj
    proj_polar = Proj(proj="stere", lat_0=-90, lat_ts=-70, lon_0=0, x_0=0, y_0=0, a=6378273, b=6356889.449, units="m")


    # Create a meshgrid of x and y
    seaice_x_grid, seaice_y_grid = np.meshgrid(x, y2)

    # Convert x, y to longitude and latitude
    seaice_lon_grid, seaice_lat_grid = proj_polar(seaice_x_grid, seaice_y_grid, inverse=True)


    model3=True
    plotccf2=True

    def calculate_average_sea_ice(lat_grid, lon_grid, sea_ice_concentration, lat_bounds, lon_bounds, time_index):
        """
        Extracts a subset of data from lat_grid, lon_grid and calculates the average
        of sea_ice_concentration over the specified region for a specific time index.

        Parameters:
            lat_grid (ndarray): 2D array of latitude values.
            lon_grid (ndarray): 2D array of longitude values.
            sea_ice_concentration (ndarray): 3D array of sea ice concentration (time, lat, lon).
            lat_bounds (tuple): Latitude bounds as (min_lat, max_lat).
            lon_bounds (tuple): Longitude bounds as (min_lon, max_lon).
            time_index (int): Time index to extract data for.

        Returns:
            float: Average sea ice concentration over the specified region.
        """
        # Extract the subset of data within the latitude and longitude bounds
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # Create a mask for the region of interest
        region_mask = (
            (lat_grid >= lat_min) & (lat_grid <= lat_max) &
            (lon_grid >= lon_min) & (lon_grid <= lon_max)
        )

        # Apply the mask to the sea ice concentration data for the given time index
        region_data = sea_ice_concentration[time_index, :, :][region_mask]

        # Calculate and return the average, ignoring NaN values
        return np.nanmean(region_data)

    

    
    avg_sea_ice_conc=[calculate_average_sea_ice(seaice_lat_grid, seaice_lon_grid, sea_ice_concentration, lat_bounds, lon_bounds, t)for t in range(len(dates))]
    print(len(avg_sea_ice_conc))
    array_avg_sea_ice=pd.DataFrame({'si':avg_sea_ice_conc})

    print(array_avg_sea_ice)
    array_avg_sea_ice['si1']=array_avg_sea_ice['si'].shift(1)
    array_avg_sea_ice['si13']=array_avg_sea_ice['si'].shift(13)
    array_avg_sea_ice=array_avg_sea_ice.dropna()

    if plotccf2:
        ccf_values = ccf(y, array_avg_sea_ice['si13'])
        plt.stem(range(len(ccf_values)), ccf_values)
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation between sea-ice concentration and Chlorophyl Concentration')
        for lag in range(12, len(ccf_values), 12):
            plt.axvline(x=lag, color='gray', linestyle='--', alpha=0.5)
            plt.text(lag, max(ccf_values)*0.95, f'{lag//12}y', ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.show()

    if model3:
        dffilled['seaIceConc']=array_avg_sea_ice['si1']
        X2 = dffilled[['X_lag_1','X_lag_365','X_lag_13','seaIceConc']]
        y2 = dffilled['X']

        X_train2 = X2.iloc[:split_point]
        y_train2 = y2.iloc[:split_point]
        X_test2 = X2.iloc[split_point:]
        y_test2 = y2.iloc[split_point:]
        
        #print(X2.head)
        
        model2 = LinearRegression() 
        model2.fit(X_train2, y_train2)
        y_pred_test2 = model2.predict(X_test2)
        alpha = 0.05
        n = len(X_train2)
        p = X_train2.shape[1]
        y_pred_train2 = model2.predict(X_train2)
        residuals = y_train2 - y_pred_train2
        mse = mean_squared_error(y_train2, y_pred_train2)

        X_test_np = np.hstack((np.ones((len(X_test2), 1)), X_test2.values))  # add intercept
        X_train_np = np.hstack((np.ones((len(X_train2), 1)), X_train2.values))
        XtX_inv = np.linalg.inv(X_train_np.T @ X_train_np)
        se = np.sqrt(np.sum((X_test_np @ XtX_inv) * X_test_np, axis=1) * mse)
        tval = t.ppf(1 - alpha/2, df=n - p - 1)

        # Confidence intervals
        lower = y_pred_test - tval * se
        upper = y_pred_test + tval * se

  
        plt.figure(figsize=(12, 6))
        plt.plot(dates[13:split_point], y_train, label="Training data", color='gray')

        plt.plot(dates[13:split_point],y_pred_train2,label='Predicted with sea ice(train)',color='purple')
        plt.plot(dates[13:split_point],y_pred_train,label='Predicted without sea ice(train)',color='orange',linestyle='-.')

        # Plot actual test values
        plt.plot(dates[split_point:], y_test, label="Actual (test)", color='blue')

        # Plot predicted test values
        plt.plot(dates[split_point:], y_pred_test2, label="Predicted with sea ice (test)", color='cyan')
        plt.plot(dates[split_point:], y_pred_test, label="Predicted without sea ice (test)", color='green', linestyle='-.')


        # Fill uncertainty
        plt.fill_between(dates[split_point:], lower, upper, color='red', alpha=0.3, label="95% Confidance Interval")
        for dt in dates:
            if dt.month == 1 and dt.day == 1:
                plt.axvline(x=dt, color='black', linestyle='--', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Chlorophyll Concentration')
        plt.title('Time Series of Mean Chlorophyll (2003–2022)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
