% Load SST and Chlorophyll-a NetCDF files
sst_file = 'SST_Monthly_Spring_Bloom_2.nc';
chl_file = 'Ch_Monthly_Spring_Bloom_2.nc';

% Read time and data from SST file
sst_time_sec = ncread(sst_file, 'time');
sst_data = ncread(sst_file, 'sstAnom');

% Read time and data from Chlorophyll file
chl_time_sec = ncread(chl_file, 'time');
chl_data = ncread(chl_file, 'chlorophyll');

% Convert time assuming it's "days since 1970-01-01"
time_ref = datetime(1970,1,1,0,0,0,'TimeZone','UTC');
sst_dates = time_ref + seconds(sst_time_sec);
chl_dates = time_ref + seconds(chl_time_sec);

% Average over spatial dimensions (lat, lon)
sst_ts = squeeze(mean(sst_data, [1, 2], 'omitnan'));
chl_ts = squeeze(mean(chl_data, [1, 2], 'omitnan'));

% Plot time series
figure;
yyaxis left
plot(sst_dates, sst_ts, '-o', 'Color', 'g');
ylabel('SST Anomaly (°C)');
ylim([-2, 2]);

yyaxis right
plot(chl_dates, chl_ts, '-s', 'Color', 'r');
ylabel('Chlorophyll-a (mg/m^3)');

xlabel('Date');
title('Monthly Time Series: SST Anomaly and Chlorophyll-a');
legend('SST Anomaly', 'Chlorophyll-a');
grid on;
