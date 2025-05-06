% Load SST data
sst_file = 'SST_Monthly_Spring_Bloom_3.nc';
sst = ncread(sst_file, 'sstAnom');         % [lon x lat x time]
lon = ncread(sst_file, 'longitude');   % [lon]
lat = ncread(sst_file, 'latitude');    % [lat]
time = ncread(sst_file, 'time');       % [time]

% Convert time units to datetime
time_units = ncreadatt(sst_file, 'time', 'units');
time_origin = datetime(1970,1,1,0,0,0);  % Assume "seconds since 1970-01-01"
sst_dates = time_origin + seconds(time);

% Find index for January 2022
target_date = datetime(2022,1,1);
[~, idx] = min(abs(sst_dates - target_date));  % Find nearest match

% Extract and plot SST anomaly for January 2022
sst_slice = sst(:,:,idx)';

figure;
imagesc(lon, lat, sst_slice);
set(gca, 'YDir', 'normal');
colormap(parula);
colorbar;
caxis([-2 2]);  % Adjust if needed
title(['SST Anomaly - ', datestr(sst_dates(idx), 'mmmm yyyy')]);
xlabel('Longitude');
ylabel('Latitude');
