% Load SST data
sst_file = 'SST_Monthly_Spring_Bloom_3.nc';
sst = ncread(sst_file, 'sstAnom');        % [lon x lat x time]
lon = ncread(sst_file, 'longitude');  % [lon]
lat = ncread(sst_file, 'latitude');   % [lat]
time = ncread(sst_file, 'time');      % [time]

% Convert time to datetime
time_units = ncreadatt(sst_file, 'time', 'units');  % e.g. "seconds since 1970-01-01T00:00:00Z"
time_origin = datetime(1970,1,1);  % from units
sst_dates = time_origin + seconds(time);

% Create figure
figure;
colormap(parula);  % choose color map
n_months = 12;
start_index = 49;  % Start at month 13 (January 2018)

for i = 1:n_months
    subplot(3, 4, i);
    
    % Extract SST at time (start_index + i - 1)
    sst_slice = sst(:,:,start_index + i - 1)';  % Transpose for orientation
    
    % Plot with imagesc
    imagesc(lon, lat, sst_slice);
    set(gca, 'YDir', 'normal');
    axis tight;
    title(datestr(sst_dates(start_index + i - 1), 'mmmm yyyy'));
    caxis([-2 2]);  % Adjust as needed
end

% Shared colorbar
h = colorbar('Position', [0.92 0.11 0.02 0.77]);
ylabel(h, 'SST Anomaly (°C)');
sgtitle('Monthly SST Anomaly - 2021');
