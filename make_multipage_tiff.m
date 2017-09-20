



% Create multipage tiff stack of large light-sheet data

%% set parameters
new_fname      = 'm1-cre-c1-crop-scale50.tif';
increment      = 2;
starting_image = 1; % 1 is first file in directory
max_value      = 13000; % everything between 0 and max_value will be scaled to be between 0 and 2^bits
bits           = 8;
rotate_deg     = 90; % positive values are Counter-Clockwise, negative values are Clockwise
crop           = 0;
crop_coord     = [0, 1353, 8728, 3704]; % xmin, ymin, width, height
rescale        = 0.5;
max_files      = 427; % remove this?

% set save options
options.overwrite = true;
options.append    = true;
options.big       = true; % Use 64 bit addressing and allows for files > 4GB

%%
scale_factor   = max_value/(2^bits);

% get file names
path2data = '/media/greg/data/hist/';
[file_path, file_name, ~] = fileparts(uigetdir(path2data, 'Select directory with tiff images'));
tiff_dir = [file_path filesep file_name];
tiff_files = dir([tiff_dir filesep '*tif']);

% create stack directory
mkdir([tiff_dir filesep 'multi-page-tiffs'])

% iterate through all images
progressbar('images processed')
% for k = starting_image:increment:length(tiff_files)

for k = starting_image:increment:max_files
    progressbar(k/max_files)
    % load image
    img = imread([tiff_dir filesep tiff_files(k).name], 'tif');
    
    % scale image to be between 0 and 2^bits
    img = img/scale_factor;
    
    % convert to 8-bit
    img = uint8(img);
    
    % rotate image
    img = imrotate(img, rotate_deg);
    
    % crop image
    if crop
        img = imcrop(img, crop_coord);
    end
    
    % resize image
    img = imresize(img, rescale);
    
    % add to tiff stack
    saveastiff(img, [tiff_dir filesep 'multi-page-tiffs' filesep new_fname], options);
    
end

progressbar(1)
disp('#### COMPLETE ####')


