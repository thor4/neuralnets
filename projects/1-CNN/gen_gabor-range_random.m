%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

% this will generate gabors with a range of 3 contrasts & 50 tilts 
% randomly chosen between 0.1 and 3 degrees yielding 100 images per class, 
% 50*3= 150 total categories, 150*100*2(counter/clock) = 30k total images 
% Eventually the model should break down if your tilt is low enough

%match all aspects of gabors from experiment
width = 169 ; % p.stimSize in pixels 
nGaussianSDs = [] ; %default in exp (6)
contrasts = [.3, .45, 1];  % 30%, 45% and 100% contrast
% contrast = 1; %middle contrast from experiment, dropping other 2
noise = 1; %same as exp
gratingPeriod = 0.5; %same as exp
gratingPeriodUnits = 'sd'; %same as exp
orientation = pi/4; %radians (45 deg), add to / subtract from this to get classes
% num_tilts = 50; %define how many tilts to randomly generate
rng(0,'twister'); %init random number generator to make results repeatable
%create vector of num_tilts random values. Use rand function to draw 
%values from uniform distribution in open interval, (tilt_start,tilt_end)
tilt_start = 0.05;
tilt_end = 2; 
% r = (tilt_end-tilt_start).*rand(num_tilts,1) + tilt_start;
% r_range = [min(r) max(r)]; % confirm tilt range is in open interval
% %visualize to ensure relatively uniform:
% histogram(r, [tilt_start,(tilt_end-tilt_start)/2+tilt_start,tilt_end]) 
% tilt_deg = 3/50:3/50:3; %50 tilts tiled from 0.06,3
% %to make it harder, change that 3 to 1 or even .5, this will make it harder
% tilts = r .* (pi/180); %degrees->radians: *(pi/180)
% sort(r)'
%removing the 0.05 tilt and trying again, including it led to chance-level performance [05/11/21]
%p.tilts = [middle_tilt/2, middle_tilt, 2*middle_tilt];
%Middle tilt = 2.26 # supposed to yield 75% accuracy, this is in degrees
% counter_tilts = orientation - tilts; clock_tilts = orientation + tilts; 
num_gabors = 3500; %define number of gabors to make for each contrast
% 3500 gabors * 3 contrasts * 2 classes = 21,000 gabors
n_train_gabor = floor(num_gabors*(2/3)); %number of training gabor images
n_valid_gabor = floor(0.8*(num_gabors-n_train_gabor)); %number of validation gabor images
n_test_gabor = num_gabors - n_train_gabor - n_valid_gabor;
imageN = 0; %initialize image counter
%UPDATE WHERE IMAGES WILL BE STORED TO SAVE ON OSF
tic
for contrastN = 1:length(contrasts)
    contrast = contrasts(contrastN); %current contrast
    %move tilts down to 1:num gabors to generate random each time, now
    %total images = num_gabors * 3 contrasts * 2 classes
    %run a test with small num_gabors and new folders
    for i=1:num_gabors
        tilt = ((tilt_end-tilt_start)*rand(1,1) + tilt_start)*(pi/180);
        counter_tilt=orientation - tilt; clock_tilt=orientation + tilt; 
        gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
        gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
        if (ismember(i,(1:n_train_gabor)))
            imageN = imageN + 1;
            counter_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\train_range_rand\\cclock\\cclock%d.png',imageN);
            imwrite(gaborPatch_counter,counter_label);
            clock_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\train_range_rand\\clock\\clock%d.png',imageN);
            imwrite(gaborPatch_clock,clock_label);
        elseif (ismember(i,(n_train_gabor+1:n_train_gabor+n_valid_gabor))) 
            imageN = imageN + 1;
            counter_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\valid_range_rand\\cclock\\cclock%d.png',imageN);
            imwrite(gaborPatch_counter,counter_label);
            clock_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\valid_range_rand\\clock\\clock%d.png',imageN);
            imwrite(gaborPatch_clock,clock_label);
        else
            imageN = imageN + 1;
            counter_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\test_range_rand\\cclock\\cclock%d.png',imageN);
            imwrite(gaborPatch_counter,counter_label);
            clock_label = sprintf('G:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\training\\test_range_rand\\clock\\clock%d.png',imageN);
            imwrite(gaborPatch_clock,clock_label);
        end
    end
end
toc

%generate and save gabor images of size width+1 x width+1 (170x170 .png's)

%create  base case (45 deg) and visualize difference

gaborPatch = uint8(makeGaborPatch(width,[],contrast,noise,...
    gratingPeriod,gratingPeriodUnits,orientation));

figure(1), clf
subplot(1,3,1)
imshow(gaborPatch_counter)
tcounter = sprintf('Counter %.2f',counter_tilt);
title(tcounter,'FontSize',20)
subplot(1,3,2)
imshow(gaborPatch)
t = sprintf('Baseline %.2f',orientation);
title(t,'FontSize',20)
subplot(1,3,3)
imshow(gaborPatch_clock)
tclock = sprintf('Clock %.2f',clock_tilt);
title(tclock,'FontSize',20)

imwrite(gaborPatch,'base.png');


export_fig('gabors','-png','-transparent'); %save transparent pdf in pwd

%now generate 100 gabors for each class to be used in the transfer learning
%tensorflow code