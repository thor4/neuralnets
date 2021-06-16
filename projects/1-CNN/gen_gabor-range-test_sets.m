%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

% this will generate 18 sets of gabors, each with a different tilt and
% contrast combo. 500 gabors per contrast/tilt permutation, 18 total sets, 
% 9,000 total gabors

%match all aspects of gabors from experiment
width = 169 ; % p.stimSize in pixels 
nGaussianSDs = [] ; %default in exp (6)
contrasts = [.3, .45, 1];  % 30%, 45% and 100% contrast
noise = 1; %same as exp
gratingPeriod = 0.5; %same as exp
gratingPeriodUnits = 'sd'; %same as exp
orientation = pi/4; %radians (45 deg), add to / subtract from this to get classes
tilts = [.1, .2, .4, .8, 1.6, 3.2].*(pi/180); %degrees->radians: *(pi/180)
%choose which set to generate:
sets = 18; %total number of datasets to generate (# of tilt x contrast permutations)
num_gabors = 500; %define total number of gabors to make for each tilt/contrast permutation
imageN = 0; %initialize image counter

%generate and save gabor images of size width+1 x width+1 (170x170 .png's)
tic
for setN = 1:sets %load each dataset's parameters
    switch setN
        case 1 %set 1
            tilt = tilts(1); contrast = contrasts(1); fname = 's1-t_0.1-c_0.3';
        case 2 %set 2
            tilt = tilts(1); contrast = contrasts(2); fname = 's2-t_0.1-c_0.45';
        case 3 %set 3
            tilt = tilts(1); contrast = contrasts(3); fname = 's3-t_0.1-c_1';
        case 4 %set 4
            tilt = tilts(2); contrast = contrasts(1); fname = 's4-t_0.2-c_0.3';
        case 5 %set 5
            tilt = tilts(2); contrast = contrasts(2); fname = 's5-t_0.2-c_0.45';
        case 6 %set 6
            tilt = tilts(2); contrast = contrasts(3); fname = 's6-t_0.2-c_1';
        case 7 %set 7
            tilt = tilts(3); contrast = contrasts(1); fname = 's7-t_0.4-c_0.3';
        case 8 %set 8
            tilt = tilts(3); contrast = contrasts(2); fname = 's8-t_0.4-c_0.45';
        case 9 %set 9
            tilt = tilts(3); contrast = contrasts(3); fname = 's9-t_0.4-c_1';
        case 10 %set 10
            tilt = tilts(4); contrast = contrasts(1); fname = 's10-t_0.8-c_0.3';
        case 11 %set 11
            tilt = tilts(4); contrast = contrasts(2); fname = 's11-t_0.8-c_0.45';
        case 12 %set 12
            tilt = tilts(4); contrast = contrasts(3); fname = 's12-t_0.8-c_1';
        case 13 %set 13
            tilt = tilts(5); contrast = contrasts(1); fname = 's13-t_1.6-c_0.3';
        case 14 %set 14
            tilt = tilts(5); contrast = contrasts(2); fname = 's14-t_1.6-c_0.45';
        case 15 %set 15
            tilt = tilts(5); contrast = contrasts(3); fname = 's15-t_1.6-c_1';
        case 16 %set 16
            tilt = tilts(6); contrast = contrasts(1); fname = 's16-t_3.2-c_0.3';
        case 17 %set 17
            tilt = tilts(6); contrast = contrasts(2); fname = 's17-t_3.2-c_0.45';
        case 18 %set 18
            tilt = tilts(6); contrast = contrasts(3); fname = 's18-t_3.2-c_1';
    end
    counter_tilt = orientation - tilt; clock_tilt = orientation + tilt;
    for i=1:num_gabors
        imageN = imageN + 1;
        gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
        gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
        counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\datasets\\model2\\%s\\cclock\\cclock%d.png',fname,imageN);
        imwrite(gaborPatch_counter,counter_label);
        clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\datasets\\model2\\%s\\clock\\clock%d.png',fname,imageN);
        imwrite(gaborPatch_clock,clock_label);
    end
end
toc