%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

% this will generate 6 sets of gabors, each with a different tilt across 3
% contrasts. 500 gabors per contrast/tilt permutation, 1,500 per set, 6
% total sets, 9,000 total gabors

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
set = 6;
%set1: tilt2.26 con1, set2: tilt2.26 con0.3, set3: tilt2.26 con0.45
%set4: tilt1.13 con1, set5: tilt1.13 con0.3, set6: tilt1.13 con0.45
%set7: tilt4.52 con1, set8: tilt4.52 con0.3, set9: tilt4.52 con0.45
switch set
    case 1 %set 1
        tilt = tilts(1); contrast = contrasts(3); fname = 's1_1-t_0.1-c_0.3_0.45_1';
    case 2 %set 2
        tilt = tilts(2); contrast = contrasts(1); fname = 's1_2-t_0.2-c_0.3_0.45_1';
    case 3 %set 3
        tilt = tilts(3); contrast = contrasts(2); fname = 's1_3-t_0.4-c_0.3_0.45_1';
    case 4 %set 4
        tilt = tilts(4); contrast = contrasts(3); fname = 's1_4-t_0.8-c_0.3_0.45_1';
    case 5 %set 5
        tilt = tilts(5); contrast = contrasts(1); fname = 's1_5-t_1.6-c_0.3_0.45_1';
    case 6 %set 6
        tilt = tilts(6); contrast = contrasts(2); fname = 's1_6-t_3.2-c_0.3_0.45_1';
end
counter_tilt = orientation - tilt; clock_tilt = orientation + tilt; 
num_gabors = 500; %define total number of gabors to make for each tilt/contrast permutation
imageN = 0; %initialize image counter

%generate and save gabor images of size width+1 x width+1 (170x170 .png's)
for contrastN = 1:length(contrasts)
    contrast = contrasts(contrastN); %current contrast
    for i=1:num_gabors
        imageN = imageN + 1;
        gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
        gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
        counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\datasets\\%s\\cclock\\cclock%d.png',fname,imageN);
        imwrite(gaborPatch_counter,counter_label);
        clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\datasets\\%s\\clock\\clock%d.png',fname,imageN);
        imwrite(gaborPatch_clock,clock_label);
    end
end
