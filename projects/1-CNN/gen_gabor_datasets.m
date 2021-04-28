%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

% this generated the 9 different datasets for testing purposes

%match all aspects of gabors from experiment
width = 169 ; % p.stimSize in pixels 
nGaussianSDs = [] ; %default in exp (6)
contrasts = [.3, .45, 1];  % 30%, 45% and 100% contrast
noise = 1; %same as exp
gratingPeriod = 0.5; %same as exp
gratingPeriodUnits = 'sd'; %same as exp
orientation = pi/4; %radians (45 deg), add to / subtract from this to get classes
tilt_deg = 2.26; %avg tilt in degrees
tilts = [tilt_deg/2*(pi/180) tilt_deg*(pi/180) tilt_deg*2*(pi/180)]; %tilts used in exp, degrees->radians: *(pi/180)
%p.tilts = [middle_tilt/2 (1.13), middle_tilt (2.26), 2*middle_tilt (4.52)];
%middle_tilt = (p.practice{4}.finalTilt + p.practice{5}.finalTilt)/2; %uses
%the participant's results from training to create their initial middle
%tile
%Middle tilt = 2.26 # supposed to yield 75% accuracy, this is in degrees
%choose which set to generate:
set = 3;
%set1: tilt2.26 con1, set2: tilt2.26 con0.3, set3: tilt2.26 con0.45
%set4: tilt1.13 con1, set5: tilt1.13 con0.3, set6: tilt1.13 con0.45
%set7: tilt4.52 con1, set8: tilt4.52 con0.3, set9: tilt4.52 con0.45
switch set
    case 1 %set 1
        tilt = tilts(2); contrast = contrasts(3); fname = 'set1-t_2.26-c_1';
    case 2 %set 2
        tilt = tilts(2); contrast = contrasts(1); fname = 'set2-t_2.26-c_0.3';
    case 3 %set 3
        tilt = tilts(2); contrast = contrasts(2); fname = 'set3-t_2.26-c_0.45';
    case 4 %set 4
        tilt = tilts(1); contrast = contrasts(3); fname = 'set4-t_1.13-c_1';
    case 5 %set 5
        tilt = tilts(1); contrast = contrasts(1); fname = 'set5-t_1.13-c_0.3';
    case 6 %set 6
        tilt = tilts(1); contrast = contrasts(2); fname = 'set6-t_1.13-c_0.45';
    case 7 %set 7
        tilt = tilts(3); contrast = contrasts(3); fname = 'set7-t_4.52-c_1';
    case 8 %set 8
        tilt = tilts(3); contrast = contrasts(1); fname = 'set8-t_4.52-c_0.3';
    case 9 %set 9
        tilt = tilts(3); contrast = contrasts(2); fname = 'set9-t_4.52-c_0.45';
end
counter_tilt = orientation - tilt; clock_tilt = orientation + tilt; 
num_gabors = 500; %define total number of gabors to make for each class

%generate and save gabor images of size width+1 x width+1 (170x170 .png's)
for i=1:num_gabors
    gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
        gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
    gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
        gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
    counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\%s\\cclock\\cclock%d.png',fname,i);
    imwrite(gaborPatch_counter,counter_label);
    clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\%s\\clock\\clock%d.png',fname,i);
    imwrite(gaborPatch_clock,clock_label);
end

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
