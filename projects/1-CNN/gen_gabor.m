%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

%match all aspects of gabors from experiment
width = 169 ; % p.stimSize in pixels 
nGaussianSDs = [] ; %default in exp (6)
contrast = 0.45 ; %middle contrast from experiment, dropping other 2
noise = 1 ; %same as exp
gratingPeriod = 0.5 ; %same as exp
gratingPeriodUnits = 'sd' ; %same as exp
orientation = 45 ; %add to / subtract from this to get classes
tilt = 2.26; %avg tilt used in exp
counter_tilt = orientation + tilt; clock_tilt = orientation - tilt; 

gaborPatch = uint8(makeGaborPatch(width,[],contrast,noise,...
    gratingPeriod,gratingPeriodUnits,orientation));

gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
    gratingPeriod,gratingPeriodUnits,counter_tilt));

% gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
%     gratingPeriod,gratingPeriodUnits,60));

gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
    gratingPeriod,gratingPeriodUnits,clock_tilt));

% gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
%     gratingPeriod,gratingPeriodUnits,-60));

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


export_fig('gabors','-png','-transparent'); %save transparent pdf in pwd

%something is up with the orientation change Farshad made in the
%makeGaborPatch function. ie: when I choose 0 for orientation, it gives a
%vertical grating, pi/2 gives a horizontal. counter at 47.26 looks the same
%as 60. clock at 42.74 looks like -45