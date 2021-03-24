%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

%match all aspects of gabors from experiment
width = 169 ; % p.stimSize in pixels 
nGaussianSDs = [] ; %default in exp (6)
%contrasts = [.3, .45, 1]  # 30%, 45% and 100% contrast
contrast = 1; %middle contrast from experiment, dropping other 2
noise = 1; %same as exp
gratingPeriod = 0.5; %same as exp
gratingPeriodUnits = 'sd'; %same as exp
orientation = pi/4; %radians (45 deg), add to / subtract from this to get classes
tilt = 2.26*(pi/180); %avg tilt used in exp, degrees->radians: *(pi/180)
%p.tilts = [middle_tilt/2, middle_tilt, 2*middle_tilt];
%Middle tilt = 2.26 # supposed to yield 75% accuracy, this is in degrees
counter_tilt = orientation - tilt; clock_tilt = orientation + tilt; 
num_gabors = 100; %define total number of gabors to make for each class
n_train_gabor = floor(num_gabors*(2/3)); %number of training gabor images
n_valid_gabor = floor(0.8*(num_gabors-n_train_gabor)); %number of validation gabor images
n_test_gabor = num_gabors - n_train_gabor - n_valid_gabor;

%generate and save gabor images of size width+1 x width+1 (170x170 .png's)
for i=1:num_gabors
    gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
        gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
    gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
        gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
    if (ismember(i,(1:n_train_gabor)))
        counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\train\\cclock\\cclock%d.png',i);
        imwrite(gaborPatch_counter,counter_label);
        clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\train\\clock\\clock%d.png',i);
        imwrite(gaborPatch_clock,clock_label);
    elseif (ismember(i,(n_train_gabor+1:n_train_gabor+n_valid_gabor))) 
        counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\validation\\cclock\\cclock%d.png',i);
        imwrite(gaborPatch_counter,counter_label);
        clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\validation\\clock\\clock%d.png',i);
        imwrite(gaborPatch_clock,clock_label);
    else
        counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\test\\cclock\\cclock%d.png',i);
        imwrite(gaborPatch_counter,counter_label);
        clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\test\\clock\\clock%d.png',i);
        imwrite(gaborPatch_clock,clock_label);
    end
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

%now generate 100 gabors for each class to be used in the transfer learning
%tensorflow code