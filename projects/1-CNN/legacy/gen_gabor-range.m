%% Generate gabor patches with labels to be used in training CNN model
% exp code can be found here: OneDrive\Documents\postdoc @ Georgia Tech\projects\cnn\doby-exp\Expt1
% example can be found in one_block.m, line 35:
% gaborPatch = makeGaborPatch(p.stimSize, [], p.contrasts(contr_num), p.noiseContrast, .5, 'sd');
% func args: makeGaborPatch(width,nGaussianSDs,contrastFraction,contrastNoise,gratingPeriod,gratingPeriodUnits,orientation,black,white)
% to get example 'p' var: load('results11_10_50.mat')

% this will generate gabors with a range of tilts and contrasts yielding 
% 500 images per class, 21 total categories, 21k total images 
% did not work, now trying 18 total categories (remove lowest tilt),
% 500*18*2 = 18k total images [05/11/21]
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
% tilt = 2.26*(pi/180); %avg tilt used in exp, degrees->radians: *(pi/180)
tilts = [.1, .2, .4, .8, 1.6, 3.2].*(pi/180); %degrees->radians: *(pi/180)
%removing the 0.05 tilt and trying again, including it led to chance-level performance [05/11/21]
%p.tilts = [middle_tilt/2, middle_tilt, 2*middle_tilt];
%Middle tilt = 2.26 # supposed to yield 75% accuracy, this is in degrees
counter_tilts = orientation - tilts; clock_tilts = orientation + tilts; 
num_gabors = 500; %define number of gabors to make for each class
n_train_gabor = floor(num_gabors*(2/3)); %number of training gabor images
n_valid_gabor = floor(0.8*(num_gabors-n_train_gabor)); %number of validation gabor images
n_test_gabor = num_gabors - n_train_gabor - n_valid_gabor;
imageN = 0; %initialize image counter

tic
for contrastN = 1:length(contrasts)
    contrast = contrasts(contrastN); %current contrast
    for tiltN = 1:length(tilts)
        tilt=tilts(tiltN); counter_tilt=counter_tilts(tiltN); clock_tilt=clock_tilts(tiltN); %current tilt
        for i=1:num_gabors
            gaborPatch_counter = uint8(makeGaborPatch(width,[],contrast,noise,...
                gratingPeriod,gratingPeriodUnits,counter_tilt)); %counter clockwise
            gaborPatch_clock = uint8(makeGaborPatch(width,[],contrast,noise,...
                gratingPeriod,gratingPeriodUnits,clock_tilt)); %clockwise
            if (ismember(i,(1:n_train_gabor)))
                imageN = imageN + 1;
                counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\train_range\\cclock\\cclock%d.png',imageN);
                imwrite(gaborPatch_counter,counter_label);
                clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\train_range\\clock\\clock%d.png',imageN);
                imwrite(gaborPatch_clock,clock_label);
            elseif (ismember(i,(n_train_gabor+1:n_train_gabor+n_valid_gabor))) 
                imageN = imageN + 1;
                counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\validation_range\\cclock\\cclock%d.png',imageN);
                imwrite(gaborPatch_counter,counter_label);
                clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\validation_range\\clock\\clock%d.png',imageN);
                imwrite(gaborPatch_clock,clock_label);
            else
                imageN = imageN + 1;
                counter_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\test_range\\cclock\\cclock%d.png',imageN);
                imwrite(gaborPatch_counter,counter_label);
                clock_label = sprintf('C:\\Users\\bryan\\Documents\\GitHub\\neuralnets\\projects\\1-CNN\\images\\test_range\\clock\\clock%d.png',imageN);
                imwrite(gaborPatch_clock,clock_label);
            end
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