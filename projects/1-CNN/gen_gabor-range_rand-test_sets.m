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
contrasts = [0.3, .45, 1];  % 30%, 45% and 100% contrast, qualifying tilt/contrast corpus
% contrast = 0.45; %middle contrast from experiment, dropping other 2
noise = 1; %same as exp
gratingPeriod = 0.5; %same as exp
gratingPeriodUnits = 'sd'; %same as exp
orientation = pi/4; %radians (45 deg), add to / subtract from this to get classes
% num_tilts = 50; %define how many tilts to randomly generate
rng(0,'twister'); %init random number generator to make results repeatable
%create vector of num_tilts random values. Use rand function to draw 
%values from uniform distribution in open interval, (tilt_start,tilt_end)
% tilt_start = 0.05;
% tilt_end = 2; 
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
% tilts = [0.05:13/160:2] .* (pi/180); %25 tilts tiled evenly from 0.05-2, converted to radians
% tilts = [0.5:3.5/24:4] .* (pi/180); %25 tilts tiled evenly from 0.5-4, converted to radians
% tilts = [0.05:13/160:0.05+(13/160)*2] .* (pi/180); %3 tilts for van gabor
tilts = [0.5+(3.5/24*4) 0.5+(3.5/24*13) 4] .* (pi/180); %3 tilts for cifar10
% fnames = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12",...
%     "s13","s14","s15","s16","s17","s18","s19","s20","s21","s22","s23",...
%     "s24","s25"];
fnames = "%s-%s"; %qualifying tilt/contrast corpus
contrast_names = ["c_0_3","c_0_45","c_1"]; %qualifying tilt/contrast corpus
tilt_names = ["t_1_0833","t_2_3958","t_4"]; %cifar10 tilts
% tilt_names = ["t_0_05","t_0_1313","t_0_2125"]; %van gabor tilts
% root_dir_clock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_contrast-van_gabor\\%s\\clock\\clock%d.png";
% root_dir_cclock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_contrast-van_gabor\\%s\\cclock\\cclock%d.png";
root_dir_clock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_contrast-cifar10\\%s\\clock\\clock%d.png";
root_dir_cclock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_contrast-cifar10\\%s\\cclock\\cclock%d.png";
% root_dir_clock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_0_5-4_contrast_0_45\\%s\\clock\\clock%d.png";
% root_dir_cclock = "J:\\OneDrive - Georgia Institute of Technology\\projects\\metacognitive bias\\stimuli\\testing\\tilt_0_5-4_contrast_0_45\\%s\\cclock\\cclock%d.png";
num_gabors = 2000; %define number of gabors to make for each contrast
% 3500 gabors * 3 contrasts * 2 classes = 21,000 gabors
imageN = 1; %initialize image counter

tic %generate qualifying tilt/contrast corpus
for contrastN = 1:length(contrasts)
    contrast = contrasts(contrastN);
    contrast_name = contrast_names(contrastN);
    for tiltN = 1:length(tilts)
        counter_tilt = orientation - tilts(tiltN); 
        clock_tilt = orientation + tilts(tiltN); 
        fname = sprintf(fnames,tilt_names(tiltN),contrast_names(contrastN));
        imageN = gen_gabor(num_gabors,contrast,clock_tilt,width,noise,...
            gratingPeriod,gratingPeriodUnits,imageN,root_dir_clock,fname);
        imageN = gen_gabor(num_gabors,contrast,counter_tilt,width,noise,...
            gratingPeriod,gratingPeriodUnits,imageN,root_dir_cclock,fname);
    end
end
toc
% tic %generate test sets
% for tiltN = 1:length(tilts)
%     counter_tilt = orientation - tilts(tiltN); 
%     clock_tilt = orientation + tilts(tiltN); 
%     fname = fnames(tiltN);
%     imageN = gen_gabor(num_gabors,contrast,clock_tilt,width,noise,...
%         gratingPeriod,gratingPeriodUnits,imageN,root_dir_clock,fname);
%     imageN = gen_gabor(num_gabors,contrast,counter_tilt,width,noise,...
%         gratingPeriod,gratingPeriodUnits,imageN,root_dir_cclock,fname);
% end
% toc
