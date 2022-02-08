function imageN = gen_gabor(num_gabors,contrast,tilt,width,noise,gratingPeriod,gratingPeriodUnits,imageN,root_dir,fname)
    for i=1:num_gabors
        gaborPatch = uint8(makeGaborPatch(width,[],contrast,noise,...
            gratingPeriod,gratingPeriodUnits,tilt)); 
        gabor_path = sprintf(root_dir,fname,imageN);
        imwrite(gaborPatch,gabor_path);
        imageN = imageN + 1;
    end
end