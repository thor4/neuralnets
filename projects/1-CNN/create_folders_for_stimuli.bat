:: Creates SubB folders within SubA folders all within the RootDir

@ECHO ON
SET RootDir=J:\OneDrive - Georgia Institute of Technology\projects\metacognitive bias\stimuli\testing\tilt_contrast-van_gabor
SET SubA=s1-t_0_05-c_1,s2-t_0_1313-c_0_45,s3-t_0_2125-c_0_3
SET SubB=clock,cclock
FOR %%A IN (%SubA%) DO FOR %%B IN (%SubB%) DO IF NOT EXIST "%RootDir%\%%~A\%%~B" MD "%RootDir%\%%~A\%%~B"
EXIT