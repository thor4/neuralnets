:: Creates SubB folders within SubA folders all within the RootDir

@ECHO ON
SET RootDir=J:\OneDrive - Georgia Institute of Technology\projects\metacognitive bias\stimuli\testing\tilt_contrast-cifar10
SET SubA=t_1_0833-c_0_3,t_1_0833-c_0_45,t_1_0833-c_1,t_2_3958-c_0_3,t_2_3958-c_0_45,t_2_3958-c_1,t_4-c_0_3,t_4-c_0_45,t_4-c_1
SET SubB=clock,cclock
FOR %%A IN (%SubA%) DO FOR %%B IN (%SubB%) DO IF NOT EXIST "%RootDir%\%%~A\%%~B" MD "%RootDir%\%%~A\%%~B"
EXIT