:: Creates SubB folders within SubA folders all within the RootDir

@ECHO ON
SET RootDir=D:\projects\metacognitive bias\stimuli\training\
SET SubA=train_range_rand,valid_range_rand,test_range_rand
SET SubB=clock,cclock
FOR %%A IN (%SubA%) DO FOR %%B IN (%SubB%) DO IF NOT EXIST "%RootDir%\%%~A\%%~B" MD "%RootDir%\%%~A\%%~B"
EXIT