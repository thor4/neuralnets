:: Creates SubB folders within SubA folders all within the RootDir

@ECHO ON
SET RootDir=J:\OneDrive - Georgia Institute of Technology\projects\metacognitive bias\stimuli\testing\test
SET SubA=s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25
SET SubB=clock,cclock
FOR %%A IN (%SubA%) DO FOR %%B IN (%SubB%) DO IF NOT EXIST "%RootDir%\%%~A\%%~B" MD "%RootDir%\%%~A\%%~B"
EXIT