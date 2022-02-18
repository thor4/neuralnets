:: deletes all png files in all subfolders of ...\testing\test
:: /s parameter deletes all files contained in the directory subfolders
:: /f parameter ignores any read-only setting.
:: /q “quiet mode,” meaning you won’t be prompted yes/no

ECHO deleting png's
del "J:\OneDrive - Georgia Institute of Technology\projects\metacognitive bias\stimuli\testing\tilt_contrast-van_gabor\*.png" /s /f /q
echo done!