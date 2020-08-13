# torus
Detection and analysis of liquid torus

The main class is contained in torusClass.py

I recommend using IPython when working on a video.

An example of use is given in example.py


The main variables to watch out for when dealing with a video:
  
  torus.seuil - the thresholding value (only the lower one, future version will have upper and lower limit for better precison)
  torus.percInit - the percantage of the calibration image that gets substracted from each frame (needs to be updated)
  torus.calibImg - integer, 1 or 0, if there is a calibration image of a static torus
