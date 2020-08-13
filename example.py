"""
An example where we load a video of torus with the required parameters
and extract the signal, after which the 2D FFT is calculated and
drawn. We set the effective propagation velocity c_eff and
surface tension sigma_eff which are needed to then plot
the dispersion relation

It is recommended to run these in the IPython console for interactive use of the class

"""


import load.py
import torusClass.py




filenames=["soliton.mp4","calibFull.bmp","calibEmpty.bmp"] # set filenames

examp=torus(filenames,0.06,700,400) # create a torus object

examp.openVideo() # open the video and the images

examp.runVideo() # run the video frame by frame to obtain a signal

examp.fourier() # calculate the fourier transform

examp.drawSpectrum() # draw the calculated spectrum

examp.c_eff = 0.075  # define the velocity 
examp.sigma_eff = 0.05 # define surface tension

examp.plotDispersion(1) # plot dispersion relation


