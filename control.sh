#!/bin/sh

echo Defining parameters

read -p '   Framerate: ' frate
read -p '   Resolution x: ' resx
read -p '   Resolution y: ' resy

echo
echo Choose one of the following and press Enter
echo 1 - Live view
echo 2 - Record video AVI
echo 3 - Record video RAW
echo
read -p 'Choice: ' usrin

if [ $usrin -eq 1 ];
then
    gst-launch-1.0 aravissrc camera-name="The Imaging Source Europe GmbH-51710050" exposure=1000 gain=20 ! video/x-raw, format=GRAY8, width=$resx, height=$resy, framerate=$frate/1 ! videoconvert ! ximagesink
elif [ $usrin -eq 2 ];
then 
    read -p 'Video title: ' name
    gst-launch-1.0 aravissrc camera-name="The Imaging Source Europe GmbH-51710050" exposure=1000 gain=20 ! video/x-raw, format=GRAY8, width=900, height=900, framerate=21/1 ! videoconvert ! avimux ! filesink location=$name.avi
elif [ $usrin -eq 3 ];
then
    read -p 'Video title: ' name
    gst-launch-1.0 aravissrc camera-name="The Imaging Source Europe GmbH-51710050" exposure=1000 gain=20 ! video/x-raw, format=GRAY8, width=900, height=900, framerate=21/1 ! filesink location=$name.raw
fi
