
def runVideo(video):
    """Loops over the video to give a final output signal

    Input:
        video - the video which we want to analyze

    Output:
        numpy array - (time,theta) numpy array containng the signal in time and space

    TODO:
        this will end up as a part of a class, will be different probz
    """

    borderCalib=borderDetect(calibFull)
    signalCalib=getSignal(borderCalib)

    completeSignal=np.zeros((numFrames,1401))
    i=0

    #plt.plot(signalCalib[0],signalCalib[1])
    while(video.isOpened()):

        ret, frame = video.read()

        if ret == True:

            border=borderDetect(frame)
            signal=getSignal(border)

            completeSignal[i,:]=signal[1]-signalCalib[1]

            i+=1

            os.system('clear')
            print(int(100*i/numFrames))

            #plt.plot(signal[0],completeSignal[i,:])
        else:
            break

    return completeSignal
