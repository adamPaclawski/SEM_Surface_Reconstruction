#!/usr/bin/python
##SEMSurfaceReconstruction_v0.5
#Authors: Adam Paclawski
#E-mail: adam.paclawski@uj.edu.pl


######################### For user modification ################################


######################### Main code - don't modifiy ##############################
import scipy
import matplotlib
import cv2
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import itertools
from scipy import ndimage
import imageio
from PIL import Image
from pylab import *


name1="left.jpg"
name2="right.jpg"
tiltangle=5.650
pxSize=1.0/34.538
metricUnit="mm"

img2 = cv2.imread(name2,0) #queryimage # left image
img1 = cv2.imread(name1,0) #trainimage # right image
img10 = img1 ##It could be used to crop images size
img20 = img2 ##It could be used to crop images size
y,x = img10.shape

imageio.imwrite('left.jpg', img10)
imageio.imwrite('right.jpg', img20)

class SEMimage(object):
    def __init__(self, name, tiltangle, pxSize):
        self.name=name
        self.data=cv2.imread(name)
        self.tiltangle=tiltangle
        self.pxSize=pxSize
        
def makeDisparityMap(imgLeft, imgRight, tiltangle, pxSize, dispartiesNum, min, blcSize, prefilterCap):
    gray_left = imgLeft
    gray_right = imgRight
    disparityAlg=cv2.StereoSGBM_create()
    disparityAlg.setBlockSize(blcSize)
    disparityAlg.setNumDisparities(dispartiesNum)
    disparityAlg.setPreFilterCap(prefilterCap)
    disparityAlg.setMinDisparity(min)
    disparityAlg.setUniquenessRatio(10)
    disparityAlg.setSpeckleWindowSize(0)
    disparityAlg.setSpeckleRange(2)
    disparityAlg.setDisp12MaxDiff(-1)
    disparityAlg.setP1(8*disparityAlg.getBlockSize()*disparityAlg.getBlockSize())
    disparityAlg.setP2(32*disparityAlg.getBlockSize()*disparityAlg.getBlockSize())
    disparityAlg.setMode(1)
    disparity = disparityAlg.compute(gray_left, gray_right)
    heightMap = (disparity/16.)*pxSize/(2*math.sin(math.radians(tiltangle/float(2.0))))
    return heightMap


def SEMboost(imgIn, heightMap, windSize):
    
    imgOut=np.zeros_like(heightMap)
    xaxis, yaxis = heightMap.shape
    
    for s1 in range(0, windSize, 1): 
        for s2 in range(0, windSize, 1):
            windowSize=windSize
            startY=s1
            startX=s2
            stepX, stepY = [math.floor(xaxis/windowSize), math.floor(yaxis/windowSize)]
            for i in range(0, int(stepY), 1):
                for z in range(0, int(stepX), 1):
                    minValue=heightMap[startX:(startX+windowSize), startY:(startY+windowSize)].min()
                    maxValue=heightMap[startX:(startX+windowSize), startY:(startY+windowSize)].max()
                    NewRange = (maxValue - minValue)  
                    minGrey=imgIn[startX:(startX+windowSize), startY:(startY+windowSize)].min()
                    maxGrey=imgIn[startX:(startX+windowSize), startY:(startY+windowSize)].max()
                    OldRange = (maxGrey - minGrey) 
                    OldValue=np.array(imgIn[startX:(startX+windowSize), startY:(startY+windowSize)])
                    if OldRange>0:
                        NewValue = (((OldValue - minGrey) * float32(NewRange)) / float32(OldRange)) + abs(minValue)
                    else:
                        NewValue = 0
                    imgOut[startX:(startX+windowSize), startY:(startY+windowSize)]+=NewValue
                    startX=startX+windowSize
                startY=startY+windowSize    
                startX=0                
    imgOut[np.isnan(imgOut)]=0
    heightMap = ndimage.rotate(heightMap, -90)
    imgIn = ndimage.rotate(imgIn, -90)
    imgOut = ndimage.rotate(imgOut, -90)
    xaxis, yaxis = heightMap.shape
    
    for s1 in range(0, windSize, 1): 
        for s2 in range(0, windSize, 1):
            windowSize=windSize
            startY=s1
            startX=s2
            stepX, stepY = [math.floor(xaxis/windowSize), math.floor(yaxis/windowSize)]
            for i in range(0, int(stepY), 1):
                for z in range(0, int(stepX), 1):
                    minValue=heightMap[startX:(startX+windowSize), startY:(startY+windowSize)].min()
                    maxValue=heightMap[startX:(startX+windowSize), startY:(startY+windowSize)].max()
                    NewRange = (maxValue - minValue)  
                    minGrey=imgIn[startX:(startX+windowSize), startY:(startY+windowSize)].min()
                    maxGrey=imgIn[startX:(startX+windowSize), startY:(startY+windowSize)].max()
                    OldRange = (maxGrey - minGrey) 
                    OldValue=np.array(imgIn[startX:(startX+windowSize), startY:(startY+windowSize)])
                    if OldRange>0:
                        NewValue = (((OldValue - minGrey) * float32(NewRange)) / float32(OldRange)) + abs(minValue)
                    else:
                        NewValue = 0
                    imgOut[startX:(startX+windowSize), startY:(startY+windowSize)]+=NewValue
                    startX=startX+windowSize
                startY=startY+windowSize    
                startX=0                
    imgOut[np.isnan(imgOut)]=0
    imgOut = ndimage.rotate(imgOut, 90)
    return imgOut/(float(windSize)**2*2) 


def removeOutliers(heightMap, windSize, thrVar):
    xaxis, yaxis = heightMap.shape    
    windowSize=windSize
    threshold=heightMap.max()/float(thrVar)
    startY=0
    startX=0
    stepX, stepY = [math.floor(xaxis/windowSize), math.floor(yaxis/windowSize)]

    for i in range(0, int(stepY), 1):
        for z in range(0, int(stepX), 1):
            meanValue=np.median(heightMap[startX:(startX+windowSize), startY:(startY+windowSize)])
            heightMap[startX:(startX+windowSize), startY:(startY+windowSize)][heightMap[startX:(startX+windowSize), startY:(startY+windowSize)]>(meanValue+threshold)]=meanValue
            heightMap[startX:(startX+windowSize), startY:(startY+windowSize)][heightMap[startX:(startX+windowSize), startY:(startY+windowSize)]<(meanValue-threshold)]=meanValue
            startX=startX+windowSize
        startY=startY+windowSize    
        startX=0
        
    return heightMap

#def createHeightMap(leftImg, rightImg, angle, pixelSize):


minDispNum=0
blcSizeTmp=3
prefilterCap=0
dispFinal=np.zeros(shape=img10.shape[:2])
invdispFinal=np.zeros(shape=img10.shape[:2])
for dispNum in range(48,64,16):
    ##Make disparity calculation and minimize output sum
    disTemp1=makeDisparityMap(img10, img20, tiltangle, pxSize, dispNum, minDispNum, blcSizeTmp, prefilterCap)
    plt.imshow(disTemp1),plt.show()
    filename1="blcSize_%d_dispNum_%d_disTemp1.jpg" %(blcSizeTmp, dispNum)
    imageio.imsave(filename1, disTemp1)
    dispFinal=dispFinal+disTemp1
    invdisTemp1=makeDisparityMap(img20, img10, tiltangle, pxSize, dispNum, minDispNum, blcSizeTmp, prefilterCap)
    plt.imshow(invdisTemp1),plt.show()
    invfilename1="blcSize_%d_dispNum_%d_invdisTemp1.jpg" %(blcSizeTmp, dispNum)
    imageio.imsave(invfilename1, invdisTemp1)
    invdispFinal=invdispFinal+invdisTemp1
    
    if disTemp1.sum()>invdisTemp1.sum():
        heightMapX=disTemp1
    else:
        heightMapX=invdisTemp1
        
    heightMapX=heightMapX+abs(heightMapX.min())
    heightMapX=removeOutliers(heightMapX, 10, 5)
    plt.imshow(heightMapX),plt.show()

    heightMapMean=heightMapX
    heightMapFinal=SEMboost(img20, heightMapMean, 10)
    #heightMapFinal=heightMapMean
    yaxis, xaxis = heightMapFinal.shape
    
    xlock = range(0, xaxis+1, int(math.floor(xaxis/4)))
    ylock = range(0, yaxis+1, int(math.floor(yaxis/4)))
    xlabels = [math.floor(float(i)*pxSize) for i in xlock]
    ylabels = [math.floor(float(i)*pxSize) for i in ylock[::-1]]
    
    
    img=plt.imshow(heightMapFinal)
    
    plt.xticks(xlock, xlabels)
    plt.yticks(ylock, ylabels)
    scalebar=plt.colorbar()
    scalebar.ax.set_xlabel('mm', rotation=0)
    
    #plt.ylabel(r'$\mu$m')
    #plt.xlabel(r'$\mu$m')
    plt.ylabel('mm')
    plt.xlabel('mm')
    imgname=("OutputblcSize_%d_dispNum_%d.jpg") %(blcSizeTmp, dispNum)
    plt.savefig(imgname)
    plt.show()
    cv2.waitKey(0)
    
    #Save results
    imgnamemap='reconstructedSurface_blcSize_%d_dispNum_%d.jpg' %(blcSizeTmp, dispNum)
    imageio.imwrite(imgnamemap, heightMapFinal)
    outName="outputData_%d_dispNum_%d.txt"  %(blcSizeTmp, dispNum)
    outFile = open(outName, 'w')
    outFile.write("min= " + str(heightMapFinal.min()) + "\n")
    outFile.write("max= " + str(heightMapFinal.max()) + "\n")
    outFile.write("pxSize= " + str(pxSize) + "\n")
    outFile.close()
