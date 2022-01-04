import numpy as np
from skimage import io
from skimage import img_as_float
from skimage.util import img_as_ubyte
import pandas as pd
from matplotlib import pyplot
from PIL import Image

imgEntityArray = [[],[]]

def kMeansClusterAlg(imgDensity, kVal, cellTotal):

    print("####################################################################")
    print("Process Started.")
    print('')
    # Init values
    iterateNum = 1
    iterateNumConverge = 10 #Current Optimal iteration
    rgb = 3
    clusterLinearList = []
    initCentroidList = []
    # Forming centroid values
    clusterLinearList, initCentroidList = initCentroid(clusterLinearList, initCentroidList, cellTotal, kVal, rgb)
    # Checking for convergence
    convergenceAchieved = 0
    clusterLinearListBefore = [[],[]]
    initCentroidListBefore = [[],[]]
    
    # Converge loop
    while True:
        
        labelsVectorInit = {} #K labels
        
        for i in range(kVal): #labels repeat for # of K
        
            labelsVectorInit[i] = []
        
        vectorElement = 0 #indexing
        
        for index, pixelRow in enumerate(imgDensity):
            
            if vectorElement < (cellTotal+1):
                
                tempKList = []
                
                for i in range(kVal):
               
                    tempKList.append(pixelRow)
                    
                minDistanceClassIndex = findEuclideanDistance(tempKList, initCentroidList) #Finds distance
                clusterLinearList[index] = minDistanceClassIndex #Assigning labels 
                labelsVectorInit[minDistanceClassIndex].append(pixelRow) #Adding the rows
            
            vectorElement = vectorElement + 1
            
        initCentroidList = formUpdatedCluster(labelsVectorInit, initCentroidList) #Updating clusters
        
        if iterateNum > 3:
            
            clusterLinearListBefore[0] = clusterLinearListBefore[1]
            clusterLinearListBefore[1] = clusterLinearList
            initCentroidListBefore[0] = initCentroidListBefore[1]
            initCentroidListBefore[1] = initCentroidList
        
            if clusterLinearListBefore[0] == clusterLinearListBefore[1] and initCentroidListBefore[0] == initCentroidListBefore[1]:
                
                convergenceAchieved = convergenceAchieved + 1
        
        if convergenceAchieved < iterateNum:
            
            print("Process Complete:", round(((iterateNum/iterateNumConverge) * 100),3), "%")
        
        iterateNum = iterateNum + 1
        
        if(iterateNum > iterateNumConverge):
        
            break
    
    formArrayEntity(clusterLinearList, initCentroidList) #Storing image cluster form values

def formUpdatedCluster(labelsVectorInit, initCentroidList):
    
    for z in range(kVal):
            
            # Init values
            tempClusterVector = []
            appendVector = labelsVectorInit[z]
            appendVectorSize = len(appendVector)
            clusterTotal = [0,0]
            
            if appendVectorSize > 0:
                
                # Finding average
                tempClusterVector = np.asarray(appendVector)
                clusterTotal[0] = np.sum(tempClusterVector,0)
                clusterTotal[1] = clusterTotal[0]/appendVectorSize
                initCentroidList[z] = clusterTotal[1]
    
    return initCentroidList

def formArrayEntity(clusterLinearList, initCentroidList):
    
    # Forming image entity
    global imgEntityArray
    
    imgEntityArray[0] = clusterLinearList
    imgEntityArray[1] = initCentroidList
    
def initCentroid(clusterLinearList, initCentroidList, cellTotal, kVal, rgb):

    tempVector = []
    
    # Labels
    for i in range(cellTotal):
        
        clusterLinearList.append(0)
    
    # Initial random values
    for i in range(kVal):
        
        tempVector = np.random.rand(rgb)
        initCentroidList.append(tempVector)
        
    return clusterLinearList, initCentroidList
        
def findEuclideanDistance(tempKList, initCentroidList):
    
    tempClassDict = {}
    
    # Calculating distances from mean
    for i in range(kVal):
        
        currentDistanceE = np.linalg.norm(tempKList[i] - initCentroidList[i]) 
        tempClassDict[i] = currentDistanceE
    
    minClassVal = min(tempClassDict, key = lambda i : tempClassDict[i])
    
    return minClassVal

def imgTransform(img):
    
    # Image transformation
    formImg = []
    imgDensity = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])
    imgDensity = img_as_float(imgDensity)
    cellTotal = len(imgDensity)
    unitTotal = imgDensity.shape[0]
    formImg = np.zeros(imgDensity.shape)
    
    return imgDensity, cellTotal, unitTotal, formImg
    
def imgSave(unitTotal, formImg, img):
    
    # Reshaping and saving the image
    for i in range(unitTotal):
        
        formImg[i] = imgEntityArray[1][imgEntityArray[0][i]]
    
    print('')
    print('Image Saved to Local Folder as CurrentCompressedImage.')
    print('')
    formImg = formImg.reshape(img.shape[0], img.shape[1], img.shape[2])
    newImg = img_as_ubyte(formImg)
    io.imsave('CurrentCompressedImage.jpg', newImg)
    print('Process Completed.')
    print("####################################################################")
    
if __name__ == '__main__':

    # Input files for file location and K value.    
    inputFile1 = open('./FileLoc_Input.txt')
    inputVal1 = inputFile1.readlines()
    inputFile2 = open('./K_Input.txt')
    inputVal2 = inputFile2.readlines()
    
    imgLoc = inputVal1[0]
    kVal = int(inputVal2[0])
    
    img = io.imread(imgLoc) #Reading in Image as 3-D array
    
    imgDensity, cellTotal, unitTotal, formImg = imgTransform(img) #Transforming the image and setting initial values.
    kMeansClusterAlg(imgDensity, kVal, cellTotal) #The start function.
    imgSave(unitTotal, formImg, img) #Saving transformed image.
    