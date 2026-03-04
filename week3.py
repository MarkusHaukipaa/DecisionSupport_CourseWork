"""
Created on Thu Oct 23 13:26:02 2025

@author: mhauk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sc
import neurokit2 as nk2
import matplotlib.dates as mdates 
from zipfile import ZipFile
import os 


def getFile(folder,filename):
    with ZipFile(folder + os.sep + r"RawData.zip") as z:
        with z.open(filename) as file: 
            file = pd.read_csv(file, sep = " ", names = ["x","y","z"])
    return file

def main():
    #module that allows to use operating system specific features,  
    #e.g. Windows and Linux use different file separators, i.e. characters 
    # that separate individual #folder and file names in a path.  
    direc = r"C:\Users\mhauk\OneDrive - TUNI.fi\Opinnot\Kurssit\Decision_support\Week_3\Exercise 3-20251104"
    
    with ZipFile(direc + os.sep + r"RawData.zip") as z:
         filenames= z.namelist()
         filenames.pop(0) # unnesary folder name
         labelfile = filenames.pop()
         filenames.sort()
         # Remove gyro files
         filenames = [f for f in filenames if f.startswith("RawData/acc")]
         with z.open(labelfile) as file:
             labels = pd.read_csv(file, sep = " ", names = ["eID","uID","aID",
                                                           "start","stop"])
            
    f = 50 

    Ex1(getFile(direc,filenames[9]),f)
    
    return
    Ex2(labels,filenames,direc)
    df3 = pd.DataFrame()
    
    #GO thourgh all the experiement files and concate them together to for 
    #one big file with notation to walking or sationary
    
    for i in range(len(filenames)):
        
        experiement = filenames[i].split("_exp")[1].split("_")[0]
        current_labels = labels.loc[labels["eID"] == int(experiement)] 

        current_labels.reset_index(drop=True, inplace=True) 
        df3=pd.concat([df3, extractFeatures(getFile(direc, filenames[i]),
                                            current_labels)])
        
    walkingFeatures, stationaryFeatures = seperateWS(df3)
    plotFeatures(walkingFeatures, "Walking features")
    plotFeatures(stationaryFeatures, "Stationary features")
    
def plotFeatures(featureDataFrame, title):
    
    avg_mean_y = []
    avg_max_y = []
    avg_min_y = []
    
    for subject in featureDataFrame["Subject"].unique():
        subjectData = featureDataFrame.loc[featureDataFrame["Subject"] == subject]
        avg_mean_y.append(np.mean(subjectData["mean_x"]))
        avg_max_y.append(np.mean(subjectData["max_x"]))
        avg_min_y.append(np.mean(subjectData["min_x"]))
        
    plt.plot(featureDataFrame["Subject"].unique(), avg_mean_y, label = "avg_mean_x")
    plt.plot(featureDataFrame["Subject"].unique(), avg_max_y, label = "avg_max_x")
    plt.plot(featureDataFrame["Subject"].unique(), avg_min_y, label = "avg_min_x")
    
    plt.scatter(featureDataFrame["Subject"], featureDataFrame["mean_x"], label = "mean")
    plt.scatter(featureDataFrame["Subject"], featureDataFrame["max_x"], label = "max")
    plt.scatter(featureDataFrame["Subject"], featureDataFrame["min_x"], label = "min")
    plt.title(title)
    plt.xlabel("Subject")
    plt.ylabel("Acceleration")
    
    plt.legend()
    plt.show()

def seperateWS(df):
    
    walkingFeatures = df.loc[df["Type"] == "Walking"]
    walkingFeatures.reset_index(drop=True, inplace=True) 
    stationaFeatures = df.loc[df["Type"] == "Stationary"]
    stationaFeatures.reset_index(drop=True, inplace=True) 
    
    return walkingFeatures,stationaFeatures

def Ex2(labels,filenames,direc):
    
    subject = "05"
    df1 = pd.DataFrame()
    current_labels = labels.loc[(labels["uID"] == int(subject))] 
    current_labels.reset_index(drop=True, inplace=True) 
    eIDs = set(current_labels["eID"])
    
    
    for eID in eIDs:
        elabels = current_labels.loc[current_labels["eID"]== eID]
        elabels.reset_index(drop=True, inplace=True) 
        if eID < 10: 
            eID = "0" + str(eID)
        
        fileName = r"RawData/acc_exp"+ str(eID) + "_user"+ subject +".txt"
        patientFile = getFile(direc, fileName)
        df1 = pd.concat([df1, extractFeatures(patientFile, elabels)])
        
    walkingFeatures,stationaFeatures = seperateWS(df1)     
    
    walkingFeatures = walkingFeatures.sort_values(by=["Type Number"])
    stationaFeatures = stationaFeatures.sort_values(by=["Type Number"])

  
    plt.plot(walkingFeatures["Type Number"], walkingFeatures["mean_x"], label = "mean_x")
    plt.plot(walkingFeatures["Type Number"], walkingFeatures["max_x"], label = "max_x")
    plt.plot(walkingFeatures["Type Number"], walkingFeatures["min_x"], label = "min_x")
    plt.title("Walking features: sub 5")
    plt.xlabel("Movement Type")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()
    

    plt.plot(stationaFeatures["Type Number"], stationaFeatures["mean_x"], label = "mean_x")
    plt.plot(stationaFeatures["Type Number"], stationaFeatures["max_x"], label = "max_x")
    plt.plot(stationaFeatures["Type Number"], stationaFeatures["min_x"], label = "min_x")
    plt.title("Stationary features: sub 5")
    plt.xlabel("Movement type")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()

def Ex1(df,f):
    
 
    time= np.arange(0, len(df['x']))/f
    
    plt.plot(time, df['x'], label = "x", color = "r")
    plt.plot(time, df['y'], label = "y", color = "b")
    plt.plot(time, df['z'], label = "z", color = "y")
    plt.vlines([5,23,51,65,95,115,135,285,295], -1.3, 2)
    
    plt.title("Acceleration data of x, y, and z direction")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()
    

def extractFeatures(patientFile,labels):
 
    mean_value_x = [] 
    mean_value_y = [] 
    mean_value_z = [] 
    
    max_value_x = []
    max_value_y = []
    max_value_z = []
    
    min_value_x = []
    min_value_y = []
    min_value_z = []
    
    movementTypes = []
    
    for ii in range(len(labels)): 

        dataRow_x = patientFile['x'][labels['start'][ii]:labels['stop'][ii]]
        dataRow_y = patientFile['y'][labels['start'][ii]:labels['stop'][ii]]
        dataRow_z = patientFile['z'][labels['start'][ii]:labels['stop'][ii]]
        
        mean_value_x.append(np.mean(dataRow_x)) 
        mean_value_y.append(np.mean(dataRow_y))
        mean_value_z.append(np.mean(dataRow_z))
        
        
        max_value_x.append(np.max(dataRow_x)) 
        max_value_y.append(np.max(dataRow_y))
        max_value_z.append(np.max(dataRow_z))
        
        min_value_x.append(np.min(dataRow_x))
        min_value_y.append(np.min(dataRow_y))
        min_value_z.append(np.min(dataRow_z))
        
        if labels["aID"][ii] <= 4:
            movementTypes.append("Walking")
        else:  movementTypes.append("Stationary") 
    
    features = pd.DataFrame({"Subject": labels["uID"] ,
                            "Type": movementTypes,
                            "Type Number": labels["aID"],
                            "mean_x": mean_value_x,
                            "max_x": max_value_x,
                            "min_x": min_value_x,
                            "mean_y": mean_value_y,
                            "max_y": max_value_y,
                            "min_y": min_value_y,
                            "mean_z": mean_value_z,
                            "max_z": max_value_z,
                            "min_z": min_value_z
                            })
    return features
funcitons = main()