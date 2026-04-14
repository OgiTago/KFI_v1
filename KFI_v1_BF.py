# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:23:29 2023
20230524KF_ForceInf_NoBFI.pyからコピー

@author: Goshi
"""

import numpy as np
import seaborn as sns
import os
import natsort
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys
import copy 


import MyPyLib.ForceInf_lib 
import MyPyLib.OgitaInf_NL as NOgi
import MyPyLib.GetMatrixParameterEstimation as GPE
import MyPyLib.KF_functions as KF

#%% Set the number of threads used in numpy
#os.environ["MKL_NUM_THREADS"] = "4"
if os.uname().sysname ==  "Linux":
    os.environ["MKL_NUM_THREADS"] = "4"
    
#%% Set input directory and output directory
ObsSampleName = "DH4_114"    
SampleDirectory = "./SampleData/{}/Vertices/".format(ObsSampleName)
SaveDirectory = "./output/" + ObsSampleName + "/"

#%% Set parameters
ObsInterval = 0.1
TimeInterval =ObsInterval
SigmaObs = pow(10,-5)
SigmaT = 0.1
SigmaP = SigmaT
T0 = 0.3
P0 = 0
PT0 = 1.0
PP0 = 1.0


T_shift = 0
DURATION = 3000

PlotInterval=1
SkipNum = 1
Suffix=""

SM = False #Rescale forces to <P> = 0

 
print(SampleDirectory)
print(SaveDirectory)
print(ObsSampleName)
os.makedirs(SaveDirectory,exist_ok=True)

#%% Make output directory
SaveDirectory_p = SaveDirectory+"plot/"
SaveDirectory_b = SaveDirectory+"bin/"
os.makedirs(SaveDirectory,exist_ok=True)    
os.makedirs(SaveDirectory_p,exist_ok=True)    
os.makedirs(SaveDirectory_b,exist_ok=True)    

Filelist = natsort.natsorted(os.listdir(SampleDirectory))
DatList = natsort.natsorted([fi for fi in Filelist if "dat" in fi])

#%% Set initial conditions
#Load initial observation data
filename = SampleDirectory + DatList[T_shift]
#Load data
[x_old,y_old,edge_old,cell_old,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title]\
= MyPyLib.ForceInf_lib.loaddata( filename ) 
ERR_MAX = 100
[MM_old,C_NUM,X_NUM] =  MyPyLib.ForceInf_lib.GetMatrix_ForceEstimation\
                    (x_old,y_old,edge_old,cell_old,E_NUM,CELL_NUMBER,R_NUM,INV_NUM,Rnd,ERR_MAX)
Data_old = KF.Data([x_old,y_old,edge_old,cell_old,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title])
Data_old.set_Ft(edge_old,cell_old)
Data_old.MM = MM_old
Data_old.title = DatList[T_shift]

# edge, cell, forces, vertices inside tissue (exclude boundary)
E_in = list(set(range(E_NUM)) -  set(Rnd[1]))
C_in = list(set(range(CELL_NUMBER)) -  set(Rnd[2]))
F_in = list(set(E_in)|set(C_in))
V_in = list(set(range(V_NUM)) -  set(Rnd[0]))
V_in_xy = V_in.extend(list(np.array(V_in)+V_NUM))

#Mean
T0 = T0 * np.ones(E_NUM)
P0 = P0 * np.ones(CELL_NUMBER)
Fa = np.hstack((T0,P0))
#Variance-covariance matrix
Pa = np.eye(E_NUM+CELL_NUMBER)
Pa[:E_NUM,:E_NUM] = PT0**2 * np.eye(E_NUM)
Pa[E_NUM:,E_NUM:] = PP0**2 * np.eye(CELL_NUMBER)
# Variance-covariance for prediction error
Q = np.zeros([X_NUM,X_NUM])
Q[:E_NUM,:E_NUM] = SigmaT**2 * np.eye(E_NUM)
Q[E_NUM:,E_NUM:] = SigmaP**2 * np.eye(CELL_NUMBER)
Q = ObsInterval * SkipNum * Q
# Variance-covariance for observation error
R = SigmaObs**2 * np.eye(2*V_NUM)
# Systeme matrix
SM = np.eye(E_NUM+CELL_NUMBER)
#%% Save list
Pa_trace_list = []
Pf_trace_list = []
T_cor_list = []
P_cor_list = []
RearrangeRecorder = []
SigTrace = []
#%% Prediction-Analysis cycles
V_list = np.zeros((len(E_in),2))
for i,ti in enumerate(range(T_shift+1,len(DatList),SkipNum)):
 
    filename = SampleDirectory + DatList[ti]
    #Load data
    [x_new,y_new,edge_new,cell_new,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title]\
    = MyPyLib.ForceInf_lib.loaddata( filename ) #OK
    ERR_MAX = 100#2.0e-12
    [MM_new,C_NUM,X_NUM] =  MyPyLib.ForceInf_lib.GetMatrix_ForceEstimation\
                        (x_new,y_new,edge_new,cell_new,E_NUM,CELL_NUMBER,R_NUM,INV_NUM,Rnd,ERR_MAX) #OK
    Data_new = KF.Data([x_new,y_new,edge_new,cell_new,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title])
    Data_new.set_Ft(edge_new,cell_new)
    Data_new.MM = MM_new
    Data_new.title = DatList[ti]
    
    # Prediction
    Ff = SM@Fa
    Pf = SM@Pa@SM.T + Q
    
    Data_old.set_forecast(Ff,Pf)
    Pf_trace_list.append(Data_old.trPf)    
    # Analysis
    #Calculate vertex displacementd
    v = KF.CalcV(x_old,y_old,x_new,y_new,Rnd)
    
    Data_old.set_vf(v)
    # Calculate Kalman gain
    H = MM_old * TimeInterval * SkipNum
    
    K = Pf@H.T@np.linalg.inv(H@Pf@H.T + R)
    Data_old.set_KG(K)
    Fa = Ff + K@(v - H@Ff)
    IKH = np.eye((K@H).shape[0]) - K@H
    Pa = IKH@Pf@IKH.T + K@R@K.T
    #Data_old.set_analysis(Data_old.Ft,Pa)
    
    Data_old.set_analysis(Fa,Pa)
    
    
    Pa_trace_list.append(Data_old.trPa)
    # Calculate correlation coefficient # TO FIX
    TT_in = Data_old.Ft[E_in]
    TP_in = Data_old.Ft[E_NUM+np.array(C_in)]
    ET_in = np.array(Fa)[E_in]
    EP_in = np.array(Fa)[E_NUM+np.array(C_in)]
    coef_T = np.corrcoef(TT_in,ET_in)[0,1]
    T_cor_list.append(coef_T)
    coef_P = np.corrcoef(TP_in,EP_in)[0,1]
    P_cor_list.append(coef_P)
    
    
    
    MM_old = copy.deepcopy(MM_new)
    edge_old = copy.deepcopy(edge_new)
    cell_old = copy.deepcopy(cell_new)
    x_old = x_new.copy()
    y_old = y_new.copy()
    Data_old.save(SaveDirectory_b+"{}.dat".format(ti-1))
    Data_old = copy.deepcopy(Data_new)
    
#%%        
plt.title("CorrCoef@{}".format(ti))
plt.plot(T_cor_list,label="Tension")
plt.plot(P_cor_list,label="Pressure")
if min([min(T_cor_list),min(P_cor_list)]) < 0.5:
        plt.ylim(0,1.0)
else:
    plt.ylim(0.5,1.0)
plt.savefig(SaveDirectory+"CorrCoef.png",dpi=150)
#%%
T_cor_list = np.array(T_cor_list)
P_cor_list = np.array(P_cor_list)
cor_list = np.hstack([T_cor_list,P_cor_list])
np.save(SaveDirectory+"cor_list_{}.npy".format(ObsInterval),cor_list)