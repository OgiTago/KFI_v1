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
    
def CalcSM(E_NUM,CELL_NUMBER,RndC):
    InCellNumber = CELL_NUMBER - len(RndC)
    X_NUM = E_NUM + CELL_NUMBER
    SM = np.zeros((X_NUM,X_NUM))
    
    SM[:E_NUM,:E_NUM] = np.eye(E_NUM)
    SM[E_NUM:,E_NUM:] = np.full((CELL_NUMBER,CELL_NUMBER),-1/InCellNumber) + np.eye(CELL_NUMBER)
    return SM

SampleName = "Far1T004_1"
ObsInterval = 0.1
TimeInterval =ObsInterval
SigmaObs = pow(10,-5)
SigmaT = 0.1
SigmaP = SigmaT


T_shift = 0
DURATION = 300

PlotInterval=1
ObsSampleName = "D0"
SkipNum = 1
Suffix=""



#SampleDirectory = "../0001ObsData/{0}/{0}_t/Vertex/".format(ObsSampleName)
SampleDirectory = "../Data/Mechanics_2/{}/".format(ObsSampleName)


#%% 1. Set parameters
T0 = 0.5
P0 = 0.5
PT0 = 1
PP0 = 1
SM = False #Rescale forces to <P> = 0
#%% 2. Set estimation settings

#%%
if len(sys.argv)==1:
    ObsSampleName = "DH4_113"
    SampleDirectory = "../Data/{}/".format(ObsSampleName)
    SaveDirectory = "../" + ObsSampleName + "/"
 
    print(SampleDirectory)
    print(SaveDirectory)
    print(ObsSampleName)
    #SaveDirectory = OutputDirectory + ObsSampleName+"/"
    #SaveDirectory = "./20241204Mechanics_2ndRun/{}/".format(ObsSampleName)
    os.makedirs(SaveDirectory,exist_ok=True)
    #SampleDirectory = "./Data/Mechanics/{}/".format(ObsSampleName)

    
else:
    args = sys.argv
    ObsSampleName = args[3]
    SampleDirectory = args[1] + ObsSampleName + "/"
    if len(args) >= 5:
        SigmaObs = float(args[4])
    if len(args) >= 6:
        SigmaT = float(args[5])
        SigmaP = SigmaT        
    if SigmaObs == 0:
        SigmaObs = pow(10,-5)
    
    
    print("")
    print("SigmaObs = {}".format(SigmaObs))
    
    SaveDirectory = "{}{}_o{:.2e}_q{:.2e}/".format(args[2],ObsSampleName,SigmaObs,SigmaT)
    print(SampleDirectory)
    print(SaveDirectory)
    print(ObsSampleName)
    ObsSampleName = str(args[1])
    #SaveDirectory = OutputDirectory + ObsSampleName+"/"
    #SaveDirectory = "./20241204Mechanics_2ndRun/{}/".format(ObsSampleName)
    os.makedirs(SaveDirectory,exist_ok=True)
    #SampleDirectory = "./Data/Mechanics/{}/".format(ObsSampleName)

    
#%% Make output directory
SaveDirectory_p = SaveDirectory+"plot/"
SaveDirectory_b = SaveDirectory+"bin/"
os.makedirs(SaveDirectory,exist_ok=True)    
os.makedirs(SaveDirectory_p,exist_ok=True)    
os.makedirs(SaveDirectory_b,exist_ok=True)    

Filelist = natsort.natsorted(os.listdir(SampleDirectory))
DatList = natsort.natsorted([fi for fi in Filelist if "dat" in fi])

#%% 3. Set initial conditions
#Load initial observation data
filename = SampleDirectory + DatList[T_shift]
#Load data
[x_old,y_old,edge_old,cell_old,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title]\
= MyPyLib.ForceInf_lib.loaddata( filename ) 
ERR_MAX = 100#2.0e-12
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
EdgeList_new = np.zeros((E_NUM,2))
RearrangeRecorder = []
SigTrace = []
#%% Prediction-Analysis cycles
#for ti in range(1,10):
#for i,ti in enumerate(range(T_shift+1,len(DatList),ObsInterval)):
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
    
    for j,ei in enumerate(edge_new):
        EdgeList_new[j] = [ei.junc1,ei.junc2]
    
    """
    if i > 0:
        dE = EdgeList_new - EdgeList_old
        plt.title(ti)
        plt.plot(dE[:,0])
        plt.plot(dE[:,1])
        plt.savefig(SaveDirectory_p+"VID_{}.png".format(i))
        plt.close()
       
        
        if np.linalg.norm(dE) != 0:
            RearrangeRecorder.append(i)
            #sys.exit()
    """ 
    # Prediction
    Ff = SM@Fa
    Pf = SM@Pa@SM.T + Q
    #Pf = np.eye(Pa.shape[0]) * np.diagonal(Pf)
    
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
    TT_in = Data_old.Ft[:E_NUM]#[E_in]
    TP_in = Data_old.Ft[E_NUM:]#[E_NUM+np.array(C_in)]
    ET_in = np.array(Fa)[:E_NUM]#[E_in]
    EP_in = np.array(Fa)[E_NUM:]#[E_NUM+np.array(C_in)]
    coef_T = np.corrcoef(TT_in,ET_in)[0,1]
    T_cor_list.append(coef_T)
    coef_P = np.corrcoef(TP_in,EP_in)[0,1]
    P_cor_list.append(coef_P)
    
    for ei in E_in:
        V_list[ei,0] = Data_new.edge[ei].junc1
        V_list[ei,1] = Data_new.edge[ei].junc2


    if i%PlotInterval == 0:
        plt.figure(figsize=(18,12))
        plt.subplot(2,3,1)
        """
        plt.title("trace(Pa)@{}".format(ti))
        plt.plot(Pa_trace_list,label="Analysis")
        plt.plot(Pf_trace_list,label="Forecast")
        plt.legend()
        """
        Fnet_a = H@Fa
        Fnet_t = H@Data_old.Ft #True forces at the last time point
        vi = v
        xm=np.linspace(min(Fnet_t),max(Fnet_t),1000)
        #True
        [st,it] = np.polyfit(Fnet_t,vi,1)
        [sa,ia] = np.polyfit(Fnet_a,vi,1)
        [sat,iat] = np.polyfit(Fnet_t,Fnet_a,1)
        plt.scatter(Fnet_t,Fnet_a,c=range(len(Fnet_t)),label="Slope: {:.2f}".format(sat))
        
        
        #plt.xlim(-0.03,0.03)
        #plt.ylim(-0.03,0.03)
        plt.legend(loc='upper right')
        
        
        
        # Check correlation btw True vs estimated force
        plt.subplot(2,3,2)
        plt.scatter(Fnet_t,vi,marker=".",label="true")
        plt.plot(xm,st*xm+it,"b--",alpha=0.5,label="Slope: {:.2f}".format(st))
        #analysis
        
        plt.scatter(Fnet_a,vi,marker=".",label="analysis")
        plt.plot(xm,sa*xm+ia,"r--",alpha=0.5,label="Slope: {:.2f}".format(sa))
        plt.legend(loc='upper right')
        
        
        plt.subplot(2,3,3)
        
        SigTrace.append(np.trace(Pa))
        plt.plot(SigTrace)      
        """
        dF_all = np.zeros(V_NUM*2)
        #dF_all[V_in] = Fnet_t - Fnet_a
        dF_all = Fnet_t - Fnet_a
        
        dFe_list = []
        dist = np.array([e.dist for e in Data_old.edge])
        for e in Data_old.edge:
            dFe_i_junc1 = np.sqrt(dF_all[e.junc1]**2 + dF_all[e.junc1+V_NUM]**2)
            dFe_i_junc2 = np.sqrt(dF_all[e.junc2]**2 + dF_all[e.junc2+V_NUM]**2)
            dFe_i = (dFe_i_junc1 + dFe_i_junc2)/2
            dFe_list.append(dFe_i)
        dFe_list = np.array(dFe_list)
        
        plt.scatter(dist,ET_in - TT_in,c=range(E_NUM))
        #plt.scatter(dist[E_in],ET_in - TT_in,c=range(len(E_in)))
        #plt.scatter(np.log(dist[E_in]),TT_in - ET_in,c=range(len(E_in)))
        """
        
        plt.subplot(2,3,4)
        plt.title("CorrCoef@{}".format(ti))
        plt.plot(T_cor_list,label="Tension")
        plt.plot(P_cor_list,label="Pressure")
        plt.scatter(RearrangeRecorder,np.array(T_cor_list)[RearrangeRecorder],marker="x",color="r",label="Rearrange")
        plt.ylim(0,1.0)
        plt.legend(loc="upper right")
        
        plt.subplot(2,3,5)
        #plt.plot(Fnet_a - Fnet_t)
        #plt.ylim(-0.005,0.005)
        
        
        """ """
        plt.title("Corrcoef:{:.2f}@{}".format(coef_T,ti))
        plt.scatter(TT_in,ET_in,c=range(len(TT_in)))
        xm=np.linspace(min(TT_in),max(TT_in),1000)
        plt.plot(xm,xm,"r--",alpha=0.5)
        
        """ """
        
        """
        plt.title("Ta vs Tt@{}".format(ti))
        plt.ylim(0,0.75)
        plt.plot([tti[0] for tti in Ft_list],".",label="True")
        plt.plot([fai[0] for fai in Fa_list],label="Analysis")
        plt.legend()
        """
        """
        plt.title("CorrCoef@{}".format(ti))
        plt.plot(T_cor_list,label="Tension")
        plt.plot(P_cor_list,label="Pressure")
        plt.ylim(0,1.0)
        plt.legend()
        """
        """
        """
        
        plt.subplot(2,3,6)
        plt.title("Corrcoef:{:.2f}@{}".format(coef_P,ti))
        TP_in = TP_in - np.mean(TP_in)
        EP_in = EP_in - np.mean(EP_in)
        plt.scatter(TP_in,EP_in,c=range(len(TP_in)))
        xm=np.linspace(min(TP_in),max(TP_in),1000)
        plt.plot(xm,xm,"r--",alpha=0.5)
        
        
        """
        plt.title("Pa vs Pt@{}".format(ti))
        plt.ylim(0,0.75)
        plt.plot([tti[E_NUM+1] for tti in Ft_list],label="True")
        plt.plot([fai[E_NUM+1] for fai in Fa_list],label="Analysis")
        plt.legend()
        """

        
        plt.savefig(SaveDirectory_p+"{}.png".format(ti),dpi=72)
        plt.close()
        
        
        
        dT = np.zeros(E_NUM)
        dT = np.abs(ET_in - TT_in)
        #dT[E_in] = np.abs(ET_in - TT_in)
        
        MyPyLib.ForceInf_lib.Draw_Tension(Data_old.x,Data_old.y,dT,Data_old.edge,T_LINE_WIDTH = 2.0, tmin = 0,tmax= 0.3, savefile = '')
        plt.savefig(SaveDirectory_p+"dTmap_{}.png".format(ti),dpi=72)
        plt.close()
        
        
        
        EdgeList_old = copy.deepcopy(EdgeList_new)
        
        
        #sys.exit()
        
        
        if ti * SkipNum > (T_shift + DURATION + 1):
            sys.exit()
    #T_mem.append(Data_new.edge[812].TT)
    # Update variables
    MM_old = copy.deepcopy(MM_new)
    edge_old = copy.deepcopy(edge_new)
    cell_old = copy.deepcopy(cell_new)
    x_old = x_new.copy()
    y_old = y_new.copy()
    Data_old.save(SaveDirectory_b+"{}.dat".format(ti-1))
    Data_old = copy.deepcopy(Data_new)
    #del Data_new
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