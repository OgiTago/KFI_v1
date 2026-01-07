#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:00:17 2023

@author: ogita
"""
import os
import natsort
import pickle
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import   Polygon


import MyPyLib.KF_functions as KF
import MyPyLib.ForceInf_lib as FI
import MyPyLib
import seaborn as sns
from pandas import DataFrame

spinup = 0
Normalize=True
PlotEdgeID = 753
PlotCellID = 151
#%%
if len(sys.argv)==1:
    #%% Input file
    SampleName = "DH4_113"
    SampleDirectory = "../{}/".format(SampleName)
    print(SampleDirectory)
    print(SampleName)  
    #SampleDirectory = "../20241119Dynamics_1stRun/{}/".format(SampleName)
    
else:
    args = sys.argv
    SampleName = args[2]
    SampleDirectory = args[1] + SampleName + "/"
    print(SampleDirectory)
    print(SampleName)
    

BinaryDirectory = SampleDirectory + "bin/"
BinList = natsort.natsorted(glob.glob(BinaryDirectory+"*.dat"))
AnalysisDirectory = "../{}_plotting/".format(SampleName)
os.makedirs(AnalysisDirectory,exist_ok=True)
MapDirectory = AnalysisDirectory + "Tmap/"
os.makedirs(MapDirectory,exist_ok=True)    
#%%
def ScaleForces(T,P):
    c = 1/np.mean(T)
    ST = c*T
    SP = c*P
    SP_mean = np.mean(SP)
    NP = SP - SP_mean
    return [ST,NP]

def CalcRMSE(TT,TP,ET,EP,E_in,C_in):
    [TT,TP] = ScaleForces(TT,TP)
    [ET,EP] = ScaleForces(ET,EP)
    Tdev = np.sum((ET - TT)**2)
    Pdev = np.sum((EP - TP)**2)
    RMSE = np.sqrt((Tdev + Pdev)/(len(E_in)+len(C_in)))
    return [TT,TP,ET,EP,RMSE]

def CalcRMSE2(Data):
    E_NUM = Data.E_NUM
    return CalcRMSE(Data.Ft[:E_NUM],Data.Ft[E_NUM:],Data.Fa[:E_NUM],Data.Fa[E_NUM:],Data.E_in,Data.C_in)
    
#%%
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import   Polygon
from matplotlib.collections import PatchCollection


def Draw_Tension(x,y,T,edge,T_LINE_WIDTH = 2.0, tmin = 0,tmax=2.0, savefile = '',cmap=matplotlib.cm.gist_heat):
    print('...    Show tensions    ')
    #plt.rcParams['figure.figsize'] = 16,16
    fT, ax = plt.subplots(figsize=(16,16))
    fT.patch.set_facecolor('white')
    # maxT = max(T)
    # minT = min(T)
    lines = []
    colors = []

    for (i,e) in enumerate(edge):
        lines.append( [(e.x1,e.y1), (e.x2,e.y2)] )
        colors.append( T[i] )

    line_segments = LineCollection( lines, linewidths=T_LINE_WIDTH, linestyles='solid', cmap = cmap)
    line_segments.set_array( np.array(colors) )
    line_segments.set_clim([tmin, tmax])
    ax.add_collection(line_segments)
    ax.autoscale_view()
    ax.set_aspect('equal', 'datalim')
    plt.axis('off')
    #fT.colorbar(line_segments, ax=ax, orientation='vertical')

    #fT.show()
    if savefile != '':
        fT.savefig(savefile)



def Draw_Pressure(x,y,P,edge,cell,pmin,pmax,savefile = ''):
    print('...    Show pressures  ')
    fP, ax = plt.subplots(1,figsize=(16,16))
    fP.patch.set_facecolor('white')
    #maxP = max(P);
    #minP = min(P);
    patches = []
    colors = []

    for (i,e) in enumerate(edge):
        plt.plot([e.x1,e.x2],[e.y1,e.y2],linewidth=1,color = "0.2")

    for (i,c) in enumerate(cell):
        if  c.in_out == 'i':
            pg = np.vstack( (x[c.junc], y[c.junc]) ).T
            polygon = Polygon( pg, closed = True )
            patches.append(polygon)
            colors.append(P[i])

    collection = PatchCollection(patches, linewidths=0.5, cmap = matplotlib.cm.cool)
    collection.set_array( np.array(colors) )
    collection.set_clim([pmin, pmax])
    ax.add_collection( collection )
    ax.autoscale_view()
    ax.set_aspect('equal', 'datalim')
    plt.axis('off')

    #fP.colorbar(collection, ax=ax,orientation='horizontal')
    #fP.show()
    if savefile != '':
        fP.savefig( savefile)

def get_min_max(list1,list2,list_in,margin=0.05):
    list1 = list1[list_in]
    list2 = list2[list_in]
    min1,max1 = min(list1),max(list1)
    min2,max2 = min(list2),max(list2)
    margin1 = margin * (max1 - min1)
    margin2 = margin * (max2 - min2)    
    mmin,mmax = min([min1,min2]),max([max1,max2])
    return [min1,max1,margin1,min2,max2,margin2,mmin,mmax]
#%%
#Shape
l_list = []
A_list = []

#Force
Tt_list = []
Ta_list = []
Pt_list = []
Pa_list = []

#Variance
Ta_var_list = []
Pa_var_list = []



Tt_mean = []
Ta_mean = []
Pt_mean = []
Pa_mean = []

Tcor_list = []
Pcor_list = []
RMSE_list = []

#%%
for ti,bi in enumerate(BinList):
    with open(bi,"rb") as f:
        print("#Now loading: {}".format(bi))
        data = pickle.load(f)
        Tt_in = data.Ft[:data.E_NUM]
        Ta_in = data.Fa[:data.E_NUM]

        Pt_in = data.Ft[data.E_NUM:]
        Pa_in = data.Fa[data.E_NUM:]

        Tt_list.append(Tt_in)
        Ta_list.append(Ta_in)
                
        Pt_list.append(Pt_in)
        Pa_list.append(Pa_in)

        l_list.append([ei.dist for ei in data.edge])
        A_list.append([ci.area for ci in data.cell])

        Ta_var_list.append(np.diag(data.Pa)[:data.E_NUM])
        Pa_var_list.append(np.diag(data.Pa)[data.E_NUM:])

        
        
        """
        plt.title(np.corrcoef(Tt_in[data.E_in],Ta_in[data.E_in]))
        plt.scatter(Tt_in[data.E_in],Ta_in[data.E_in])
        plt.pause(np.nan)
        """
        
        Draw_Tension(data.x,data.y,Tt_in,data.edge,T_LINE_WIDTH = 2.0, tmin = 0.3,tmax= 0.5, savefile = '',cmap="jet")
        plt.savefig(MapDirectory+"TmapT_{}.png".format(ti),dpi=150)
        plt.close()
        
        Draw_Tension(data.x,data.y,Ta_in,data.edge,T_LINE_WIDTH = 2.0, tmin = 0.3,tmax= 0.5, savefile = '',cmap="jet")
        plt.savefig(MapDirectory+"TmapA_{}.png".format(ti),dpi=150)
        
        plt.close()
        
        Tt_mean.append(np.mean(Tt_in))
        Ta_mean.append(np.mean(Ta_in))
        Pt_mean.append(np.mean(Pt_in))
        Pa_mean.append(np.mean(Pa_in))
        Tcor_list.append(np.corrcoef(Tt_in[data.E_in],Ta_in[data.E_in])[0,1])
        Pcor_list.append(np.corrcoef(Pt_in[data.C_in],Pa_in[data.C_in])[0,1])
        RMSE_list.append(CalcRMSE2(data))
E_in = data.E_in
C_in = data.C_in
#%% colorbar: Tension

norm = matplotlib.colors.Normalize(vmin=0.3, vmax=0.5)
fig, cbar = plt.subplots(figsize=(0.5, 5))
cmap = plt.get_cmap("jet")
matplotlib.colorbar.Colorbar(
    ax=cbar,
    mappable=matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    orientation="vertical",
)#.set_label("Tension", fontsize=20)

plt.savefig(MapDirectory+"T_colorbar.png",dpi=150,bbox_inches="tight")


#%%
Tt_list = np.array(Tt_list)
Ta_list = np.array(Ta_list)
Ta_var_list = np.array(Ta_var_list)

Pt_list = np.array(Pt_list)
Pa_list = np.array(Pa_list)
Pa_var_list = np.array(Pa_var_list)


l_list = np.array(l_list)
A_list = np.array(A_list)

Tt_mean = np.array(Tt_mean)
Pt_mean = np.array(Pt_mean)
Ta_mean = np.array(Ta_mean)
Pa_mean = np.array(Pa_mean)


#%% Plotting for figure
Fig2Directory = AnalysisDirectory+"Fig2/"
os.makedirs(Fig2Directory,exist_ok=True)    


# %% get min,max and margin
[Tt_min,Tt_max,Tt_margin,Ta_min,Ta_max,Ta_margin,Tmin,Tmax] = get_min_max(Tt_in,Ta_in,data.E_in)

#%% Tmap: GT
# plot
Draw_Tension(data.x,data.y,Tt_in,data.edge,T_LINE_WIDTH = 2.0, tmin = Tmin,tmax= Tmax, savefile = '',cmap="jet")

# axis
plt.xlim(1.05*min(data.x),1.05*max(data.x))
plt.ylim(1.05*min(data.y),1.05*max(data.y))

# save
plt.savefig(Fig2Directory+"Tmap_T_t{}.pdf".format(ti),bbox_inches="tight")

#%% Tmap: KFI
Draw_Tension(data.x,data.y,Ta_in,data.edge,T_LINE_WIDTH = 2.0, tmin = Tmin,tmax= Tmax, savefile = '',cmap="jet")
# axis
plt.xlim(1.05*min(data.x),1.05*max(data.x))
plt.ylim(1.05*min(data.y),1.05*max(data.y))
# Point junction to plot edge tension
e = data.edge[PlotEdgeID]
xc = (data.x[e.junc1] + data.x[e.junc2])/2
yc = (data.y[e.junc1] + data.y[e.junc2])/2 - 0.2
plt.text(xc,yc,"*",ha="center",va="center_baseline",color="k",fontsize=72)

plt.savefig(Fig2Directory+"Tmap_A_t{}.pdf".format(ti),bbox_inches="tight")


# %% get min,max and margin
[Pt_min,Pt_max,Pt_margin,Pa_min,Pa_max,Pa_margin,Pmin,Pmax] = get_min_max(Pt_in,Pa_in,data.C_in)

#%% Pmap: GT
# plot
Draw_Pressure(data.x,data.y,Pt_in,data.edge,data.cell,pmin = Pmin,pmax= Pmax)

# axis
plt.xlim(1.05*min(data.x),1.05*max(data.x))
plt.ylim(1.05*min(data.y),1.05*max(data.y))

# save
plt.savefig(Fig2Directory+"Pmap_T_t{}.pdf".format(ti),bbox_inches="tight")

#%% Pmap: KFI

# plot
Draw_Pressure(data.x,data.y,Pa_in,data.edge,data.cell,pmin = Pmin,pmax= Pmax)

# axis
plt.xlim(1.05*min(data.x),1.05*max(data.x))
plt.ylim(1.05*min(data.y),1.05*max(data.y)) 

c = data.cell[PlotCellID]
xc = np.mean(data.x[c.junc])
yc = np.mean(data.y[c.junc]) -0.2
plt.text(xc,yc,"*",ha="center",va="center_baseline",color="k",fontsize=72)

# save
plt.savefig(Fig2Directory+"Pmap_A_t{}.pdf".format(ti),bbox_inches="tight")

#%% Edge and Cell ID
# plot

cfig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
for i,e in enumerate(data.edge):
    tx = [data.x[e.junc1], data.x[e.junc2]]
    ty = [data.y[e.junc1], data.y[e.junc2]]
    plt.plot( tx, ty, '-k' ,linewidth=1,color="gray",alpha=0.5)
    xc = (data.x[e.junc1] + data.x[e.junc2])/2
    yc = (data.y[e.junc1] + data.y[e.junc2])/2 - 0.2
    plt.text(xc,yc,str(i),ha="center",va="center",color="red",fontsize=2)

for i,c in enumerate(data.cell):
    xc = np.mean(data.x[c.junc])
    yc = np.mean(data.y[c.junc])
    plt.text(xc,yc,str(i),ha="center",va="center",color="blue",fontsize=2)

cfig.show()
plt.xlim(1.05*min(data.x),1.05*max(data.x))
plt.ylim(1.05*min(data.y),1.05*max(data.y))
plt.axis('off')

# save
plt.savefig(Fig2Directory+"ID_t{}.pdf".format(ti),bbox_inches="tight")

#%% colorbar: Tension

norm = matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
fig, cbar = plt.subplots(figsize=(0.5, 5))
cmap = plt.get_cmap("jet")
matplotlib.colorbar.Colorbar(
    ax=cbar,
    mappable=matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    orientation="vertical",
)#.set_label("Tension", fontsize=20)

plt.savefig(Fig2Directory+"T_colorbar.pdf",bbox_inches="tight")

#%% colorbar: Pressure
norm = matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
fig, cbar = plt.subplots(figsize=(0.5, 5))
cmap = plt.get_cmap("cool")
matplotlib.colorbar.Colorbar(
    ax=cbar,
    mappable=matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    orientation="vertical",
)#.set_label("Pressure", fontsize=20)

plt.savefig(Fig2Directory+"P_colorbar.pdf",bbox_inches="tight")
#%% Tension: Corr coef
label_size = 24
r_size = 24
tick_size = 16
plt.figure(figsize=(8,8))

# scatter plot
plt.scatter(Tt_in[data.E_in],Ta_in[data.E_in],c="k")

# y=x line
x = np.linspace(Tt_min-Tt_margin,Tt_max+Tt_margin,1000)
plt.plot(x,x,ls="--",c="r")

# corr coef
r_t = np.corrcoef(Tt_in[data.E_in],Ta_in[data.E_in])[0,1] 
plt.text(min(Tt_in[data.E_in]),max(Ta_in[data.E_in])-0.025,"$r={:.2f}$".format(r_t),fontsize=r_size)

#axis
plt.xlim(Tt_min - Tt_margin,Tt_max + Tt_margin)
plt.ylim(Ta_min - Ta_margin,Ta_max + Ta_margin)


#label
plt.xlabel("Ground truth",fontsize=label_size)
plt.ylabel("KFI",fontsize=label_size)
plt.tick_params(labelsize=tick_size)

#save
plt.savefig(Fig2Directory+"Tcorr_t{}.pdf".format(ti),bbox_inches="tight")

#%% Pressure: Corr coef
plt.figure(figsize=(8,8))

# scatter plot
plt.scatter(Pt_in[data.C_in],Pa_in[data.C_in],c="k")

# y=x line
x = np.linspace(Pt_min-Pt_margin,Pt_max+Pt_margin,1000)
plt.plot(x,x,ls="--",c="r")

# corr coef
r_t = np.corrcoef(Pt_in[data.C_in],Pa_in[data.C_in])[0,1] 
plt.text(min(Pt_in[data.C_in]),max(Pa_in[data.C_in]-0.025),"$r={:.2f}$".format(r_t),fontsize=r_size)

#axis
plt.xlim(Pt_min - Pt_margin,Pt_max + Pt_margin)
plt.ylim(Pa_min - Pa_margin,Pa_max + Pa_margin)


#label
plt.xlabel("Ground truth",fontsize=label_size)
plt.ylabel("KFI",fontsize=label_size)
plt.tick_params(labelsize=tick_size)

#save
plt.savefig(Fig2Directory+"Pcorr_t{}.pdf".format(ti),bbox_inches="tight")

#%% Timecourse of mean tension and pressure
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Mean tension: {}".format(SampleName))
plt.plot(Tt_mean,label="True")
plt.plot(Ta_mean,label="Analysis")
plt.legend()
plt.subplot(1,2,2)
plt.title("Mean pressure: {}".format(SampleName))
plt.plot(Pt_mean,label="True")
plt.plot(Pa_mean,label="Analysis")
plt.legend()
plt.savefig(AnalysisDirectory+"MeanForce.png",dpi=150)
plt.close()

#%% Plot T-t 
Rescale=False
if Rescale:
    TtDirectory=AnalysisDirectory+"Tt_r/"

else:
    TtDirectory=AnalysisDirectory+"Tt/"

os.makedirs(TtDirectory,exist_ok=True)
RMSE_T = []
RMSE_T_spinup = []
plt.close()
if Rescale:
    Ta_list = (Ta_list.T * Tt_mean/Ta_mean).T
time = np.linspace(1,len(Tt_list[:,0])/10,len(Tt_list[:,i]))
for i in data.E_in:
    """
    if i == PlotEdgeID:
        pass
    else:
        continue
    """
    rmse_i = KF.calc_rmse_0(Tt_list[:,i],Ta_list[:,i])
    rmse_i_spinup = KF.calc_rmse_0(Tt_list[spinup:,i],Ta_list[spinup,i])
    RMSE_T.append(rmse_i)
    RMSE_T_spinup.append(rmse_i_spinup)
    #plt.title("Edge tension\nID:{}; RMSE:{:.2f}".format(i,rmse_i))
    plt.plot(time,Tt_list[:,i],label="True")
    plt.plot(time,Ta_list[:,i],label="Analysis")
    uns = 1.96 * np.sqrt(Ta_var_list[:,i])
    plt.fill_between(time,Ta_list[:,i]-uns, Ta_list[:,i]+uns, alpha=0.5,color="gray")
    #plt.legend()
    plt.savefig(TtDirectory+"{}.png".format(i),dpi=150)
    if i == PlotEdgeID:
        label_size=16
        tick_size=12
        plt.xlabel("Time",fontsize=label_size)
        plt.ylabel("Tension",fontsize=label_size)
        plt.tick_params(labelsize=tick_size)

        plt.savefig(Fig2Directory+"T{}.pdf".format(i),bbox_inches="tight")
    plt.close()

RMSE_T = np.array(RMSE_T)
RMSE_T_spinup = np.array(RMSE_T_spinup)
#%% Spatial distribution of Ta_VAR_MEAN
Ta_var_mean = np.mean(Ta_var_list,axis=0)
Draw_Tension(data.x,data.y,Ta_var_mean,data.edge,tmin=0,tmax=0.001,cmap="jet")
plt.close()
#%%
plt.scatter(RMSE_T,Ta_var_mean[data.E_in])
plt.xlim(0,0.02)
plt.ylim(0,0.005)

#%% Distribution of RMSE_T
max_ID = data.E_in[np.argmax(RMSE_T)]
min_ID = data.E_in[np.argmin(RMSE_T)]
plt.title("RMSE_T: {},\nmean={:.2f}, median={:.2f}\nmin={:.2f}(ID:{}); max={:.2f}(ID:{})".format(SampleName,np.mean(RMSE_T),np.median(RMSE_T),RMSE_T[min_ID],min_ID,RMSE_T[max_ID],max_ID))
plt.hist(RMSE_T)
plt.savefig(AnalysisDirectory + "RMSE_T_dist.png",dpi=150)
plt.close()

#%% Correlation btw mean or std of edge length vs RMSE of tension
rmse_plot_range = [0,0.05]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
lmean = np.mean(l_list,axis=0)
lmean_in = lmean[data.E_in]
plt.title("RMSE vs mean length")
plt.scatter(lmean_in,RMSE_T)
plt.ylim(rmse_plot_range)
plt.xlabel("l_mean")
plt.ylabel("RMSE")

plt.subplot(1,2,2)
lsd = np.std(l_list,axis=0)
lsd_in = lsd[data.E_in]
plt.title("RMSE vs length std")
plt.scatter(lsd_in,RMSE_T)
plt.ylim(rmse_plot_range)
plt.xlabel("l_std")
plt.ylabel("RMSE")
plt.close()

#%% Spatial distribution of  RMSE (Tension) 
RMSE_ALL = []
j=0
for i in range(data.E_NUM):
    if i in data.E_in:
        RMSE_ALL.append(RMSE_T[j])
        j+=1
    else:
        RMSE_ALL.append(0)
#plt.plot(RMSE_ALL)
FI.Draw_Tension(data.x,data.y,RMSE_ALL, data.edge,tmin=min(RMSE_ALL),tmax=0.01)
plt.title("RMSE (Tension): {} ".format(SampleName))
plt.savefig(AnalysisDirectory+"RMSE_T_spatial_dist.png",dpi=150)
plt.close()
#%% Plot P-t
if Normalize:
    PtDirectory=AnalysisDirectory+"Pt_norm/"
else:
    PtDirectory=AnalysisDirectory+"Pt/"
os.makedirs(PtDirectory,exist_ok=True)
RMSE_P = []
RMSE_P_spinup = []
if Normalize:
    Pt_list = (Pt_list.T - Pt_mean.T).T
    Pa_list = (Pa_list.T - Pa_mean.T).T

if Rescale:
    Pa_list = (Pa_list.T * (Tt_mean/Ta_mean)).T
for i in data.C_in:
    if i == PlotCellID:
        pass
    else:
        continue
    rmse_i = KF.calc_rmse_0(Pt_list[:,i],Pa_list[:,i])
    rmse_i_spinup = KF.calc_rmse_0(Pt_list[spinup:,i],Pa_list[spinup:,i])
    RMSE_P.append(rmse_i)
    RMSE_P_spinup.append(rmse_i_spinup)
    #plt.title("Cell pressure\nID:{}; RMSE:{:.2f}".format(i,rmse_i))
    plt.plot(time,Pt_list[:,i],label="Ground truth")
    plt.plot(time,Pa_list[:,i],label="KFI")
    uns = 1.96 * np.sqrt(Pa_var_list[:,i])
    plt.fill_between(time,Pa_list[:,i]-uns, Pa_list[:,i]+uns, alpha=0.5,color="gray")
    #plt.plot(A_list[:,i],label="Area")
    #plt.legend()
    if Normalize:
        plt.savefig(PtDirectory+"{}.png".format(i),dpi=150)
    else:
        plt.savefig(PtDirectory+"{}_n.png".format(i),dpi=150)
    if i == PlotCellID:
        label_size=16
        tick_size=12
        #label
        plt.xlabel("Time",fontsize=label_size)
        plt.ylabel("Pressure",fontsize=label_size)
        plt.tick_params(labelsize=tick_size)
        
        plt.savefig(Fig2Directory+"P{}.pdf".format(i),bbox_inches="tight")
    plt.close()

#%% Distribution of RMSE_P
max_ID = data.E_in[np.argmax(RMSE_P)]
min_ID = data.E_in[np.argmin(RMSE_P)]
plt.title("RMSE_P: {}\nmean={:.2f}, median={:.2f})\nmin={:.2f}(ID:{}); max={:.2f}(ID:{})".format(SampleName,np.mean(RMSE_P),np.median(RMSE_P),RMSE_P[min_ID],min_ID,RMSE_P[max_ID],max_ID))
plt.hist(RMSE_P)
if Normalize:
    plt.savefig(AnalysisDirectory + "RMSE_P_dist_n.png",dpi=150)
else:
    plt.savefig(AnalysisDirectory + "RMSE_P_dist.png",dpi=150)    
plt.close()
#%% Correlation btw mean or std of edge length vs RMSE of pressure
#rmse_plot_range = [0,max(RMSE_P)]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
Amean = np.mean(A_list,axis=0)
Amean_in = Amean[data.C_in]
plt.title("RMSE vs mean area")
plt.scatter(Amean_in,RMSE_P)
#plt.ylim(rmse_plot_range)
plt.xlabel("A_mean")
plt.ylabel("RMSE")

plt.subplot(1,2,2)
Asd = np.std(A_list,axis=0)
Asd_in = lsd[data.C_in]
plt.title("RMSE vs area std")
plt.scatter(Asd_in,RMSE_P)
#plt.ylim(rmse_plot_range)
plt.xlabel("A_std")
plt.ylabel("RMSE")
if Normalize:
    plt.savefig(AnalysisDirectory+"RMSE_PvsA_n.png",dpi=150)
else:
    plt.savefig(AnalysisDirectory+"RMSE_PvsA.png",dpi=150)
plt.close()

#%% Spatial distribution of  RMSE (Tension) 
RMSE_P_ALL = []
j=0
for i in range(data.CELL_NUMBER):
    if i in data.C_in:
        RMSE_P_ALL.append(RMSE_P[j])
        j+=1
    else:
        RMSE_P_ALL.append(0)
#plt.plot(RMSE_ALL)
FI.Draw_Pressure(data.x,data.y,RMSE_P_ALL, data.edge,data.cell,pmin=min(RMSE_P_ALL),pmax=max(RMSE_P_ALL))
plt.title("RMSE (Pressure): {}".format(SampleName))
if Normalize:
    plt.savefig(AnalysisDirectory+"RMSE_P_spatial_dist_n.png",dpi=150)
else:
    plt.savefig(AnalysisDirectory+"RMSE_P_spatial_dist.png",dpi=150)
plt.close()
#%% RMSE vs polygon number
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("RMSE_P vs polygon number")
polygon = np.array([ci.jnum for ci in data.cell])
polygon_in = polygon[data.C_in]
plt.scatter(polygon_in,RMSE_P)
plt.subplot(1,2,2)
plt.title("Area vs polygon number")
plt.scatter(polygon_in,Amean_in)
if Normalize:
    plt.savefig(AnalysisDirectory+"RMSE_P_polygon_n.png",dpi=150)
else:
    plt.savefig(AnalysisDirectory+"RMSE_P_polygon.png",dpi=150)
plt.close()


#%% Find the cell whose RMSE_P is median
for i,ri in enumerate(RMSE_P): 
    if round(ri,4) == round(np.median(RMSE_P),4):
        print(i)
#%% Heatmap tension
Cutoff=0
rowcol=(3,1)
plt.figure(figsize=(6,12))
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.subplot(rowcol[0],rowcol[1],1)
plt.title("True tension")
sns.heatmap(Tt_list[Cutoff:,:],cmap="bwr")
plt.subplot(rowcol[0],rowcol[1],2)
plt.title("Estimated tension")
sns.heatmap(Ta_list[Cutoff:,:],vmin=np.min(Tt_list[Cutoff:,:]),vmax=np.max(Tt_list[Cutoff:,:]),cmap="bwr")
plt.subplot(rowcol[0],rowcol[1],3)
plt.title("Estimated tension - true tension")
sns.heatmap((Ta_list-Tt_list)[Cutoff:,:],cmap="bwr",center=0)
plt.savefig(AnalysisDirectory+"Heatmap_T.png",dpi=150)
#%% Heatmap pressure
plt.figure(figsize=(6,12))
plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.subplot(rowcol[0],rowcol[1],1)
plt.title("True pressure")
sns.heatmap(Pt_list[Cutoff:,:],cmap="bwr")
plt.subplot(rowcol[0],rowcol[1],2)
plt.title("Estimated pressure")
sns.heatmap(Pa_list[Cutoff:,:],vmin=np.min(Pt_list[Cutoff:,:]),vmax=np.max(Pt_list[Cutoff:,:]),cmap="bwr")
plt.subplot(rowcol[0],rowcol[1],3)
plt.title("Estimated pressure - true pressure")
sns.heatmap((Pa_list-Pt_list)[Cutoff:,:],cmap="bwr",center=0)
plt.savefig(AnalysisDirectory+"Heatmap_P.png",dpi=150)
plt.close()
#%% Correlation coefficient
plt.close()
label_size=16
tick_size=12

#plotting
plt.plot(time,Tcor_list,color="magenta",label="Tension")
plt.plot(time,Pcor_list,color="cyan",label="Pressure")

#axis
plt.ylim(0,1.05)

#label
plt.xlabel("Time",fontsize=label_size)
plt.ylabel("Correlation coefficient",fontsize=label_size)
plt.tick_params(labelsize=tick_size)

plt.legend(loc="lower right")
plt.savefig(Fig2Directory+"CorrCoef.pdf",bbox_inches="tight")

#%% Save RMSE_T and RMSE_P
#np.save(SampleDirectory+"RMSE_T_{}.npy".format(ObsInterval),RMSE_T)
#np.save(SampleDirectory+"RMSE_P_{}.npy".format(ObsInterval),RMSE_P)
np.save(SampleDirectory+"Tcor_{}.npy".format(SampleName),np.array(Tcor_list))
np.save(SampleDirectory+"Pcor_{}.npy".format(SampleName),np.array(Pcor_list))
np.save(SampleDirectory+"RMSE_{}.npy".format(SampleName),np.array(RMSE_list))

#%% Output RMSE to CSV
RMSE_T_mean = np.mean(RMSE_T)
RMSE_P_mean = np.mean(RMSE_P)
RMSE_T_mean_spinup = np.mean(RMSE_T_spinup)
RMSE_P_mean_spinup = np.mean(RMSE_P_spinup)
"""
f_rmse = open("{}/RMSE.csv".format(SampleParentDirectory ),"a")
f_rmse.write("{},{},{},{},{}\n".format(ID,RMSE_T_mean,RMSE_P_mean,RMSE_T_mean_spinup,RMSE_P_mean_spinup))
f_rmse.close()
"""






