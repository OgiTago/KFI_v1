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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import   Polygon



Normalize=True
#%% Input file
SampleName = "DH4_114"
SampleDirectory = "./output/{}/".format(SampleName)
        

BinaryDirectory = SampleDirectory + "bin/"
BinList = natsort.natsorted(glob.glob(BinaryDirectory+"*.dat"))
AnalysisDirectory = "{}/analysis/".format(SampleDirectory)
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

 
#%%

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


        Ta_var_list.append(np.diag(data.Pa)[:data.E_NUM])
        Pa_var_list.append(np.diag(data.Pa)[data.E_NUM:])

        
        # Plot true tensions         
        Draw_Tension(data.x,data.y,Tt_in,data.edge,T_LINE_WIDTH = 2.0, tmin = 0.3,tmax= 0.5, savefile = '',cmap="jet")
        plt.savefig(MapDirectory+"TmapT_{}.png".format(ti),dpi=150)
        plt.close()
        
        # Plot estimated pressure
        Draw_Tension(data.x,data.y,Ta_in,data.edge,T_LINE_WIDTH = 2.0, tmin = 0.3,tmax= 0.5, savefile = '',cmap="jet")
        plt.savefig(MapDirectory+"TmapA_{}.png".format(ti),dpi=150)        
        plt.close()

        # Save        
        Tt_mean.append(np.mean(Tt_in))
        Ta_mean.append(np.mean(Ta_in))
        Pt_mean.append(np.mean(Pt_in))
        Pa_mean.append(np.mean(Pa_in))
        Tcor_list.append(np.corrcoef(Tt_in[data.E_in],Ta_in[data.E_in])[0,1])
        Pcor_list.append(np.corrcoef(Pt_in[data.C_in],Pa_in[data.C_in])[0,1])
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


Tt_mean = np.array(Tt_mean)
Pt_mean = np.array(Pt_mean)
Ta_mean = np.array(Ta_mean)
Pa_mean = np.array(Pa_mean)


#%% Plot T-t 
Rescale=False
label_size=20
tick_size=18

if Rescale:
    TtDirectory=AnalysisDirectory+"Tt_r/"

else:
    TtDirectory=AnalysisDirectory+"Tt/"

os.makedirs(TtDirectory,exist_ok=True)
plt.close()
if Rescale:
    Ta_list = (Ta_list.T * Tt_mean/Ta_mean).T
time = np.linspace(1,len(Tt_list[:,0])/10,len(Tt_list[:,0]))
for i in data.E_in:
    plt.plot(time,Tt_list[:,i],label="True")
    plt.plot(time,Ta_list[:,i],label="Analysis")
    uns = 1.96 * np.sqrt(Ta_var_list[:,i])
    plt.fill_between(time,Ta_list[:,i]-uns, Ta_list[:,i]+uns, alpha=0.5,color="gray")
    plt.xlabel("Time",fontsize=label_size)
    plt.ylabel("Tension",fontsize=label_size)
    plt.tick_params(labelsize=tick_size)
    plt.savefig(TtDirectory+"{}.png".format(i),dpi=150,bbox_inches="tight")
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

    plt.plot(time,Pt_list[:,i],label="True",lw=3)
    plt.plot(time,Pa_list[:,i],label="Analysis",lw=3)
    uns = 1.96 * np.sqrt(Pa_var_list[:,i])
    plt.fill_between(time,Pa_list[:,i]-uns, Pa_list[:,i]+uns, alpha=0.5,color="gray",lw=3)
    plt.xlabel("Time",fontsize=label_size)
    plt.ylabel("Pressure",fontsize=label_size)
    plt.tick_params(labelsize=tick_size)

    if Normalize:
        plt.savefig(PtDirectory+"{}.png".format(i),dpi=150,bbox_inches="tight"))
    else:
        plt.savefig(PtDirectory+"{}_n.png".format(i),dpi=150,bbox_inches="tight"))
    plt.close()

#%% Correlation coefficient
plt.close()
label_size=20
tick_size=18

#plotting
plt.plot(time,Tcor_list,color="magenta",label="Tension",lw=3)
plt.plot(time,Pcor_list,color="cyan",label="Pressure",lw=3)

#axis
plt.ylim(0,1.05)

#label
plt.xlabel("Time",fontsize=label_size)
plt.ylabel("Correlation coefficient",fontsize=label_size)
plt.tick_params(labelsize=tick_size)

plt.legend(loc="lower right",fontsize=16)
plt.savefig(AnalysisDirectory+"CorrCoef.png",bbox_inches="tight")

#%% Save RMSE_T and RMSE_P
np.save(SampleDirectory+"Tcor_{}.npy".format(SampleName),np.array(Tcor_list))
np.save(SampleDirectory+"Pcor_{}.npy".format(SampleName),np.array(Pcor_list))






