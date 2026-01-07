# -*- coding: utf8 -*-
"""
Created on Wed Jun 12 17:51:26 2019

@author: Goshi
"""
import numpy as np
import copy as cp

import MyPyLib.ForceInf_lib as FI


def scale_converter(x,y,edge,cell,sc_mean = 1.0,sc = 0,flip=""):
    E_NUM = len(edge)
    CELL_NUMBER = len(cell)
    
    dist_error = 1.0e-10
    
    if(sc == 0):
        lmean = np.mean([edge[i].dist for i in range(E_NUM)])
        sc = sc_mean/lmean
    if(flip == "v"):
        xsc = -sc*x
        ysc = sc*y
    elif(flip == "h"):
        xsc = sc*x
        ysc = -sc*y
    else:
        xsc = sc*x
        ysc = sc*y
    
    edgesc = cp.deepcopy(edge)
    for i in range(E_NUM):
        edgesc[i].x1 = xsc[edgesc[i].junc1]
        edgesc[i].y1 = ysc[edgesc[i].junc1]
        edgesc[i].x2 = xsc[edgesc[i].junc2]
        edgesc[i].y2 = ysc[edgesc[i].junc2]
        edgesc[i].set_distance(x,y)
        if( edgesc[i].dist - sc*edge[i].dist > dist_error):
            print("dist_error = %f > %f" %(edge[i].dist - sc*edge[i].dist,dist_error))
        
    cellsc = cp.deepcopy(cell)
    for i in range(CELL_NUMBER):
        cellsc[i].area = sc*sc*cell[i].area
        
    return [xsc,ysc,edgesc,cellsc,sc]

#def force_scale_converter(x,y,edge,cell,T,P,sc_mean = 1.0):