#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:59:09 2023

@author: ogita
"""

import numpy as np
import pickle

def CalcV(x_old,y_old,x_new,y_new,Rnd):
    vx = x_new - x_old
    vy = y_new - y_old
    #vx = np.delete(vx,Rnd[0])
    #vy = np.delete(vy,Rnd[0])
    v = np.hstack((vx,vy))
    return v


class Data:
    Ff,Fa,Ft = np.full(3,np.nan)
    Pf,Pa = np.full(2,np.nan)
    vx, vy, vv, vabs = np.full(4,np.nan)
    K = np.nan
    trPf, trPa = np.full(2,np.nan)
    edge, cell = np.nan, np.nan
    V_in, E_in, C_in = [], [], []
    RndV, RndE, RndC = [], [], []
    F_in, V_in_xy = [], []
    CELL_NUMBER, V_NUM, E_NUM = np.nan, np.nan, np.nan
    title = ""
    MM = np.nan
        
    def __init__(self,data):
        x,y,edge,cell,Rnd,CELL_NUMBER,E_NUM,V_NUM,INV_NUM,R_NUM,stl,title = data
        self.x = x
        self.y = y
        self.edge = edge
        self.cell = cell
        self.CELL_NUMBER = CELL_NUMBER
        self.E_NUM = E_NUM
        self.V_NUM = V_NUM
        [self.RndV,self.RndE, self.RndC] = Rnd
        self.E_in = list(set(range(E_NUM)) -  set(Rnd[1]))
        self.C_in = list(set(range(CELL_NUMBER)) -  set(Rnd[2]))
        self.F_in = list(set(self.E_in)|set(self.C_in))
        self.V_in = list(set(range(V_NUM)) -  set(Rnd[0]))
        self.V_in_xy = self.V_in.extend(list(np.array(self.V_in)+V_NUM))

    def set_Ft(self,edge,cell):
        Tt = np.array([ei.TT for ei in edge])
        Pt = np.array([ci.TP for ci in cell])
        self.Ft = np.hstack((Tt,Pt))
    
    def set_forecast(self,Ff,Pf):
        self.Ff = Ff
        self.Pf = Pf
        self.trPf = np.trace(Pf)
    
    def set_KG(self,K):
        self.K = K
    
    def set_analysis(self,Fa,Pa):
        self.Fa = Fa
        self.Pa = Pa
        self.trPa = np.trace(Pa)
        
    def set_vf(self,v):
        self.vv = v
        self.vx = v[int(len(v)/2):]
        self.vy = v[:int(len(v)/2)]
        self.vabs = np.array(list(map(np.linalg.norm,np.array([self.vx.T,self.vy.T]).T)))
    
    def save(self,file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
#%% Functions for analysis
def calc_rmse_0(seq1,seq2):
    return np.sqrt(np.mean((seq1-seq2)**2))
def calc_rmse(ID,Fa_list,Ft_list):
    return calc_rmse_0(Fa_list[:,ID],Ft_list[:,ID])
