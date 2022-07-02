# -*- coding: utf-8 -*-
"""
test_problem.py
Copyright (c) 2022 Nobuo Namura
This code is released under the MIT License, see LICENSE.txt.

This Python code is for multi-objective Bayesian optimization (MBO) with/without constraint handling.
MBO part is based on MBO-EPBII-SRVA and MBO-EPBII published in the following articles:
・N. Namura, "Surrogate-Assisted Reference Vector Adaptation to Various Pareto Front Shapes 
  for Many-Objective Bayesian Optimization," IEEE Congress on Evolutionary Computation, 
  Krakow, Poland, pp.901-908, 2021.
・N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary 
  Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary 
  Computation, vol. 21, no. 6, pp. 898-913, 2017.
Please cite the article(s) if you use the code.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import functools

#======================================================================
def sphere(x, nf=1, ng=0):
    x = np.array(x)
    f = np.dot(x,x)
    return f

#======================================================================
def DTLZ1(x, nf=3, ng=0):
    x = np.array(x)
    
    g = 100.0*(float(len(x[nf-1:])) + np.sum((x[nf-1:]-0.5)**2.0 - np.cos(20.0*np.pi*(x[nf-1:]-0.5))))
    f = np.full(nf, 0.5*(1.0+g))
    for i in range(nf):
        f[i] *= np.prod(x[:nf-i-1])
        if i > 0:
            f[i] *= 1.0-x[nf-i-1]
    
    return f

#======================================================================
def DTLZ2(x, nf=3, ng=0):
    x = np.array(x)
    
    g = np.sum((x[nf-1:] - 0.5)**2.0)
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ3(x, nf=3, ng=0):
    x = np.array(x)
    
    g = 100.0*(float(len(x[nf-1:])) + np.sum((x[nf-1:]-0.5)**2.0 - np.cos(20.0*np.pi*(x[nf-1:]-0.5))))
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ4(x, nf=3, ng=0):
    x = np.array(x)
    alpha = 100.0
    
    g = np.sum((x[nf-1:]-0.5)**2.0)
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]**alpha))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1]**alpha)
    
    return f

#======================================================================
def DTLZ5(x, nf=3, ng=0):
    x = np.array(x)
    
    g = np.sum((x[nf-1:]-0.5)**2.0)
    theta = 0.25*np.pi/(1.0 + g)*(1.0 + 2.0*g*x)
    theta[0] = 0.5*np.pi*x[0]
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(theta[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(theta[nf-i-1])
    
    return f

#======================================================================
def DTLZ6(x, nf=3, ng=0):
    x = np.array(x)
    
    g = np.sum(x[nf-1:]**0.1)
    theta = 0.25*np.pi/(1.0 + g)*(1.0 + 2.0*g*x)
    theta[0] = 0.5*np.pi*x[0]
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(theta[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(theta[nf-i-1])
    
    return f

#======================================================================
def DTLZ7(x, nf=3, ng=0):
    x = np.array(x)
    
    g = 1.0 + 9.0/float(len(x[nf-1:]))*np.sum(x[nf-1:])
    h = float(nf)
    f = np.zeros(nf)
    for i in range(nf-1):
        f[i] = x[i]
        h -= (1.0 + np.sin(3.0*np.pi*f[i]))*f[i]/(1.0 + g)
    f[-1] = (1.0 + g)*h
    
    return f

#======================================================================
def DTLZ2max1(x, nf=3, ng=0):
    x = np.array(x)
    
    g = np.sum(1.0 - 4.0*(x[nf-1:]-0.5)**2.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ2max2(x, nf=3, ng=0):
    x = np.array(x)
    x[:nf-1] = 0.25 + 0.5*x[:nf-1]
    
    g = np.sum(1.0 - 4.0*(x[nf-1:]-0.5)**2.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ2max3(x, nf=3, ng=0):
    x = np.array(x)
    x[:nf-1] = 0.25 + 0.5*x[:nf-1]
    
    g = np.sum(1.0 - (x[nf-1:]-0.5)**2.0 + (np.cos(4.0*np.pi*(x[nf-1:]-0.5))-1.0)/3.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def ZDT1(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - np.sqrt(f[0]/g))
    
    return f

#======================================================================
def ZDT2(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - (f[0]/g)**2.0)
    
    return f

#======================================================================
def ZDT3(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - np.sqrt(f[0]/g) - x[0]/g*np.sin(10.0*np.pi*x[0]))
    
    return f

#======================================================================
def ZDT4(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 10.0*float(len(x[1:])) + np.sum(x[1:]**2.0 - 10.0*np.cos(4.0*np.pi*x[1:]))
    f[1] = g*(1.0 - np.sqrt(f[0]/g))
    
    return f

#======================================================================
def ZDT6(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = 1.0 - np.exp(-4.0*x[0])*(np.sin(6.0*np.pi*x[0]))**6.0
    g = 1.0 + 9.0*(np.sum(x[1:])/float(len(x[1:])))**0.25
    f[1] = g*(1.0 - (f[0]/g)**2.0)
    
    return f

#======================================================================
def LZ08F1(x, nf=2, ng=0):
    x = np.array(x)
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-x[0]**(0.5+3*(i-2)/(2*(nx-2))))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-x[0]**(0.5+3*(i-2)/(2*(nx-2))))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F2(x, nf=2, ng=0):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F3(x, nf=2, ng=0):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-0.8*np.cos(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-0.8*np.cos(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F4(x, nf=2, ng=0):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-0.8*x[0]*np.cos(2.0*np.pi*x[0]+i*np.pi/(3*nx)))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-0.8*x[0]*np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
#Scalable Gaussian Mixture
def SGM(x, nf=3, ng=0, seed=1, nb=10):
    x = np.array(x)
    nx = len(x)
    seed0 = np.random.randint(low=0,high=2**31-1)
    np.random.seed(seed)
    xc = np.random.rand(nf+ng,nb,nx)
    sc = np.sqrt(nx*0.5)*np.random.uniform(low=0.2,high=0.4,size=[nf+ng,nb])
    hc = 1.0/(0.3*float(nb))*np.random.normal(loc=-0.3,scale=0.5,size=[nf+ng, nb])   
    f = np.zeros(nf+ng)
    for j in range(nf+ng):
        for i in range(nb):
            dist = np.dot(x - xc[j,i,:], x - xc[j,i,:])
            f[j] += hc[j,i]*np.exp(-(dist/sc[j,i]**2.0))
    f += 0.15
    np.random.seed(seed0)
    return f

#======================================================================
if __name__ == "__main__":
    func_name = 'SGM'
    seed = 3
    nx = 2
    nf = 2
    ng = 1
    xmin = np.full(nx, 0.0)
    xmax = np.full(nx, 1.0)
    if func_name=='SGM':
        func = functools.partial(eval(func_name), nf=nf+ng, seed=seed)
    else:
        func = functools.partial(eval(func_name), nf=nf+ng)
    
    if nx ==2:
        Fs = []
        Gs = []
        x = xmin[0]+np.arange(0., 1.01, 0.01)*(xmax[0]-xmin[0])
        y = xmin[1]+np.arange(0., 1.01, 0.01)*(xmax[1]-xmin[1])
        X, Y = np.meshgrid(x, y)
        F = X.copy()
        
        for iobj in range(nf+ng):
            for i in range(len(X[:,0])):
                for j in range(len(X[0,:])):
                    F[i,j] = func(np.array([X[i,j],Y[i,j]]))[iobj]
            if iobj < nf:
                Fs.append(F.copy())
            else:
                Gs.append(F.copy())
            plt.figure(str(iobj+1)+'-th function')
            plt.pcolor(X,Y,F,cmap='jet',shading="auto")
            plt.colorbar()
            plt.contour(X,Y,F,40,colors='black')
            plt.show()
            # fig = plt.figure(str(iobj+1)+'-th function-3D ')
            # ax = Axes3D(fig)
            # ax.plot_surface(X, Y, F, rstride=1, cstride=1, cmap=cm.jet)
        Fs = np.array(Fs)
        Gs = np.array(Gs)
        feasible = np.all(Gs<=0, axis=0)
        for iobj in range(nf):
            F = np.where(feasible, Fs[iobj,:,:], np.nan)
            plt.figure(str(iobj+1)+'-th function in feasible area')
            plt.pcolor(X,Y,F,cmap='jet',shading="auto")
            plt.colorbar()
            plt.contour(X,Y,F,40,colors='black')
            plt.show()
