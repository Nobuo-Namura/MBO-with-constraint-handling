# -*- coding: utf-8 -*-
"""
gp.py
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
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.special import erf
from decimal import Decimal, getcontext
from scipy.spatial import distance

from singlega import SingleGA
import test_problem
from initial_sample import generate_initial_sample

#======================================================================
class GaussianProcess:
#======================================================================
    def __init__(self, df_sample, df_design_space):
        self.ns = len(df_sample)
        self.nx = sum(df_sample.columns.str.count('x'))
        self.nf = sum(df_sample.columns.str.count('f'))
        self.ng = sum(df_sample.columns.str.count('g'))
        self.x = np.asarray(df_sample.iloc[:,:self.nx])
        self.f = np.asarray(df_sample.iloc[:,self.nx:self.nx+self.nf])
        self.g = np.asarray(df_sample.iloc[:,self.nx+self.nf:])
        self.xmin = df_design_space['min'].values
        self.xmax = df_design_space['max'].values
        self.x0 = (self.x - self.xmin)/(self.xmax - self.xmin)
        self.LU_error = 0.1
        self.nfg = 0
        
#======================================================================
    def training(self, theta0 = 3.0, npop = 500, ngen = 500, mingen = 0, \
                 STOP=True, NOISE=[False], PRINT=True, KERNEL='Gaussian'):
        self.kern = self.kernel_function(KERNEL)
        self.NOISE = NOISE
        self.theta = np.zeros((self.nf+self.ng, self.nx+1))
        func = self._likelihood
        getcontext().prec = 28#56
        
        for i in range(self.nf + self.ng):
            print('--- '+str(i+1)+'-th function estimation -------------------')
            self.nfg = i
            if NOISE[i]:
                theta_min = np.full(self.nx+1, -4.0)
                theta_max = np.full(self.nx+1, theta0)
                theta_min[-1] = -20.0
                theta_max[-1] = 1.0
            else:
                theta_min = np.full(self.nx, -4.0)
                theta_max = np.full(self.nx, theta0)
            sga = SingleGA(func, theta_min, theta_max, npop=npop, ngen=ngen, mingen=mingen, MIN=False, \
                      STOP=STOP, PRINT=PRINT, HOTSTART=False, \
                      pcross=0.9, pmut=1.0/len(theta_min), eta_c=10.0, eta_m=20.0)
            Ln, theta = sga.optimize()
            if NOISE[i]:
                self.theta[i,:] = 10.0**theta
            else:
                self.theta[i,:self.nx] = 10.0**theta
        return self.theta

#======================================================================
    def kernel_function(self, KERNEL='Gaussian'):
        def gaussian_kernel(r):
            return np.exp(-r**2)
        def matern5_kernel(r):
            rs5 = np.sqrt(5)*np.abs(r)
            return (1.0 + rs5 + (rs5**2)/3.0)*np.exp(-rs5)
        def matern3_kernel(r):
            rs3= np.sqrt(3)*np.abs(r)
            return (1.0 + rs3)*np.exp(-rs3)
        def exponential_kernel(r):
            return np.exp(-np.abs(r))

        if KERNEL == 'Gaussian':
            kern = gaussian_kernel
        elif KERNEL == 'Matern5':
            kern = matern5_kernel
        elif KERNEL == 'Matern3':
            kern = matern3_kernel
        elif KERNEL == 'Exponential':
            kern = exponential_kernel
        else:
            kern = gaussian_kernel
        return kern

#======================================================================
    def _likelihood(self, theta0):
        theta = 10.0**theta0
        R, detR, mu, sigma, xtheta = self._corr_matrix(theta, self.nfg)
        if detR > 0.0 and sigma > 0.0:
            Ln = -0.5*float(Decimal(self.ns*np.log(sigma)) + detR.ln())
        else:
            Ln = -1.e+20
        return Ln

#======================================================================
    def _corr_matrix(self, theta, nfg):
        xtheta = np.sqrt(theta[:self.nx])*self.x0
#        R = np.exp(-distance.cdist(xtheta, xtheta)**2.0)
        R = self.kern(distance.cdist(xtheta, xtheta))
        if self.nx < len(theta):
            R += np.diag(np.full(len(R),theta[-1]))
        ones = np.ones(self.ns)
        Ri = linalg.lu_factor(R) #cholesky and cho_factor are not suitable; ldl is possible but slow
        detR = np.prod([Decimal(Ri[0][i,i]) for i in range(len(Ri[0]))])
        error = np.max(np.abs(linalg.lu_solve(Ri,R) - np.identity(len(R))))
        if detR > 0.0 and error < self.LU_error:
            if nfg < self.nf:
                mu = np.dot(ones, linalg.lu_solve(Ri,self.f[:,nfg]))/np.dot(ones, linalg.lu_solve(Ri,ones))
                fmu = self.f[:,nfg] - mu
                sigma =  np.dot(fmu, linalg.lu_solve(Ri,fmu))/self.ns
            else:
                mu = np.dot(ones, linalg.lu_solve(Ri,self.g[:,nfg-self.nf]))/np.dot(ones, linalg.lu_solve(Ri,ones))
                fmu = self.g[:,nfg-self.nf] - mu
                sigma =  np.dot(fmu, linalg.lu_solve(Ri,fmu))/self.ns
        else:
            detR = 0.0
            mu = 0.0
            sigma = 1.0
        return R, detR, mu, sigma, xtheta

#======================================================================
    def construction(self, theta, KERNEL='Gaussian'):
        try:
            self.kern
        except:
            self.kern = self.kernel_function(KERNEL)
        self.theta = theta
        self.mu = np.zeros(self.nf + self.ng)
        self.sigma = np.zeros(self.nf + self.ng)
        self.Ri = []
        self.Rifm = np.zeros([self.ns, self.nf + self.ng])
        self.Ri1 = np.zeros([self.ns, self.nf + self.ng])
        self.xtheta = np.zeros([self.ns, self.nx, self.nf + self.ng])
        
        for i in range(self.nf + self.ng):
            R, detR, self.mu[i], self.sigma[i], self.xtheta[:,:,i] = self._corr_matrix(self.theta[i,:], i)
            Ri = linalg.lu_factor(R)
            self.Ri.append(Ri)
            self.Ri1[:,i] = linalg.lu_solve(Ri, np.ones(self.ns))
            if i < self.nf:
                self.Rifm[:,i] = linalg.lu_solve(Ri, self.f[:,i]-self.mu[i])
            else:
                self.Rifm[:,i] = linalg.lu_solve(Ri, self.g[:,i-self.nf]-self.mu[i])
        return

#======================================================================
    def estimation(self, xs, nfg=-1):
        if nfg >= 0:
            self.nfg = nfg
        xs0 = (xs - self.xmin)/(self.xmax - self.xmin)
        xstheta = np.sqrt(self.theta[self.nfg,:self.nx])*xs0
#        r = np.exp(-distance.cdist(xstheta.reshape([1,len(xstheta)]), self.xtheta[:,:,self.nfg])**2.0).reshape(self.ns)
        r = self.kern(distance.cdist(xstheta.reshape([1,len(xstheta)]), self.xtheta[:,:,self.nfg])).reshape(self.ns)
        f = self.mu[self.nfg] + np.dot(r, self.Rifm[:,self.nfg])
        Rir = linalg.lu_solve(self.Ri[self.nfg], r)
        ones = np.ones(len(self.Rifm[:,0]))
        s = self.sigma[self.nfg]*(1.0 - np.dot(r,Rir) + ((1.0-np.dot(ones,Rir))**2.0)/np.dot(ones,self.Ri1[:,self.nfg]))
        s = np.sqrt(np.max([s, 0.0]))
        return f, s

#======================================================================
    def estimate_f(self, xs, nfg=0):
        return self.estimation(xs, nfg=nfg)[0]

#======================================================================
    def estimate_s(self, xs, nfg=0):
        return self.estimation(xs, nfg=nfg)[1]

#======================================================================
    def estimate_multiobjective_fg(self, xs):
        f = np.array([self.estimate_f(xs, nfg=i) for i in range(self.nf + self.ng)])
        return f

#======================================================================
    def probability_of_improvement(self, fref, f, s, MIN=True):
        if s > 0.0:
            y = np.where(MIN, 1.0, -1.0)*(fref - f)/s
            pi = 0.5*(1.0 + erf(y/np.sqrt(2.0))) 
        elif (MIN and f < fref) or (not MIN and f > fref):
            pi = 1.0
        else:
            pi = 0.0
        return pi

#======================================================================
    def expected_improvement(self, fref, f, s, MIN=True):
        if s > 0.0:
            y = np.where(MIN, 1.0, -1.0)*(fref - f)/s
            cdf = 0.5*(1.0 + erf(y/np.sqrt(2.0)))
            pdf = 1.0/np.sqrt(2.0*np.pi)*np.exp(-0.5*y**2.0)
            ei = s*(y*cdf + pdf)
        elif (MIN and f < fref) or (not MIN and f > fref):
            ei = np.abs(f - fref)
        else:
            ei = 0.0
        return ei

#======================================================================
    def add_sample(self, x_add, f_add, g_add):
        self.ns += len(x_add)
        if x_add.ndim == 1:
            x_add = np.reshape(x_add, [1,len(x_add)])
        if f_add.ndim == 1:
            f_add = np.reshape(f_add, [1,len(f_add)])
        if g_add.ndim == 1:
            g_add = np.reshape(g_add, [1,len(g_add)])
        self.x = np.vstack([self.x, x_add])
        x0_add = (x_add - self.xmin)/(self.xmax - self.xmin)
        self.x0 = np.vstack([self.x0, x0_add])
        self.f = np.vstack([self.f, f_add])
        self.g = np.vstack([self.g, g_add])
        return

#======================================================================
    def delete_sample(self, n_del):
        self.ns -= n_del
        self.x = self.x[:-n_del, :]
        self.x0 = self.x0[:-n_del, :]
        self.f = self.f[:-n_del, :]
        self.g = self.g[:-n_del, :]
        return

#======================================================================
if __name__ == "__main__":
        
    """=== Edit from here ==========================================="""
    func_name = 'SGM'            # Test problem name in test_problem.py
    seed = 3                     # Random seed for SGM function
    nx = 2                       # Number of design variables
    nf = 2                       # Number of objective functions
    ng = 0                       # Number of constraint functions where g <= 0 is satisfied for feasible solutions
    k = 1                        # Position paramete k in WFG problems
    ns = 30                      # Number of initial sample points when GENE=True
    MIN = np.full(nf,True)       # Minimization: True, Maximization: False
    NOISE = np.full(nf+ng,False) # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    KERNEL = 'Gaussian'          # Kernel function: Gaussian, Matern5, Matern3, Exponential
    xmin = np.full(nx, 0.0)      # Lower bound of design sapce
    xmax = np.full(nx, 1.0)      # Upper bound of design sapce
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    """=== Edit End ================================================="""


    func = test_problem.define_problem(func_name, nf, ng, k, seed)
    df_samples, df_design_space = generate_initial_sample(func_name, nx, nf, ng, ns, 1, xmin, xmax, current_dir, fname_design_space, fname_sample, k=k, seed=seed, FILE=False)
    df_sample = df_samples[0]
    
    gp = GaussianProcess(df_sample, df_design_space)
    theta = gp.training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE, KERNEL=KERNEL)
    gp.construction(theta)
    
    if nx == 2:
        x = gp.xmin[0]+np.arange(0., 1.01, 0.01)*(gp.xmax[0]-gp.xmin[0])
        y = gp.xmin[1]+np.arange(0., 1.01, 0.01)*(gp.xmax[1]-gp.xmin[1])
        X, Y = np.meshgrid(x, y)
        F = X.copy()
        S = X.copy()
        EI = X.copy()
        for k in range(nf+ng):
            if k < nf:
                if MIN[k]:
                    fref = gp.f[:,k].min()
                else:
                    fref = gp.f[:,k].max()
            for i in range(len(X[:,0])):
                for j in range(len(X[0,:])):
                    F[i,j], S[i,j] = gp.estimation(np.array([X[i,j],Y[i,j]]), nfg=k)
                    if k < nf:
                        EI[i,j] = gp.expected_improvement(fref, F[i,j], S[i,j], MIN=MIN[k])
            plt.figure('objective function '+str(k+1))
            plt.plot(gp.x[:,0],gp.x[:,1],'o',c='black')
            plt.pcolor(X,Y,F,cmap='jet',shading='auto')
            plt.colorbar()
            plt.contour(X,Y,F,40,colors='black',linestyles='solid')
            
            plt.figure('estimation error '+str(k+1))
            plt.plot(gp.x[:,0],gp.x[:,1],'o',c='black')
            plt.pcolor(X,Y,S,cmap='jet',shading='auto')
            plt.colorbar()
            plt.contour(X,Y,S,40,colors='black',linestyles='solid')
            
            if k < nf:
                plt.figure('expected improvement '+str(k+1))
                plt.plot(gp.x[:,0],gp.x[:,1],'o',c='black')
                plt.pcolor(X,Y,EI,cmap='jet',shading='auto')
                plt.colorbar()
    
    n_valid = 10000
    fs = np.zeros([n_valid,2])
    R2 = np.zeros(nf+ng)
    x_valid = gp.xmin + np.random.rand(n_valid, nx)*(gp.xmax - gp.xmin)
    for i in range(nf+ng):
        for j in range(n_valid):
            fs[j,0], ss = gp.estimation(x_valid[j,:], nfg=i)
            if nf+ng > 1:
                fs[j,1] = func(x_valid[j,:])[i]
            else:
                fs[j,1] = func(x_valid[j,:])
        delt = fs[:,0]-fs[:,1]
        R2[i] = 1-(np.dot(delt,delt)/float(n_valid))/np.var(fs[:,1])
        plt.figure('cross validation for objective function '+str(i+1))
        plt.plot(fs[:,1], fs[:,0], '.')
        linear = [fs.min(), fs.max()]
        plt.plot(linear, linear, c='black')
    print(R2)

