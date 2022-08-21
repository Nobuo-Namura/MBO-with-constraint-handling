# -*- coding: utf-8 -*-
"""
mbo.py
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
import sys
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm
from scipy.spatial import distance
import functools
from sklearn.cluster import KMeans
from pyDOE2 import lhs
from mpl_toolkits.mplot3d import Axes3D

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from gp import GaussianProcess
from moead_epbii import MOEAD_EPBII as MOEAD
import test_problem
from initial_sample import generate_initial_sample

#======================================================================
class BayesianOptimization(GaussianProcess):
#======================================================================
    def __init__(self, df_sample, df_design_space, MIN=[True, True], dist_threshold=1.0e-8):
        super().__init__(df_sample, df_design_space)
        self.MIN = np.array(MIN)
        self.delta_mi = 1.0e-6
        self.sqrt_alpha = np.sqrt(np.log(2.0/self.delta_mi))
        self.gamma_mi = np.zeros(self.nf)
        self.nrand = 100
        self.epsilon = 0.01
        self.tiny = 1.0e-20
        self.dist_threshold = dist_threshold
        self.multiplier = 10
        self.uni_rand = lhs(self.nf, samples=self.nrand, criterion='cm',iterations=100)

# functions for MOP
#======================================================================
    def _estimate_fg_as_min(self, xs, nfg):
        nfgs = np.hstack([nfg, np.arange(self.nf, self.nf + self.ng)])
        f = np.array([self.estimation(xs, nfg=i)[0] for i in nfgs])
        f[0] *= np.where(self.MIN[nfg], 1.0, -1.0)
        return f

#======================================================================
    def _estimate_multiobjective_fg_as_min(self, xs):
        f = self.estimate_multiobjective_fg(xs)
        f[:self.nf] *= np.where(self.MIN, 1.0, -1.0)
        return f

#======================================================================
    def _infill_preprocess(self, PLOT=False):
        self.nref, refvec_on_hp = self.generate_refvec(self.nf, self.n_randvec, self.nh, self.nhin, self.multiplier)
        self.ref_theta, self.normalized_refvec_distance, self.dmin = self._evaluate_ref_theta(refvec_on_hp)
        km = KMeans(n_clusters=self.n_add, init='k-means++', n_init=100, max_iter=10000)
        self.refvec_cluster = km.fit_predict(refvec_on_hp) #clustering on hyperplane
        self.refvec = refvec_on_hp/np.reshape(np.linalg.norm(refvec_on_hp,axis=1),[-1,1]) #normalize
        if self.CRITERIA == 'EIPBII':
            self.refvec = -self.refvec

        if PLOT:
            if self.nf==2:
                plt.figure('refvec-2D')
                for i in range(self.n_add):
                    plt.scatter(refvec_on_hp[self.refvec_cluster==i,0],refvec_on_hp[self.refvec_cluster==i,1])
                title = 'refvec_at_'+str(self.ns)+'_samples.png'
                plt.savefig(title, dpi=300)
                plt.close()
            elif self.nf==3: 
                fig = plt.figure('refvec-3D')
                ax = Axes3D(fig)
                for i in range(self.n_add):
                    ax.scatter3D(refvec_on_hp[self.refvec_cluster==i,0],refvec_on_hp[self.refvec_cluster==i,1],refvec_on_hp[self.refvec_cluster==i,2])
                title = 'refvec_at_'+str(self.ns)+'_samples.png'
                plt.savefig(title, dpi=300)
                plt.close()
        return

#======================================================================
    def generate_refvec(self, nf, n_randvec, nh, nhin, multiplier=10):
        def _simplex_lattice_design(refvec_on_hp, vector, ih, nh, nf, iref, i):
            if i == nf-1:
                vector[i] = float(ih)/float(nh)
                refvec_on_hp[iref,:] = vector.copy()
                iref += 1
            else:
                for j in range(ih+1):
                    vector[i] = float(j)/float(nh)
                    refvec_on_hp, iref = _simplex_lattice_design(refvec_on_hp, vector, ih-j, nh, nf, iref, i+1)
            return refvec_on_hp, iref
        
        def _generate_uniform_vector(nf, nh, nhin):
            nref = 0
            if nh > 0:
                nref += int(comb(nh+nf-1, nf-1))
            if nhin > 0:
                nrefin = int(comb(nhin+nf-1, nf-1))
                nref += nrefin
            univec = np.zeros([nref, nf])
            vector = np.zeros(nf)
            iref= 0
            if nh > 0:
                univec, iref = _simplex_lattice_design(univec, vector, nh, nh, nf, iref, 0)
            if nhin > 0:
                inref = iref
                univec, iref = _simplex_lattice_design(univec, vector, nhin, nhin, nf, iref, 0)
                tau = 0.5
                univec[inref:,:] = (1.0-tau)/float(nf) + tau*univec[inref:,:]
            return univec
        
        def _generate_random_vector(nf, n_randvec, multiplier=10):
            nvec = np.ones(nf)
            randvec = np.random.rand(multiplier*n_randvec, nf)
            if n_randvec > 0:
                randvec = randvec/np.reshape(np.dot(nvec, randvec.T),[len(randvec),1]) #projection to hyperplane
                flag = np.full(len(randvec), False)
                flag[np.random.randint(0, len(flag))] = True
                for i in range(n_randvec-1):
                    past = randvec[flag, :]
                    dist = distance.cdist(randvec, past)
                    i_add = np.argmax(np.min(dist,axis=1))
                    flag[i_add] = True
                randvec = randvec[flag]
            return randvec

        univec = _generate_uniform_vector(nf, nh, nhin)
        randvec = _generate_random_vector(nf, n_randvec, multiplier)
        refvec_on_hp = np.vstack([univec, randvec]) # reference vector on hyperplane
        nref = len(refvec_on_hp)
        return nref, refvec_on_hp

#======================================================================
    def _evaluate_ref_theta(self, refvec):
        normalized_refvec_distance = distance.cdist(refvec, refvec)
        dmax = np.max(normalized_refvec_distance)
        normalized_refvec_distance = np.where(normalized_refvec_distance>0, normalized_refvec_distance, dmax)
        dmin = np.mean(np.min(normalized_refvec_distance, axis=0))
        temp = normalized_refvec_distance/dmin
        normalized_refvec_distance = np.where(temp>1.0, 2.0*temp-1.0, temp**2.0) + 1.0
        ref_theta = np.sqrt(2.0)/dmin
        return ref_theta, normalized_refvec_distance, dmin

#======================================================================
    def optimize_multiobjective_problem(self, CRITERIA='EPBII', OPTIMIZER='NSGA3', SRVA=True, \
                                         n_add = 5, n_randvec=40, nh=0, nhin=0, \
                                         n_randvec_ea=0, nh_ea=20, nhin_ea=0, npop_ea=100, ngen_ea=200, \
                                         pbi_theta=1.0, PLOT=False, PRINT=True):
        self.CRITERIA = CRITERIA
        self.n_add = n_add
        self.pbi_theta =  pbi_theta
        self.n_randvec, self.nh, self.nhin = n_randvec, nh, nhin
        
        if self.CRITERIA == 'EPBII' or self.CRITERIA == 'EIPBII':
            if SRVA:
                self.nref = self.n_randvec
            else:
                self._infill_preprocess(PLOT=False)
            self._utopia_nadir_on_gp(OPTIMIZER, SRVA, n_randvec_ea, nh_ea, nhin_ea, npop_ea, ngen_ea, PLOT, PRINT)
            self._reference_pbi()
            
            # MOEA/D to maximize EPBII for all reference vectors
            f_opt0 = (self.f_opt - self.utopia)/(self.nadir - self.utopia)
            if self.CRITERIA == 'EIPBII':
                f_opt0 = -1.0 + f_opt0
            x_init = np.zeros([self.nref, self.nx])
            for iref in range(self.nref):
                i_opt = np.argmax(np.abs(np.dot(f_opt0/np.reshape(np.linalg.norm(f_opt0,axis=1),[-1,1]), self.refvec[iref,:])))
                x_init[iref,:] = self.x_opt[i_opt,:]
            if self.CRITERIA == 'EPBII':
                self.acquisition_function = self._evaluate_epbii
            else:
                self.acquisition_function = self._evaluate_eipbii
            print(self.CRITERIA + ' maximization with MOEA/D')
            moead = MOEAD(self.refvec, self.acquisition_function, self.xmin, self.xmax, ngen=50, PRINT=PRINT, HOTSTART=True, \
                          nghbr=20, factor=1.0, pcross=1.0, pmut=1.0/self.nx, eta_c=10.0, eta_m=20.0)
            self.epbii, self.x_candidate = moead.optimize(x_init=x_init)
            self.f_candidate = np.zeros([self.nref,self.nf])
            self.g_candidate = np.zeros([self.nref,self.ng])
            for iref in range(self.nref):
                for iobj in range(self.nf):
                    self.f_candidate[iref,iobj], s = self.estimation(self.x_candidate[iref,:], nfg=iobj)
                for icon in range(self.ng):
                    self.g_candidate[iref,icon], s = self.estimation(self.x_candidate[iref,:], nfg=self.nf+icon)
            
            # compute fitness
            self.rank = self.pareto_ranking(self.f_candidate) # constraints have already been considered in epbii maximization
            self.fitness = self.epbii/(self.nich_count*self.rank)
            dist = np.min(distance.cdist(self.x, self.x_candidate), axis=0)
            self.fitness = np.where(dist>self.dist_threshold, self.fitness, -1.0e20)
            x_add = np.zeros([self.n_add,self.nx])
            f_add = np.zeros([self.n_add,self.nf])
            g_add = np.zeros([self.n_add,self.ng])
            self.fitness_org = self.fitness.copy()
            for i_add in range(self.n_add):
                i_candidate = np.where((self.refvec_cluster==i_add) & \
                                       (self.fitness==np.max(self.fitness[self.refvec_cluster==i_add])))[0][0]
                x_add[i_add,:] = self.x_candidate[i_candidate,:]
                f_add[i_add,:] = self.f_candidate[i_candidate,:]
                g_add[i_add,:] = self.g_candidate[i_candidate,:]
                self.nich[i_candidate] += 1
                self.nich_count = np.array([np.sum(self.nich/self.normalized_refvec_distance[i,:]) for i in range(self.nref)])
                self.fitness = np.where(dist>self.dist_threshold, self.epbii/(self.nich_count*self.rank), -1.0e20)
            # remove duplicated samples
            dist = distance.cdist(x_add, x_add)
            for i_add in range(self.n_add):
                if (self.n_add > 1) and (np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) <= self.dist_threshold):
                    print('a sample point candidate was changed')
                    flag = True
                    # replaced from the same cluster
                    for j in range(1, len(self.fitness[self.refvec_cluster==i_add])):
                        i_candidate = np.argsort(self.fitness[self.refvec_cluster==i_add])[::-1][j]
                        x_add[i_add,:] = self.x_candidate[self.refvec_cluster==i_add][i_candidate,:]
                        f_add[i_add,:] = self.f_candidate[self.refvec_cluster==i_add][i_candidate,:]
                        g_add[i_add,:] = self.g_candidate[self.refvec_cluster==i_add][i_candidate,:]
                        dist = distance.cdist(x_add, x_add)
                        if np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) > self.dist_threshold:
                            flag = False
                            break
                    # replaced from other clusters
                    if flag:
                        for j in range(len(self.fitness)):
                            i_candidate = np.argsort(self.fitness)[::-1][j]
                            x_add[i_add,:] = self.x_candidate[i_candidate,:]
                            f_add[i_add,:] = self.f_candidate[i_candidate,:]
                            g_add[i_add,:] = self.g_candidate[i_candidate,:]
                            dist = distance.cdist(x_add, x_add)
                            if np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) > self.dist_threshold:
                                flag = False
                                break
            return x_add, f_add, g_add

#======================================================================
    def _utopia_nadir_on_gp(self, OPTIMIZER='NSGA3', SRVA=True, n_randvec_ea=0, nh_ea=20, nhin_ea=0, npop_ea=100, ngen_ea=200, PLOT=False, PRINT=True):
        self.utopia = np.zeros(self.nf)
        self.nadir = np.ones(self.nf)
        self.f_opt, self.g_opt, self.x_opt = self._multiobjective_optimization_on_gp(OPTIMIZER, n_randvec_ea, nh_ea, nhin_ea, npop_ea, ngen_ea, PLOT=False, PRINT=PRINT)
        f_sga, g_sga, x_sga = self._single_objective_optimization_on_gp()
        self.f_opt = np.vstack([self.f_opt, f_sga])
        self.g_opt = np.vstack([self.g_opt, g_sga])
        self.x_opt = np.vstack([self.x_opt, x_sga])
        rank = self.pareto_ranking(self.f_opt, self.g_opt)
        if len(rank[rank==1.0]) >= self.nf:
            self.f_opt = self.f_opt[rank==1.0]
            self.x_opt = self.x_opt[rank==1.0]
            self.f_ref, f_epsilon = self._select_from_estimation(self.f_opt, self.x_opt)
            if SRVA:
                self.refvec, self.refvec_cluster, self.normalized_refvec_distance, self.ref_theta, self.dmin \
                = self._vector_adaptation(self.f_opt, self.f_ref, f_epsilon, PLOT)
        elif len(self.f) >= self.nf:
            self.f_ref = self._select_from_samples()
            f_epsilon = np.zeros(self.nf)
        else:
            print('normalization failed')
            sys.exit()
        f_min = np.min(self.f_ref, axis=0) - f_epsilon
        f_max = np.max(self.f_ref, axis=0) + f_epsilon
        minmax = 1.0 - 2.0*self.MIN.astype(np.float)
        self.utopia = np.where(minmax<0.0, f_min, f_max)
        self.nadir = np.where(minmax<0.0, f_max, f_min)
        if PRINT:
            print('Utopia: ', self.utopia)
            print('Nadir: ', self.nadir)
        return

#======================================================================
    def _multiobjective_optimization_on_gp(self, OPTIMIZER='NSGA3', n_randvec_ea=0, nh_ea=20, nhin_ea=0, npop_ea=100, ngen_ea=200, PLOT=False, PRINT=True):
        class MOProblem(ElementwiseProblem):
            def __init__(self, func, nx, nf, ng, xmin, xmax):
                self.func = func
                self.nf = nf
                super().__init__(n_var=nx, n_obj=nf, n_constr=ng, xl=xmin, xu=xmax)
            def _evaluate(self, x, out, *args, **kwargs):
                fg = self.func(x)
                out["F"] = fg[:self.nf]
                out["G"] = fg[self.nf:]

        print('Multi-objective optimization on the Kriging models')
        problem = MOProblem(self._estimate_multiobjective_fg_as_min, self.nx, self.nf, self.ng, self.xmin, self.xmax)
        if OPTIMIZER=='NSGA3':
            nref, refvec_on_hp = self.generate_refvec(self.nf, n_randvec_ea, nh_ea, nhin_ea, self.multiplier)
            algorithm = NSGA3(pop_size=nref, ref_dirs=refvec_on_hp)
        else:
            algorithm = NSGA2(pop_size=self.npop_ea, n_offsprings=self.npop_ea, 
                              sampling=get_sampling("real_random"), 
                              crossover=get_crossover("real_sbx", prob=0.9, eta=10), 
                              mutation=get_mutation("real_pm", eta=20), 
                              eliminate_duplicates=True)
        res = minimize(problem, algorithm, return_least_infeasible=True, seed=1, termination=('n_gen', ngen_ea), ave_history=False, verbose=False)
        return np.where(self.MIN, 1.0, -1.0)*res.F, res.G, res.X

#======================================================================
    def _single_objective_optimization_on_gp(self):
        class SOProblem(ElementwiseProblem):
            def __init__(self, func, nx, ng, xmin, xmax):
                self.func = func
                super().__init__(n_var=nx, n_obj=1, n_constr=ng, xl=xmin, xu=xmax)
            def _evaluate(self, x, out, *args, **kwargs):
                fg = self.func(x)
                out["F"] = fg[:1]
                out["G"] = fg[1:]
            
        x_opt = np.zeros([self.nf, self.nx])
        f_opt = np.zeros([self.nf, self.nf])
        g_opt = np.zeros([self.nf, self.ng])
        for iobj in range(self.nf):
            print('Single objective optimization for the '+str(iobj+1)+'-th objective function')
            func = functools.partial(self._estimate_fg_as_min, nfg=iobj)
            problem = SOProblem(func, self.nx, self.ng, self.xmin, self.xmax)
#            algorithm = CMAES(x0=np.random.random(problem.n_var))
#            res = minimize(problem, algorithm, return_least_infeasible=True, seed=1, termination=('n_evals', 10000), ave_history=False, verbose=False)
            algorithm = GA(pop_size=100, eliminate_duplicates=True)
            res = minimize(problem, algorithm, return_least_infeasible=True, seed=1, termination=('n_gen', 100), ave_history=False, verbose=False)
            x_opt[iobj,:] = res.X
        for iobj in range(self.nf):
            fg_opt = self.estimate_multiobjective_fg(x_opt[iobj,:])
            f_opt[iobj,:] = fg_opt[:self.nf]
            g_opt[iobj,:] = fg_opt[self.nf:]
        return f_opt, g_opt, x_opt

#======================================================================
    def _select_from_estimation(self, f_opt, x_opt):
        #remove weak Pareto optimal solutions
        flag, f_epsilon = self._remove_weak_pareto_with_epsilon_dominance(f_opt, self.epsilon)
        #reference solution selection from estimated optimal solutions
        if len(f_opt[flag]) >= self.nf:
            f_ref = f_opt[flag]
        else:
            f_ref = f_opt
        return f_ref, f_epsilon

#======================================================================
    def _remove_weak_pareto_with_epsilon_dominance(self, f_opt, epsilon):
        f_min = np.min(f_opt, axis=0)
        f_max = np.max(f_opt, axis=0)
        f_opt0 = (f_opt - f_min)/(f_max - f_min)
        flag = np.full(len(f_opt0), True)
        n = len(f_opt0)
        f_epsilon = epsilon*(f_max - f_min)
        for i in range(n):
            for j in range(n):
                if flag[j]:
                    irank = 0
                    for iobj in range(self.nf):
                        if self.MIN[iobj] and f_opt0[i,iobj]>=f_opt0[j,iobj]-epsilon:
                            irank += 1
                        elif (not self.MIN[iobj]) and f_opt0[i,iobj]<=f_opt0[j,iobj]+epsilon:
                            irank += 1
                    if i!=j and irank == self.nf:
                        flag[i] = False
        return flag, f_epsilon

#======================================================================
    def _vector_adaptation(self, f_opt, f_ref, f_epsilon, PLOT=False):
        refvec_on_pf = self._select_refvec(f_opt, f_ref, f_epsilon) #refvec on non-dominated front
        refvec_cluster = self._classify_refvec(refvec_on_pf, PLOT=PLOT) #clustering on non-dominated front
        refvec = refvec_on_pf/np.reshape(np.linalg.norm(refvec_on_pf,axis=1),[-1,1]) #normalized
        nvec = np.ones(self.nf)
        if self.CRITERIA == 'EIPBII':
            nvec = -nvec
        refvec_on_hp = refvec/np.reshape(np.dot(nvec, refvec.T),[len(refvec),1]) #projection to hyperplane
        ref_theta, normalized_refvec_distance, dmin = self._evaluate_ref_theta(refvec_on_hp)
        return refvec, refvec_cluster, normalized_refvec_distance, ref_theta, dmin

#======================================================================
    def _select_refvec(self, f_opt, f_ref, f_epsilon):
        f_min = np.min(f_ref, axis=0) - f_epsilon
        f_max = np.max(f_ref, axis=0) + f_epsilon
        utopia = np.where(self.MIN, f_min, f_max)
        nadir = np.where(self.MIN, f_max, f_min)
        f_ref0 = (f_opt - utopia)/(nadir - utopia)
        f0 = (self.f - utopia)/(nadir - utopia)
        nvec = np.ones(self.nf)
        sign = 1.0
        if self.CRITERIA == 'EIPBII':
            f_ref0 = -1.0 + f_ref0
            sign = -1.0
        if self.nref > len(f_opt):
            nref, initvec = self.generate_refvec(self.nf, self.n_randvec, self.nh, self.nhin, self.multiplier)
            f_ref0 = np.vstack([sign*initvec, f_ref0])
            f_ref0 = f_ref0/np.reshape(np.dot(nvec, f_ref0.T),[len(f_ref0),1]) #projection to hyperplane
            f0 = f0/np.reshape(np.dot(nvec, f0.T),[len(f0),1]) #projection to hyperplane
        flag = np.full(len(f_ref0), False)
        for i in range(self.nref):
            past = np.vstack([f0, f_ref0[flag, :].reshape([np.sum(flag),self.nf])])
            dist = distance.cdist(f_ref0, past)
            i_add = np.argmax(np.min(dist,axis=1))
            flag[i_add] = True
        refvec_on_pf = f_ref0[flag]
        return refvec_on_pf

#======================================================================
    def _classify_refvec(self, refvec, PLOT=False): 
        #Learning: clustering reference vectors inside of hyperbox generated by ideal and nadir points
        if self.CRITERIA == 'EPBII':
            refvec_inside = np.array([refvec[i,:] for i in range(len(refvec[:,0])) if np.all(refvec[i,:]>=0.0) and np.all(refvec[i,:]<=1.0)])
        else:
            refvec_inside = np.array([refvec[i,:] for i in range(len(refvec[:,0])) if np.all(refvec[i,:]>=-1.0) and np.all(refvec[i,:]<=0.0)])
        km = KMeans(n_clusters=self.n_add, init='k-means++', n_init=100, max_iter=10000)
        if self.n_add < len(refvec_inside):
            kmfit = km.fit(refvec_inside)
        else:
            kmfit = km.fit(refvec)
        #Prediction: clustering all reference vectors
        refvec_cluster = kmfit.predict(refvec) #clustering on non-dominated front
        #Visualization
        if PLOT:
            if self.nf==2:
                plt.figure('refvec-2D')
                for i in range(self.n_add):
                    plt.scatter(refvec[refvec_cluster==i,0],refvec[refvec_cluster==i,1])
                title = 'refvec_at_'+str(self.ns)+'_samples.png'
                plt.savefig(title, dpi=300)
                plt.close()
            elif self.nf==3: 
                fig = plt.figure('refvec-3D')
                ax = Axes3D(fig)
                for i in range(self.n_add):
                    ax.scatter3D(refvec[refvec_cluster==i,0],refvec[refvec_cluster==i,1],refvec[refvec_cluster==i,2])
                title = 'refvec_at_'+str(self.ns)+'_samples.png'
                plt.savefig(title, dpi=300)
                plt.close()
        return refvec_cluster

#======================================================================
    def _select_from_samples(self):
        rank = self.pareto_ranking(self.f, self.g)
        f_opt = self.f[rank==1.0]
        if len(f_opt) >= self.nf:
            f_ref = f_opt
        else:
            rank = self.pareto_ranking(self.f)
            f_opt = self.f[rank==1.0]
            if len(f_opt) >= self.nf:
                f_ref = f_opt
            else:
                f_ref = self.f
        self._infill_preprocess(PLOT=False)
        return f_ref

#======================================================================
    def _reference_pbi(self):
        rank = self.pareto_ranking(self.f, self.g)
        f0 = (self.f - self.utopia)/(self.nadir - self.utopia)
        sign = 1.0
        if self.CRITERIA == 'EIPBII':
            f0 = f0 - 1.0
            sign = -1.0
        
        pbi_sample = np.zeros([self.ns, self.nref])
        self.near_vector = np.zeros(self.ns, dtype=int)
        z = np.zeros(self.nf)
        for i in range(self.ns):
            distmin = 1.0e+20
            for j in range(self.nref):
                pbi_sample[i,j], d1, d2 = self._evaluate_pbi(z, f0[i,:], self.refvec[j,:], self.pbi_theta, sign=sign)
                if d2 < distmin:
                    distmin = d2
                    self.near_vector[i] = j
        
        self.nich = np.zeros(self.nref, dtype=int)
        if self.CRITERIA == 'EPBII':
            self.pbiref = 1.1e+20*np.ones(self.nref)
        else:
            self.pbiref = -1.1e+20*np.ones(self.nref)
        for i in range(self.ns):
            if self.ng==0 or np.all(self.g[i,:] <= 0):
                k = self.near_vector[i]
                #assign each solution to the vector whose territory includes the solution
                flag = True
                for j in range(self.nref):
                    terr, d1, d2 = self._evaluate_pbi(z, f0[i,:], self.refvec[j,:], self.ref_theta, sign=-1.0)
                    if terr >= 0.0:
                        if rank[i] == 1:
                            self.nich[j] += 1
                        if self.CRITERIA == 'EPBII' and pbi_sample[i,j] < self.pbiref[j]:
                            self.pbiref[j] = pbi_sample[i,j]
                        elif self.CRITERIA == 'EIPBII' and pbi_sample[i,j] > self.pbiref[j]:
                            self.pbiref[j] = pbi_sample[i,j]
                        if j == k:
                            flag = False
                if flag:
                    if rank[i] == 1:
                        self.nich[k] += 1
                    if self.CRITERIA == 'EPBII' and pbi_sample[i,k] < self.pbiref[k]:
                        self.pbiref[k] = pbi_sample[i,k]
                    elif self.CRITERIA == 'EIPBII' and pbi_sample[i,k] > self.pbiref[k]:
                        self.pbiref[k] = pbi_sample[i,k]
        
        self.nich_count = np.array([np.sum(self.nich/self.normalized_refvec_distance[i,:]) for i in range(self.nref)])
        if self.CRITERIA == 'EPBII':
            pbimax = np.max(self.pbiref[self.pbiref<1.0e20])
            self.pbiref = np.where(self.pbiref<1.0e20, self.pbiref, 1.1*pbimax)
            refpoint = self.pbiref.reshape([self.nref,1])*self.refvec
            self.refpoint = self.utopia + refpoint*(self.nadir - self.utopia)
        else:
            pbimin = np.min(self.pbiref[self.pbiref>-1.0e20])
            self.pbiref = np.where(self.pbiref>-1.0e20, self.pbiref, pbimin-0.1*np.abs(pbimin))
            refpoint = self.pbiref.reshape([self.nref,1])*self.refvec
            self.refpoint = self.utopia + (1.0 + refpoint)*(self.nadir - self.utopia)
        return

#======================================================================
    def _evaluate_epbii(self, xs, kref):
        f = np.zeros(self.nf)
        s = np.zeros(self.nf)
        z = np.zeros(self.nf)
        for i in range(self.nf):
            f[i], s[i] = self.estimation(xs, nfg=i)
        f0 = (f - self.utopia)/(self.nadir - self.utopia)
        s0 = np.max([s/(self.nadir - self.utopia), np.full(self.nf,self.tiny)],axis=0)
        #Constraint penalty
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        #Territory
        terr, d1t, d2t = self._evaluate_pbi(z, f0, self.refvec[kref,:], self.ref_theta, sign=-1.0)
        if terr < 0.0:
            epbii = terr/np.max([p_const, self.tiny])
        #EPBII
        else:
            fp0 = norm.ppf(self.uni_rand, loc=f0, scale=s0)
            pbis, d1s, d2s = self._evaluate_pbis(z, fp0, self.refvec[kref,:], self.pbi_theta)
            pbiis = np.where(self.pbiref[kref]>pbis, self.pbiref[kref]-pbis, 0)
            epbii = np.mean(pbiis)*p_const
            #Accelerate convergence
            if epbii <= 0.0:
                pbi, d1, d2 = self._evaluate_pbi(z, f0, self.refvec[kref,:], self.pbi_theta)
                epbii = np.min([0.0, self.pbiref[kref]-pbi])/np.max([p_const, self.tiny])
        return epbii

#======================================================================
    def _evaluate_eipbii(self, xs, kref):
        f = np.zeros(self.nf)
        s = np.zeros(self.nf)
        z = np.zeros(self.nf)
        for i in range(self.nf):
            f[i], s[i] = self.estimation(xs, nfg=i)
        f0 = -1.0 + (f - self.utopia)/(self.nadir - self.utopia)
        s0 = np.max([s/(self.nadir - self.utopia), np.full(self.nf,self.tiny)],axis=0)
        #Constraint penalty
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        #Territory
        terr, d1t, d2t = self._evaluate_pbi(z, f0, self.refvec[kref,:], self.ref_theta, sign=-1.0)
        if terr < 0.0:
            iepbii = terr/np.max([p_const, self.tiny])
        #EIPBII
        else:
            fp0 = norm.ppf(self.uni_rand, loc=f0, scale=s0)
            ipbis, d1s, d2s = self._evaluate_pbis(z, fp0, self.refvec[kref,:], self.pbi_theta, sign=-1.0)
            ipbiis = np.where(ipbis>self.pbiref[kref], ipbis-self.pbiref[kref], 0)
            iepbii = np.mean(ipbiis)*p_const
            #Accelerate convergence
            if iepbii <= 0.0:
                ipbi, d1, d2 = self._evaluate_pbi(z, f0, self.refvec[kref,:], self.pbi_theta, sign=-1.0)
                iepbii = np.min([0.0, ipbi-self.pbiref[kref]])/np.max([p_const, self.tiny])
        return iepbii

#======================================================================
    def _evaluate_pbi(self, z, f, vector, theta, sign=1.0):
        d1 = np.dot(f - z, vector)
        d2 = np.linalg.norm(f - (z + d1*vector))
        pbi = d1 + sign*theta*d2
        
        return pbi, d1, d2

#======================================================================
    def _evaluate_pbis(self, z, f, vector, theta, sign=1.0):
        d1 = np.dot(f - z, vector)
        d2 = np.linalg.norm(f - (z + np.dot(d1.reshape([len(d1),1]),vector.reshape([1,len(vector)]))), axis=1)
        pbi = d1 + sign*theta*d2
        
        return pbi, d1, d2

#======================================================================
    def pareto_ranking(self, f, g=[]):
        ns = len(f[:,0])
        nf = len(f[0,:])
        rank = np.ones(ns)
        if len(g) == 0:
            g = np.full([ns,1], -1)
        for i in range(ns):
            if all(g[i,:]<=0):
                for j in range(ns):
                    if all(g[j,:]<=0):
                        irank = 0
                        for iobj in range(nf):
                            if self.MIN[iobj] and f[i,iobj]>=f[j,iobj]:
                                irank += 1
                            elif (not self.MIN[iobj]) and f[i,iobj]<=f[j,iobj]:
                                irank += 1
                        if i!=j and irank == nf:
                            rank[i] += 1
            else:
                rank[i] = -1
        rank = np.where(rank>0, rank, np.max(rank)+1)
        return rank

# functions for SOP
#======================================================================
    def _estimation_as_min(self, xs, nfg=0):
        f, s = self.estimation(xs, nfg)
        f *= np.where(self.MIN[nfg], 1.0, -1.0)
        
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        return f*p_const

#======================================================================
    def _error_as_min(self, xs, nfg=0):
        f, s = self.estimation(xs, nfg)
        
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        return -s*p_const

#======================================================================
    def _ei_as_min(self, xs, fref=0, nfg=0):
        f, s = self.estimation(xs, nfg)
        ei = self.expected_improvement(fref, f, s, MIN=self.MIN[nfg])
        
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        return -ei*p_const

#======================================================================
    def _mi_as_min(self, xs, nfg=-1):
        f, s = self.estimation(xs, nfg)
        s2 = s**2.0
        phi = self.sqrt_alpha*(np.sqrt(s2 + self.gamma_mi[nfg]) - np.sqrt(self.gamma_mi[nfg]))
        mi = f + np.where(self.MIN[nfg], -1.0, 1.0)*phi
        if not self.MIN[nfg]:
            mi *= -1
        
        p_const = 1.0
        for i in range(self.ng):
            g, sg = self.estimation(xs, nfg=self.nf+i)
            p_const *= self.probability_of_improvement(0.0, g, sg, MIN=True)
        return mi*p_const

#======================================================================
    def _optimize_sop(self, npop=100, ngen=100, nfg=0, PRINT=False):
        
        class SOProblem(ElementwiseProblem):
            def __init__(self, func, nx, xmin, xmax):
                self.func = func
                super().__init__(n_var=nx, n_obj=1, n_constr=0, xl=xmin, xu=xmax)
            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = self.func(x)
                
        if self.CRITERIA == 'Estimation':
            self.acquisition_function = functools.partial(self._estimation_as_min, nfg=nfg)
        elif self.CRITERIA == 'Error':
            self.acquisition_function = functools.partial(self._error_as_min, nfg=nfg)
        elif self.CRITERIA == 'EI':
            mask = np.all(self.g<=0, axis=1)
            if self.MIN[nfg]:
                if mask.sum()>0:
                    fref = self.f[mask, nfg].min()
                else:
                    fref = self.f[:, nfg].max()
            else:
                if mask.sum()>0:
                    fref = self.f[mask, nfg].max()
                else:
                    fref = self.f[:, nfg].min()
            self.acquisition_function = functools.partial(self._ei_as_min, fref=fref, nfg=nfg)
        elif self.CRITERIA == 'GP-MI':
            self.acquisition_function = functools.partial(self._mi_as_min, nfg=nfg)
        else:
            self.acquisition_function = functools.partial(self._error_as_min, nfg=nfg)
        
        problem = SOProblem(self.acquisition_function, self.nx, self.xmin, self.xmax)
#        algorithm = CMAES(x0=np.random.random(problem.n_var))
#        res = minimize(problem, algorithm, return_least_infeasible=True, seed=1, termination=('n_evals', n_eval), ave_history=False, verbose=PRINT)
        algorithm = GA(pop_size=npop, eliminate_duplicates=True)
        res = minimize(problem, algorithm, return_least_infeasible=True, seed=1, termination=('n_gen', ngen), ave_history=False, verbose=PRINT)
        
        x_opt = res.X
        fg_opt = self.estimate_multiobjective_fg(x_opt)
        x_opt = x_opt.reshape([1,len(x_opt)])
        fg_opt = fg_opt.reshape([1,len(fg_opt)])

        return x_opt, fg_opt[:,:self.nf], fg_opt[:,self.nf:]

#======================================================================
    def optimize_single_objective_problem(self, CRITERIA='Error', n_add=1, npop_ea=500, ngen_ea=100, nfg=0, \
                                          theta0 = 3.0, npop = 100, ngen = 100, mingen=0, STOP=True, \
                                          PRINT=False, RETRAIN=True):
        self.CRITERIA = CRITERIA
        self.n_add = n_add
        NOISE = np.where(self.theta[:,-1]>0, True, False)
        x_opt, f_opt, g_opt = self._optimize_sop(npop_ea, ngen_ea, nfg, PRINT)
        if distance.cdist(self.x, x_opt).min() < self.dist_threshold:
            print('selected x is too close to stored ones')
            x_opt, f_opt, g_opt = self._optimize_sop(npop_ea, ngen_ea, nfg, 'Error', PRINT)
        x_add = x_opt.copy()
        f, s = self.estimation(x_opt[0], nfg)
        self.gamma_mi[nfg] += s**2.0
        if self.n_add > 1:
            theta = self.theta.copy()
            theta_saved = theta.copy()
            for i_add in range(1, self.n_add):
                self.add_sample(x_opt, f_opt, g_opt)
                if RETRAIN:
                    theta = self.training(theta0, npop, ngen, mingen, STOP, NOISE, PRINT)
                self.construction(theta)
                x_opt, f_opt, g_opt = self._optimize_sop(npop_ea, ngen_ea, nfg, PRINT)
                if distance.cdist(self.x, x_opt).min() < self.dist_threshold:
                    print('selected x is too close to stored ones')
                    x_opt, f_opt, g_opt = self._optimize_sop(npop_ea, ngen_ea, nfg, 'Error', PRINT)
                x_add = np.vstack([x_add, x_opt])
                f, s = self.estimation(x_opt[0], nfg)
                self.gamma_mi[nfg] += s**2.0
            self.delete_sample(self.n_add-1)
            self.construction(theta_saved)
        fg_add = np.array([self.estimate_multiobjective_fg(x_add[i,:]) for i in range(len(x_add[:,0]))])

        return x_add, fg_add[:, :self.nf], fg_add[:, self.nf:]

#======================================================================
if __name__ == "__main__":
    
    division = pd.read_csv('reference_vector_division.csv', index_col=0)
    """=== Edit from here ==========================================="""
    func_name = 'SGM'                       # Test problem name in test_problem.py
    seed = 3                                # Random seed for SGM function
    nx = 2                                  # Number of design variables
    nf = 2                                  # Number of objective functions
    ng = 1                                  # Number of constraint functions where g <= 0 is satisfied for feasible solutions
    k = 1                                   # Position paramete k in WFG problems
    ns = 30                                 # Number of initial sample points
    n_add = 5                               # Number of additional sample points at each iteration
    ns_max = 40                             # Number of maximum function evaluation
    CRITERIA = 'EPBII'                      # EPBII or EIPBII for multi-objective problems, EI, GP-MI, Error, or Estimation for single-objective problems
    MIN = np.full(nf,True)                  # Minimization: True, Maximization: False
    NOISE = np.full(nf+ng,False)            # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    xmin = np.full(nx, 0.0)                 # Lower bound of design sapce
    xmax = np.full(nx, 1.0)                 # Upper bound of design sapce
    SRVA = True                             # True=surrogate-assisted reference vector adaptation, False=two-layered simplex latice-design
    n_randvec = division.loc[nf, 'npop']    # Number of adaptive(SRVA=True) or random(SRVA=False) reference vector (>=0)
    ngen_ea = 200                           # Number of generation
    npop_ea = division.loc[nf, 'npop_ea']   # Number of population for GA in single-objective problems
    nh_ea = division.loc[nf, 'nh_ea']       # Division number for the outer layer of the two-layered simplex latice-design for NSGA3 in multi-objective problems (>=0)
    nhin_ea = division.loc[nf, 'nhin_ea']   # Division number for the inner layer of the two-layered simplex latice-design for NSGA3 in multi-objective problems (>=0)
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    """=== Edit End ================================================="""

    func = test_problem.define_problem(func_name, nf, ng, k, seed)
    df_samples, df_design_space = generate_initial_sample(func_name, nx, nf, ng, ns, 1, xmin, xmax, current_dir, fname_design_space, fname_sample, k=k, seed=seed, FILE=False)
    df_sample = df_samples[0]
    
    gp = BayesianOptimization(df_sample, df_design_space, MIN=MIN)
    max_iter = int((ns_max + (n_add - 1) - ns)/n_add)
    for itr in range(max_iter):
        print('=== Iteration = '+str(itr)+', Number of sample = '+str(gp.ns)+' ======================')
        theta = gp.training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE)
        gp.construction(theta)
        if nf == 1:
            x_add, f_add_est, g_add_est = gp.optimize_single_objective_problem(CRITERIA=CRITERIA, n_add=n_add, npop_ea=npop_ea, ngen_ea=ngen_ea, PRINT=False, RETRAIN=True, theta0=3.0, npop=100, ngen=100, mingen=0, STOP=True)
        elif nf > 1:
            x_add, f_add_est, g_add_est = gp.optimize_multiobjective_problem(CRITERIA=CRITERIA, n_add=n_add, n_randvec=n_randvec, nh_ea=nh_ea, nhin_ea=nhin_ea, ngen_ea=ngen_ea, PLOT=False, PRINT=True)
        fg_add = np.array([func(x_add[i]) for i in range(len(x_add))])
        if itr < max_iter-1:
            gp.add_sample(x_add, fg_add[:,:gp.nf], fg_add[:,gp.nf:])
            
    print('Visualization')
    if nx == 2:
        iref = np.random.randint(0, gp.nref)
        x = gp.xmin[0]+np.linspace(0, 1, 101)*(gp.xmax[0]-gp.xmin[0])
        y = gp.xmin[1]+np.linspace(0, 1, 101)*(gp.xmax[1]-gp.xmin[1])
        X, Y = np.meshgrid(x, y)
        F = np.zeros(np.shape(X))
        S = np.zeros(np.shape(X))
        EI = np.zeros(np.shape(X))
        Fs = []
        for k in range(nf+ng):
            for i in range(len(X[:,0])):
                for j in range(len(X[0,:])):
                    F[i,j], S[i,j] = gp.estimation(np.array([X[i,j],Y[i,j]]), nfg=k)
                    if k==0:
                        if CRITERIA=='EPBII' or CRITERIA=='EIPBII':
                            EI[i,j] = gp.acquisition_function(np.array([X[i,j],Y[i,j]]), iref)
                        elif CRITERIA=='EI' or CRITERIA=='Error':
                            EI[i,j] = -1*gp.acquisition_function(np.array([X[i,j],Y[i,j]]))
                        elif CRITERIA=='GP-MI':
                            EI[i,j] = np.where(MIN, 1, -1)*gp.acquisition_function(np.array([X[i,j],Y[i,j]]))
            Fs.append(F.copy())
            plt.figure('objective/constraint function '+str(k+1))
            plt.scatter(gp.x[:ns,0], gp.x[:ns,1], c='black', zorder=3)
            plt.scatter(gp.x[ns:,0], gp.x[ns:,1], c='white', edgecolor='black', zorder=4)
            plt.scatter(x_add[:,0], x_add[:,1], marker='*', s=100, c='white', edgecolor='black', zorder=5)
            plt.pcolor(X, Y, F, cmap='jet', shading='auto', zorder=1)
            plt.colorbar()
            plt.contour(X, Y, F, 40, colors='black', linestyles='solid', linewidths=0.1, zorder=2)
            
            if k==0:
                plt.figure('acquisition function')
                plt.scatter(gp.x[:ns,0], gp.x[:ns,1], c='black', zorder=2)
                plt.scatter(gp.x[ns:,0], gp.x[ns:,1], c='white', edgecolor='black', zorder=3)
                plt.scatter(x_add[:,0], x_add[:,1], marker='*', s=100, c='white', edgecolor='black', zorder=4)
                plt.pcolor(X, Y, EI, cmap='jet', shading='auto', vmin=0, zorder=1)
                plt.colorbar()

    if nf==2:
        plt.figure('objective_space')
        plt.scatter(gp.f[:ns,0], gp.f[:ns,1], c='black', zorder=1)
        plt.scatter(gp.f[ns:,0], gp.f[ns:,1], c='white', edgecolor='black', zorder=2)
        plt.scatter(fg_add[:,0], fg_add[:,1], marker='*', s=100, c='white', edgecolor='black', zorder=3)
        plt.scatter(Fs[0].reshape(len(Fs[0][:,0])*len(Fs[0][0,:])), Fs[1].reshape(len(Fs[1][:,0])*len(Fs[1][0,:])), c=EI.reshape(len(EI[:,0])*len(EI[0,:])), cmap='jet', vmin=0, alpha=0.2, zorder=0)
        if CRITERIA=='EPBII':
            refvec = gp.utopia + (gp.nadir - gp.utopia)*gp.refvec[iref]
            plt.plot([gp.utopia[0], refvec[0]], [gp.utopia[1], refvec[1]], c='black', linestyle='dashed')
        elif CRITERIA=='EIPBII':
            refvec = gp.nadir + (gp.nadir - gp.utopia)*gp.refvec[iref]
            plt.plot([gp.nadir[0], refvec[0]], [gp.nadir[1], refvec[1]], c='white', linestyle='dashed')
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
        plt.scatter(fs[:,1], fs[:,0], c='none', edgecolor='blue', alpha=0.2)
        linear = [fs.min(), fs.max()]
        plt.plot(linear, linear, c='black')
    print(R2)
