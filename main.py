# -*- coding: utf-8 -*-
"""
main.py
Copyright (c) 2022 Nobuo Namura
This code is released under the MIT License, see LICENSE.txt.

This Python code is for single/multi-objective Bayesian optimization (MBO) with/without constraint handling.
MBO part is based on MBO-EPBII-SRVA and MBO-EPBII published in the following articles:
・N. Namura, "Surrogate-Assisted Reference Vector Adaptation to Various Pareto Front Shapes 
  for Many-Objective Bayesian Optimization," IEEE Congress on Evolutionary Computation, 
  Krakow, Poland, pp.901-908, 2021.
・N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary 
  Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary 
  Computation, vol. 21, no. 6, pp. 898-913, 2017.
Please cite the article(s) if you use the MBO code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools
import time
import shutil
import os

from bo import BayesianOptimization
import test_problem
from initial_sample import generate_initial_sample
import indicator

#======================================================================
if __name__ == "__main__":
    division = pd.read_csv('reference_vector_division.csv', index_col=0)
    #test problem
    func_name = 'SGM'                        # Test problem name in test_problem.py
    seed = 3                                 # Random seed for SGM function
    nx = 2                                   # Number of design variables (>=1)
    nf = 2                                   # Number of objective functions (>=1)
    ng = 1                                   # Number of constraint functions where g <= 0 is satisfied for feasible solutions (>=0)
    k = 1                                    # Position paramete k in WFG problems
    xmin = np.full(nx, 0.0)                  # Lower bound of design sapce
    xmax = np.full(nx, 1.0)                  # Upper bound of design sapce
    MIN = np.full(nf, True)                  # True=Minimization, False=Maximization
    #initial sample
    GENE = True                              # True=Generate initial sample with LHS, False=Read files
    ns = 30                                  # If GENE=True, number of initial sample points (>=2)
    #Bayesian optimization
    n_trial = 1                              # Number of independent run with different initial samples (>=1)
    n_add = 5                                # Number of additional sample points at each iteration (>=1)
    ns_max = 40                              # Number of maximum function evaluation
    CRITERIA = 'EPBII'                       # EPBII or EIPBII for multi-objective problems, EI, GP-MI, Error, or Estimation for single-objective problems
    NOISE = np.full(nf+ng,False)             # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    #for multiobjective problems
    SRVA = True                              # True=surrogate-assisted reference vector adaptation, False=two-layered simplex latice-design
    OPTIMIZER = 'NSGA3'                      # NSGA3 or NSGA2 for ideal and nadir point determination (and reference vector adaptation if SRVA=True)
    n_randvec = division.loc[nf, 'npop']     # Number of adaptive reference vector for EPBII/EIPBII (>=0)
    nh = 0 #division.loc[nf, 'nh']           # Division number for the outer layer of the two-layered simplex latice-design for EPBII/EIPBII (>=0)
    nhin = 0 #division.loc[nf, 'nhin']       # Division number for the inner layer of the two-layered simplex latice-design for EPBII/EIPBII (>=0)
    #evolutionary algorithm
    ngen_ea = 200                            # Number of generation
    npop_ea = division.loc[nf, 'npop_ea']    # Number of population for NSGA2 for multi-objective problems and GA for single-objective problems
    nh_ea = division.loc[nf, 'nh_ea']        # Division number for the outer layer of the two-layered simplex latice-design for NSGA3 (>=0)
    nhin_ea = division.loc[nf, 'nhin_ea']    # Division number for the inner layer of the two-layered simplex latice-design for NSGA3 (>=0)
    n_randvec_ea = 0                         # Number of random reference vector for NSGA3 (>=0)
    #others
    hv_ref = np.array([0.2, 0.15])           # reference point for hypervolume
    IGD_plus = True                          # True=IGD+, False=IGD
    PLOT = True                              # True=Plot the results
    RESTART = False                          # True=Read sample*_out.csv if it exists, False=Read sample*.csv
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    fname_indicator = 'indicators'
    path_IGD_ref = current_dir + '/IGD_ref'
    """=== Edit End ================================================="""
    
    #Initial sample
    problem = test_problem.define_problem(func_name, nf, ng, k, seed)
    if GENE:
        df_samples, df_design_space = generate_initial_sample(func_name, nx, nf, ng, ns, n_trial, xmin, xmax, current_dir, fname_design_space, fname_sample, k=k, seed=seed, FILE=True)
    else:
        df_design_space = pd.read_csv(current_dir + '/' + fname_design_space + '.csv')
    
    if func_name == 'SGM':
        file = path_IGD_ref + '/' + func_name + str(seed) + 'x' + str(nx) +'f' + str(nf) + '.csv'
        if os.path.exists(file):
            IGD_FLAG = True
            igd_ref = np.loadtxt(path_IGD_ref + '/' + func_name + str(seed) + 'x' + str(nx) +'f' + str(nf) + '.csv', delimiter=',')
        else:
            IGD_FLAG = False
    else:
        file = path_IGD_ref + '/' + func_name + 'f' + str(nf) + '.csv'
        if os.path.exists(file):
            IGD_FLAG = True
            igd_ref = np.loadtxt(path_IGD_ref + '/' + func_name + 'f' + str(nf) + '.csv', delimiter=',')
        else:
            IGD_FLAG = False        
        
    #Preprocess for RMSE
    if nx == 2:
        ndiv = 101
        x_rmse0 = np.zeros([ndiv**2, nx])
        for i in range(101):
            for j in range(101):
                x_rmse0[i*ndiv+j,0] = float(i)/float(ndiv-1)
                x_rmse0[i*ndiv+j,1] = float(j)/float(ndiv-1)
    else:
        x_rmse0 = np.random.uniform(size=[10000, nx])

    #Independent run
    print('EGO')
    for itrial in range(1,n_trial+1,1):
        #Preprocess
        print('trial '+ str(itrial))
        if RESTART:
            f_sample = current_dir + '/' + fname_sample + str(itrial) + '_out.csv'
            FILEIN = False
            if not os.path.exists(f_sample):
                f_sample = current_dir + '/' + fname_sample + str(itrial) + '.csv'
                FILEIN = True
        else:
            f_sample = current_dir + '/' + fname_sample + str(itrial) + '.csv'
            FILEIN = True
        df_sample = pd.read_csv(f_sample)
        
        gp = BayesianOptimization(df_sample, df_design_space, MIN)        
        x_rmse = gp.xmin + (gp.xmax-gp.xmin)*x_rmse0
        max_iter = int((ns_max + (n_add - 1) - gp.ns)/n_add)
        rmse = np.zeros([max_iter, gp.nf + gp.ng])
        igd = np.zeros(max_iter+1)
        hv = np.zeros(max_iter+1)
        times = []
        rank = gp.pareto_ranking(gp.f, gp.g)
        if not IGD_FLAG:
            igd[0] = np.nan
        else:
            igd[0] = indicator.igd_history(gp.f[rank==1.0], igd_ref, IGD_plus, MIN)
        hv[0] = indicator.hv_history(gp.f[rank==1.0], hv_ref, MIN)
        f_indicator = current_dir + '/' + fname_indicator + str(itrial) +'.csv'
        if FILEIN:
            with open(f_indicator, 'w') as file:
                data = ['iteration', 'samples', 'time', 'IGD', 'Hypervolume']
                for i in range(gp.nf + gp.ng):
                    data.append('RMSE'+str(i+1))
                data = np.array(data).reshape([1,len(data)])
                np.savetxt(file, data, delimiter=',', fmt = "%s")
            f_sample_out =  current_dir + '/' + fname_sample + str(itrial) + '_out.csv'
            shutil.copyfile(f_sample, f_sample_out)
        else:
            f_sample_out = f_sample
        
        #Main loop for EGO
        for itr in range(max_iter):
            times.append(time.time())
            print('=== Iteration = '+str(itr)+', Number of sample = '+str(gp.ns)+' ======================')
            
            #Kriging and infill criterion
            theta = gp.training(theta0 = 3.0, npop = 100, ngen = 100, mingen=0, STOP=True, NOISE=NOISE) # reasonable
            # theta = gp.training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE) # setting used in the papers
            gp.construction(theta)
            if gp.nf == 1:
                x_add, f_add_est, g_add_est = gp.optimize_single_objective_problem(CRITERIA, n_add, npop_ea, ngen_ea, theta0=3.0, npop=100, ngen=100, mingen=0, STOP=True, PRINT=False, RETRAIN=True)
            elif nf > 1:
                x_add, f_add_est, g_add_est = gp.optimize_multiobjective_problem(CRITERIA, OPTIMIZER, SRVA, n_add, n_randvec, nh, nhin, n_randvec_ea, nh_ea, nhin_ea, npop_ea, ngen_ea, pbi_theta=1.0, PLOT=False, PRINT=True)
            times.append(time.time())

            #RMSE
            for ifg in range(gp.nf + gp.ng):
                krig = functools.partial(gp.estimate_f, nfg=ifg)
                rmse[itr, ifg] = indicator.rmse_history(x_rmse, problem, krig, ifg)

            #Add sample points
            fg_add = np.array([problem(x_add[i]) for i in range(len(x_add))])
            gp.add_sample(x_add, fg_add[:,:gp.nf], fg_add[:,gp.nf:])
            
            #Indicators and file output
            with open(f_indicator, 'a') as file:
                data = np.hstack([itr, gp.ns-gp.n_add, times[-1]-times[-2], igd[itr], hv[itr], rmse[itr, :]])
                np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
            with open(f_sample_out, 'a') as file:
                data = np.hstack([gp.x[-gp.n_add:,:], gp.f[-gp.n_add:,:], gp.g[-gp.n_add:,:]])
                np.savetxt(file, data, delimiter=',')
            rank = gp.pareto_ranking(gp.f, gp.g)
            if not IGD_FLAG:
                igd[itr+1] = np.nan
            else:
                igd[itr+1] = indicator.igd_history(gp.f[rank==1.0], igd_ref, IGD_plus, MIN)
            hv[itr+1] = indicator.hv_history(gp.f[rank==1.0], hv_ref, MIN)
            if itr == max_iter-1:
                with open(f_indicator, 'a') as file:
                    data = np.array([itr+1, gp.ns, 0.0, igd[itr+1], hv[itr+1]])
                    np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
            
            #Visualization
            if PLOT:
                pareto = rank==1.0
                feasible = ~np.any(gp.g>0, axis=1)
                if nf == 2:
                    plt.figure('2D Objective-space '+func_name+' with '+str(gp.ns-gp.n_add)+'-samples')
                    plt.scatter(gp.f[feasible,0], gp.f[feasible,1], marker='o', c='black', s=10, label='feasible sample points')
                    plt.scatter(gp.f[~feasible,0], gp.f[~feasible,1], marker='x', c='black', s=10, label='infeasible sample points')
                    plt.scatter(gp.f_opt[:,0], gp.f_opt[:,1], marker='o', c='grey', s=10, label='estimated PF')
                    plt.plot(gp.utopia[0], gp.utopia[1], '+', c='black', label='utopia point')
                    plt.plot(gp.nadir[0], gp.nadir[1], '+', c='black', label='nadir point')
                    plt.scatter(gp.f_candidate[:,0], gp.f_candidate[:,1], c=gp.fitness_org, cmap='jet', marker='o', s=40, label='candidate points')
                    plt.scatter(f_add_est[:,0],f_add_est[:,1], facecolors='none', edgecolors='magenta', marker='o', s=60, linewidth=2, label='selected candidate points')
                    plt.scatter(gp.f[-gp.n_add:,0], gp.f[-gp.n_add:,1], facecolors='magenta', marker='x', s=30, linewidth=1.5, label='additional sample points')
                    plt.legend()
                    plt.show(block=False)
                    title = current_dir + '/2D_Objective_space_'+func_name+' with '+str(gp.ns-gp.n_add)+'-samples_in_'+str(itrial)+'-th_trial.png'
                    plt.savefig(title, dpi=300)
                    plt.close()
                    
                    plt.figure('solutions on 2D Objective-space '+func_name+' with '+str(gp.ns)+'-samples')
                    if not IGD_FLAG:
                        pass
                    else:
                        plt.scatter(igd_ref[:,0], igd_ref[:,1],c='green',s=1)
                    plt.scatter(gp.f[pareto,0], gp.f[pareto,1],c='blue',s=20,marker='o')
                    title = current_dir + '/Optimal_solutions_'+func_name+' with '+str(gp.ns)+'-samples_in_'+str(itrial)+'-th_trial.png'
                    plt.savefig(title)
                    plt.close()
                    
                elif nf == 3:
                    fig = plt.figure('3D Objective-space '+func_name+' with '+str(gp.ns-gp.n_add)+'-samples')
                    ax = Axes3D(fig)
                    # ax.scatter3D(gp.f[rank>1,0], gp.f[rank>1,1], gp.f[rank>1,2], marker='o', c='black', s=10, label='sample points')
                    # ax.scatter3D(gp.f_opt[:,0], gp.f_opt[:,1], gp.f_opt[:,2], marker='o', c='grey', s=10, alpha=0.5, label='estimated PF')
                    ax.scatter3D(gp.f[pareto,0], gp.f[pareto,1], gp.f[pareto,2], marker='o', c='blue', s=20, label='NDSs among sample points')
                    ax.scatter3D(gp.f_candidate[:,0], gp.f_candidate[:,1], gp.f_candidate[:,2], c=gp.fitness_org, cmap='jet', marker='*', s=40, label='candidate points')
                    ax.scatter3D(f_add_est[:,0], f_add_est[:,1], f_add_est[:,-1], marker='o', c='none', edgecolor='magenta', s=60, linewidth=2, label='selected candidate points')
                    ax.scatter3D(gp.f[-gp.n_add:,0],gp.f[-gp.n_add:,1],gp.f[-gp.n_add:,-1], marker='o', c='none', edgecolor='black', s=60, linewidth=2, label='additional sample points')
                    ax.view_init(elev=30, azim=45)
                    plt.legend()
                    title = current_dir + '/3D_Objective_space_'+func_name+' with '+str(gp.ns-gp.n_add)+'-samples_in_'+str(itrial)+'-th_trial.png'
                    plt.savefig(title)
                    plt.close()
                    
                    fig2 = plt.figure('solutions on 3D Objective-space '+func_name+' with '+str(gp.ns)+'-samples')
                    ax2 = Axes3D(fig2)
                    if not IGD_FLAG:
                        pass
                    else:
                        ax2.scatter3D(igd_ref[:,0],igd_ref[:,1],igd_ref[:,-1],c='green',s=1)
                    ax2.scatter3D(gp.f[pareto,0],gp.f[pareto,1],gp.f[pareto,-1],c='blue',s=20,marker='o')
                    ax2.view_init(elev=30, azim=45)
                    title = current_dir + '/Optimal_solutions_'+func_name+' with '+str(gp.ns)+'-samples_in_'+str(itrial)+'-th_trial.png'
                    plt.savefig(title)
                    plt.close()
    if n_trial > 1:
        dfs = []
        for i in range(n_trial):
            path = current_dir + '/' + fname_indicator + str(i+1) + '.csv'
            df = pd.read_csv(path)
            dfs.append(df.values)
        dfs = np.array(dfs)
        mean = np.mean(dfs, axis=0)
        std = np.std(dfs, axis=0)
        df_mean = pd.DataFrame(mean, columns=df.columns)
        df_std = pd.DataFrame(std, columns=df.columns)
        dfs = pd.concat([df_mean, df_std],axis=1)
        df_mean.to_csv(current_dir + '/' + fname_indicator + '_mean.csv', index=None)