# -*- coding: utf-8 -*-
"""
initial_sample.py
Copyright (c) 2022 Nobuo Namura
This code is released under the MIT License.
"""

import numpy as np
import pandas as pd
from pyDOE2 import lhs
import test_problem

#======================================================================
def generate_initial_sample(func_name, nx, nf, ng, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample, k=4, seed=3, FILE=True):
    func = test_problem.define_problem(func_name, nf, ng, k, seed)
    df_design_space = pd.DataFrame()
    df_design_space['min'] = xmin
    df_design_space['max'] = xmax
    if FILE:
        df_design_space.to_csv(current_dir+'/'+fname_design_space+'.csv', index=None)
    
    print('Initial sample generation')
    df_samples = []
    for itrial in range(ntrial):
        print('trial '+ str(itrial+1))
        x = lhs(nx, samples=ns, criterion='cm',iterations=1000)
        x = xmin + x*(xmax - xmin)
        
        f = np.zeros((ns,nf+ng))
        for i in range(ns):
            f[i,:] = func(x[i,:])
        
        df = pd.DataFrame()
        for i in range(nx):
            df['x'+str(i+1)] = x[:,i]
        for i in range(nf):
            df['f'+str(i+1)] = f[:,i]
        for i in range(ng):
            df['g'+str(i+1)] = f[:,nf+i]
        
        df_samples.append(df)
        if FILE:
            fname = fname_sample+str(itrial+1)+'.csv'
            df.to_csv(fname,index=None)
    
    return df_samples, df_design_space

#======================================================================
if __name__ == "__main__":
    func_name = 'ZDT3'
    seed = 3
    nx = 2
    nf = 2
    ng = 0
    ns = nx*11-1
    k = 1
    ntrial = 1
    xmin = np.full(nx, 0.0)
    xmax = np.full(nx, 1.0)
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    df_samples, df_design_space = generate_initial_sample(func_name, nx, nf, ng, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample, k=k, seed=seed, FILE=True)