# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:10:26 2019

@author: Gavin
"""

from scipy import integrate
from scipy.optimize import minimize
from scipy.optimize import newton
import scipy.stats as st
from scipy.stats import entropy
from sklearn.neighbors.kde import KernelDensity
import time

import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams.update({'font.size': 14})


#%% Now try the quadratic case
def exp_integrand_1d_quad(w, param):
    """ Param should be length 3"""
    return np.exp(np.dot(param, np.array([1, w, w**2])))

def exp_integrand_2d_quad(w, z, param):
    """ Param should be length 5"""
    return np.exp(np.dot(param, np.array([1, z[0], z[0]**2, w, w**2])))

#%%
def S1_quad(z, param):
    """ Param should be length 4"""
    return np.array([param[0] + integrate.quad(exp_integrand_1d_quad, 0, zz, args=(param[1:]))[0] for zz in z])

def S2_quad(z, param):
    """ Param should be legnth 7"""
    return np.array([np.dot(param[0:2],np.array([1, zz[0]])) + integrate.quad(exp_integrand_2d_quad, 0, zz[1], args=(zz, param[2:]))[0] for zz in z])

#%%
def objective_SAA_quad(param, z, k):
    """ Returns the SAA objective function to minimise. Parameters should be fed in
    in the form a1, beta0, beta1, ...
    """
    # z should be a 1d vector of inputs
    if (k == 1):
        obj = np.mean(0.5*np.power(S1_quad(z, param), 2) - np.array([np.dot(param[1:],np.array([1, zz, zz**2])) for zz in z]))
    elif (k == 2):
        obj = np.mean(0.5*np.power(S2_quad(z, param), 2) - np.array([np.dot(param[2:],np.array([1, zz[0], zz[0]**2, zz[1], zz[1]**2])) for zz in z]))

    return obj

#%%
# Get the first column first
def forward_root_T1(z, parameters, x):
    """ This function defines f(z) = S1(z) - x whose root is z if z = S1^{-1}(x_k). """ 
    return S1_quad(z, parameters) - x 

def T1_quad(x, parameters):
    """ This function uses a Newton root finder to get the forward map given an
    array of samples from the reference. """
    return newton(forward_root_T1, x0=np.zeros(x.size), args=(parameters, x))
#%%
    
def exp_integrand_2d_quad_fixed_linear(w, param, z1):
    return np.exp(np.dot(param, np.array([1, z1, z1**2, w, w**2])))

def S2_quad_fixed_linear(z, z1, param):
    """ Param should be legnth 7"""
    return np.array([np.dot(param[0:2],np.array([1, zz1])) + integrate.quad(exp_integrand_2d_quad_fixed_linear, 0, zz, args=(param[2:], zz1))[0] for zz,zz1 in zip(z,z1)])

def forward_root_T2(z, T1, parameters, x):
    """ This function defines f(z) = S1(z) - x whose root is z if z = S1^{-1}(x_k). """ 
    return S2_quad_fixed_linear(z, T1, parameters) - x 

def T2_quad(x, T1, parameters):
    """ This function uses a Newton root finder to get the forward map given an
    array of samples from the reference. """
    return newton(forward_root_T2, x0=np.zeros(x.size), args=(T1, parameters, x))

def est_KL(N):
    N = 5000
    tar_samps = banana(N)
    param_init_S1_quad = np.array([1.69, -0.827, -0.37, -0.2])
    res_S1_quad = minimize(objective_SAA_quad, param_init_S1_quad, args=(tar_samps[:,0], 1)) # Only take the first column since this is for S1
    param_final_S1_quad = res_S1_quad.x
    print('First component of inverse computed')
    param_init_S2_quad = np.array([0.25, 1.5, 1.55, 0.15, -0.7, -0.1, -2.5])
    res_S2_quad = minimize(objective_SAA_quad, param_init_S2_quad, args=(tar_samps, 2))
    param_final_S2_quad = res_S2_quad.x
    print('Second component of inverse computed')
    
    new_ref_samps = st.norm.rvs(size=(N,2))
#    print(new_ref_samps[0])
#    import pdb; pdb.set_trace();
    T1_approx = T1_quad(new_ref_samps[:,0], param_final_S1_quad)
    print('First component of forward computed')
# Feed in new_T1 into where z1 needs to be in the S2
    T2_approx = T2_quad(new_ref_samps[:,1], T1_approx, param_final_S2_quad)
    print('Second component of forward computed')
    # Now calculate the KL between [T1_approx, T2_approx] and new_tar_samps
    T12_kde_approx = np.column_stack((T1_approx,T2_approx))
    kde_approx = KernelDensity(kernel='gaussian').fit(T12_kde_approx) # Fit the approximate forward map
    log_dens_approx = kde_approx.score_samples(T12_kde_approx)

    # generate some new target_samps
    new_tar_samps = banana(N)
    T12_kde_true = new_tar_samps
    kde_true = KernelDensity(kernel='gaussian').fit(T12_kde_true)
    log_dens_true = kde_true.score_samples(T12_kde_true)


    PI_true = np.exp(log_dens_true)
    PI_approx = np.exp(log_dens_approx)
    
    # Remove any negative terms
##    neg_true_idx = np.where(PI_true <= 0)[0]
#    neg_approx_idx = np.where(PI_approx <= 0)[0]
#    negs = list(set(neg_true_idx).intersection(neg_approx_idx))
#    S = entropy(PI_approx[PI_approx != negs], PI_true[PI_true != negs])
    S = entropy(PI_approx, PI_true)
    return S

#%%
def banana(N):
#    np.random.seed(seed=42)
    z = st.norm.rvs(size=(N,2))
    t1 = 1.5*z[:,0]+2.5
    t2 = np.cos(z[:,0]) + 0.25 * z[:,1]
    rad = 3/4 * np.pi
    rot_mat = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    my_rot_mat = rot_mat(rad)
    my_rot_z = np.dot(np.column_stack((t1, t2)), my_rot_mat.T)
    
    return my_rot_z

#%%
    
#Ns = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
Ns = np.array([1000, 2000])
KLs = np.zeros(len(Ns))
Times = np.zeros(len(Ns))
# set the seed to be 42 to generate the banana consistently
#np.random.seed(seed=42)

#KLs = [est_KL(n) for n in Ns]

for i in range(len(Ns)):
    np.random.seed(seed=42)
    start = time.time()
    print('N = ' + str(Ns[i]))
    KLs[i] = est_KL(Ns[i])
    print('N = ' + str(i) + ' KL = ' + str(KLs[i]))
    end = time.time()
    Times[i] = end - start
#KLs
    
