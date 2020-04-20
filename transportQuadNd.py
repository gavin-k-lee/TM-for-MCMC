# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:25:56 2019

@author: Gavin
"""

#%%
import numpy as np
from numpy.polynomial.polynomial import polyval as polynom
from scipy import integrate
from scipy.optimize import newton
from scipy.optimize import minimize
import scipy.stats as st
import time

import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

#%%
def inv_trans_quad(z):
    """ This function implements a N dimensional inverse transport
    with a parsimonious map. Each element of the inverse transport has the form
    S^k(z1,...,z_k) = a_0^k + (Lin/Quad terms in z1, z2, ..., z_{k-1})
    + \int_0^z_k exp(b_0^k + b_1^k w + b_2^k w^2) dw
    
    
    Input: 
        - params should determine the order of polynomial (length 2*(k+1))
        - z should be a matrix of samples from the target, M * N, with
    M = number of samples
    N = number of dimensions
    
    Output:
        - array of array of parameters specifying all S^k, not a full matrix!
    """
    
#    M = np.size(z,0)
#    import pdb; pdb.set_trace();
    if (len(np.shape(z)) == 1):
        N = 1
    else:
        N = np.shape(z)[1]
    
    final_params = [None]*N
    
    for k in range(N):
        # We need to create the polynomial at each stage,
        # or have a function that will create it
        final_params[k] = np.zeros(2*(k+2)) # k starts from 0
        # Set random start values for the initialisation
        init_params = st.norm.rvs(size=2*(k+2), scale=0.2)
#        import pdb; pdb.set_trace();
        print('Starting inverse optimisation for component {:d}'.format(k+1))
        s = time.time()
        result = minimize(obj_quad_inv, init_params, args=(z[:,0:k+1]))
        e = time.time()
        print(result.message + ' for component {:d} in time {:.2f}'.format(k+1, e-s))
        final_params[k] = result.x
        
    return final_params

def exp_integ(w, params):
    """ This function evaluates the exponential integrand
    for the diagonal map. It does not depends on the data z.
    """
    return np.exp(polynom(w, params))

def S_quad_vec(z, params):
    """ Only takes in one sample of z, i.e. a 1*k vector, params = 2(k+1)
    The evaluation goes constant + all linear terms + all qudratic + integral
    """
#    import pdb; pdb.set_trace();
    k = np.shape(z)[0] # dimension of the z vector
    assert 2*(k+1) == len(params), 'Dimensions dont match...'
    
    S = params[0] + integrate.quad(exp_integ, 0, z[-1], args=(params[-3:]))[0]
    for i in range(k-1):
        S = S + params[i+1]*z[i] # Add the linear terms
        S = S + params[k+i]*z[i]**2 # Add the quadratic terms
        
    return S
        
def S_quad(z, params):
    """ This function evaluates S
    parameter vector and a matrix of samples 
    """
#    import pdb; pdb.set_trace();
    S = np.zeros(np.shape(z)[0])
    
    for m in range(np.shape(z)[0]):
       S[m] = S_quad_vec(z[m], params) 

    return S # This should be a length M vector

def obj_quad_inv(params, z):
    """ Computes the objective function for the inverse transport
    z = z_k is one dimension, with length M, the number of samples
    """
    return np.mean(0.5*(S_quad(z, params))**2 - polynom(z[:,-1], params[-3:]))

def eval_approx_ref(z, params):
    """ Assembles each component of the map given target samples and parameters"""
#    import pdb; pdb.set_trace();
    N = z.ndim
    approx_ref = np.zeros(np.shape(z)) # Should be the same shape as z (1-1)

    if (N==1):
        approx_ref = S_quad(z, params)
        return approx_ref
    else:
        approx_ref[:,0] = S_quad(z[:,[0]], params[0])
        for k in range(N-1):
            approx_ref[:,k+1] = S_quad(z[:,0:k+2], params[k+1])
        
        return approx_ref   

#%%
    
#def obj_quad_for(params, x, z):
#    """ Computes the objective function for the forward transport
#    Since we assume the form to be the same as S, reuse
#    """
##    import pdb; pdb.set_trace();
#    return np.sum((np.reshape(S_quad(x, params),(np.shape(z)[0],1)) - z)**2)
#
#def for_trans_quad(x, z):
#    """ This function implements a N dimensional forward transport
#    with a parsimonious map. Each element of the inverse transport has the form
#    T^k(x1,...,x_k) = a_0^k + (Lin/Quad terms in x1, x2, ..., x_{k-1})
#    + \int_0^x_k exp(b_0^k + b_1^k w + b_2^k w^2) dw
#    
#    
#    Input: 
#        - Standard Gaussian in N dimensions
#        - x should be a matrix of samples from the target, M * N, with
#    M = number of samples
#    N = number of dimensions
#    
#    Output:
#        - M*N matrix with the approximated target
#    """
#    
##    M = np.size(z,0)
##    import pdb; pdb.set_trace();
#    if (len(np.shape(x)) == 1):
#        N = 1
#    else:
#        N = np.shape(x)[1]
#    
#    final_params = [None]*N
#    
#    for k in range(N):
#        # We need to create the polynomial at each stage,
#        # or have a function that will create it
#        final_params[k] = np.zeros(2*(k+2)) # k starts from 0
#        # Set random start values for the initialisation
#        init_params = st.norm.rvs(size=2*(k+2), scale=0.2)
##        import pdb; pdb.set_trace();
#        print('Starting forward optimisation for component {:d}'.format(k+1))
#        s = time.time()
#        result = minimize(obj_quad_for, init_params, args=(x[:,0:k+1], z[:,0:k+1]))
#        e = time.time()
#        print(result.message + ' for component {:d} in time {:.2f}'.format(k+1, e-s))
#        final_params[k] = result.x
#        
#    return final_params
#
#def eval_approx_tar(x, params):
#    """ Assembles each component of the map given target samples and parameters"""
#    N = np.shape(x)[1]
#    approx_tar = np.zeros(np.shape(x)) # Should be the same shape as z (1-1)
##    import pdb; pdb.set_trace();
#    approx_tar[:,0] = S_quad(x[:,[0]], params[0])
#    for k in range(N-1):
#        approx_tar[:,k+1] = S_quad(x[:,0:k+2], params[k+1])
#    
#    return approx_tar   

#%% Try Newton
#def function_for_forward_root(z, parameters, x):
#    """ This function defines f(z) = S1(z) - x whose root is z if z = S1^{-1}(x_k). """ 
#    return S_quad(z, parameters) - x 

def T_quad(x, params):
    """ This function uses a Newton root finder to get the forward map given an
    array of samples from the reference. """
    # x should be M*N matrix
    N = np.shape(x)[1]
    z = np.zeros(np.shape(x)) # Should be the same shape
#    import pdb; pdb.set_trace();
    for k in range(N):
#        z_k = z[:,[k]] # Should just be a M*1 vector
        param_k = params[k]
        x_k = x[:,k]
        func = lambda z_k, param_k, x_k : eval_approx_ref(z_k, param_k).reshape(-1,1) - x_k.reshape(-1,1)
        # Find the zs which correspond to x 
#        import pdb; pdb.set_trace();
        print('Using Newton to get the forward map for component {:d}'.format(k+1))
        z[:,k] = newton(func, x0=st.norm.rvs(size=np.shape(x)[0],scale=0.2), args=(param_k, x_k))

    return 0

#%%
M = 200
N = 3
alpha = 5.5
beta = 3.1

#init_params = np.array([0.1, 0.4, -0.1, -0.5])
tar_samps = st.gamma.rvs(size=(M,N), a=alpha, scale=1./beta)
#sb.jointplot(tar_samps[:,0], tar_samps[:,1], kind='scatter', color = 'lightgreen').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
#plt.show()

final_params = inv_trans_quad(z=tar_samps)

new_tar_samps = st.gamma.rvs(size=(2*M,N), a=alpha, scale=1./beta)

approx_ref = eval_approx_ref(new_tar_samps, final_params)

sb.jointplot(approx_ref[:,1], approx_ref[:,2], kind='scatter', color = 'pink').plot_joint(sb.kdeplot, n_levels=6, color='red')
plt.show()

#%% Now do the forward map
















#%% Banana test
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
my_banana = banana(M)
sb.jointplot(my_banana[:,0], my_banana[:,1], kind='scatter', color = 'gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.show()

banana_params = inv_trans_quad(z=my_banana)
approx_ref_banana = eval_approx_ref(banana(2*M), banana_params)
sb.jointplot(approx_ref_banana[:,0], approx_ref_banana[:,1], kind='scatter', color = 'pink').plot_joint(sb.kdeplot, n_levels=6, color='red')
plt.show()

#%% Test the forward maps obtained by least squares
ref_samps = st.norm.rvs(size=(M,2))
# Could also use approx_ref I guess
T_quad(x=ref_samps, params=banana_params)
final_forward_params = for_trans_quad(x=ref_samps, z=tar_samps)
approx_tar = eval_approx_tar(ref_samps, final_forward_params)
#approx_tar_newton = T_quad(ref_samps, final_forward_params)
sb.jointplot(approx_tar[:,0], approx_tar[:,1], kind='scatter', color = 'pink').plot_joint(sb.kdeplot, n_levels=6, color='red')
plt.show()