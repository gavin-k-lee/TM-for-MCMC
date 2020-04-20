# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:40:17 2019

@author: Gavin

These functions implement inverse and forward transport maps
for 1D problems when the reference is a standard Gaussian
that is N(0,1).

It is currently supported for targets which have distributions
- Gamma
- Beta

ref refers to reference
tar refers to target
"""

#%%
from scipy import integrate
from scipy.optimize import minimize
from scipy.optimize import newton
import scipy.stats as st
from scipy.stats import entropy
from sklearn.neighbors.kde import KernelDensity

import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

#%% Generate the banana distribution as the target
N = 1000

def banana(N):
    z = st.norm.rvs(size=(N,2))
    t1 = 1.5*z[:,0]+2.5
    t2 = np.cos(z[:,0]) + 0.25 * z[:,1]
    rad = 3/4 * np.pi
    rot_mat = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    my_rot_mat = rot_mat(rad)
    my_rot_z = np.dot(np.column_stack((t1, t2)), my_rot_mat.T)
    
    return my_rot_z

my_banana = banana(N)

sb.jointplot(my_banana[:,0],my_banana[:,1],kind='scatter',color='gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.tight_layout()
#plt.savefig('banana.pdf', bbox_inches = "tight")
plt.show()

#%%
def exp_integrand_1d(w, param):
    """Evaluate the exponential integrand by evaluating the polynomial. It is exp(b1(w)). """
    return np.exp(np.dot(param, np.array([1, w])))

def exp_integrand_2d(w, z, param):
    """ Evaluates the 2D parameterisation where z should be a (Nx2) array"""
    return np.exp(np.dot(param, np.array([1, z[0], w])));

#%%
    
def S1(z, param):
    """ Param should be length 3"""
    return np.array([param[0] + integrate.quad(exp_integrand_1d, 0, zz, args=(param[-2:]))[0] for zz in z])

def S2(z, param):
    """ Param should be length 5"""
    return np.array([np.dot(param[:2],np.array([1, zz[0]])) + integrate.quad(exp_integrand_2d, 0, zz[1], args=(zz, param[-3:]))[0] for zz in z])

#%%

def objective_SAA(param, z, k):
    """ Returns the SAA objective function to minimise. Parameters should be fed in
    in the form a1, beta0, beta1, ...
    """
    # z should be a 1d vector of inputs
    if (k == 1):
        obj = np.mean(0.5*np.power(S1(z, param), 2) - np.array([np.dot(param[1:],np.array([1, zz])) for zz in z]))
    elif (k == 2):
        obj = np.mean(0.5*np.power(S2(z, param), 2) - np.array([np.dot(param[2:],np.array([1, zz[0], zz[1]])) for zz in z]))

    return obj

#%% Try the 2D minimisation on the Banana:
# Sequentially do the S1 and then S2 etc.
target_samples = banana(5000)
param_init_S1 = np.array([0.2, -0.1, 0.5]) # Corresponds to a1, beta0, beta1
res_S1 = minimize(objective_SAA, param_init_S1, args=(target_samples[:,0], 1)) # Only take the first column since this is for S1

param_final_S1 = res_S1.x

#%% Now do the S2 result

param_init_S2 = np.array([0.2, -0.1, 0.5, 0.4, -0.2])
res_S2 = minimize(objective_SAA, param_init_S2, args=(target_samples, 2))

param_final_S2 = res_S2.x

#%% Now plot to see what the result is
# Generate some new samples from the target
new_target_samples = banana(1000)
ref_S1 = S1(new_target_samples[:,0], param_final_S1)
ref_S2 = S2(new_target_samples, param_final_S2)

sb.jointplot(ref_S1,ref_S2,kind='scatter',color='lightblue').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
#plt.savefig('ref_2D.pdf', bbox_inches = "tight")
plt.show()

sb.jointplot(new_target_samples[:,0],new_target_samples[:,1],kind='scatter',color='gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
#plt.savefig('my_banana_linear.pdf',bbox_inches = 'tight')


#%% Test the distributions of the individual components
st.norm.fit(ref_S1)
st.norm.fit(ref_S2)


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

#%% Try the 2D minimisation on the Banana: QUADRATIC, NO CROSS TERMS
# Sequentially do the S1 and then S2 etc.
target_samples = banana(1000)
param_init_S1_quad = np.concatenate((param_final_S1, np.array([-2.0])),axis=None) # Corresponds to a1, beta0, beta1, beta2
res_S1_quad = minimize(objective_SAA_quad, param_init_S1_quad, args=(target_samples[:,0], 1)) # Only take the first column since this is for S1

param_final_S1_quad = res_S1_quad.x

#%% Now do the S2 result

param_init_S2_quad = np.array([0.2, -0.1, -0.1, -0.5, -0.7, -0.1, -2.5])
res_S2_quad = minimize(objective_SAA_quad, param_init_S2_quad, args=(target_samples, 2))

param_final_S2_quad = res_S2_quad.x

#%% Now plot to see what the result is
# Generate some new samples from the target
def bana(Ms):
    ApproxRefs = []
    for m in Ms:
        new_target_samples = banana(m)
        ref_S1_quad = S1_quad(new_target_samples[:,0], param_final_S1_quad)
        ref_S2_quad = S2_quad(new_target_samples, param_final_S2_quad)
        ApproxRefs.append(np.column_stack((ref_S1_quad,ref_S2_quad)))
        print("Done for " + str(m))
        
    return ApproxRefs

#sb.jointplot(ref_S1_quad,ref_S2_quad,kind='scatter',color='lightblue').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
#plt.xlabel(r'$x_1$')
#plt.ylabel(r'$x_2$')
#plt.tight_layout()
#plt.savefig('ref_2D_quad.pdf', bbox_inches = "tight")
#plt.show()


#sb.jointplot(new_target_samples[:,0],new_target_samples[:,1],kind='scatter',color='gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
#plt.xlabel(r'$\theta_1$')
#plt.ylabel(r'$\theta_2$')
#plt.tight_layout()
#plt.savefig('banana_quad.pdf', bbox_inches = "tight")
#plt.show()

#%% Test the distributions of the individual components
st.norm.fit(ref_S1_quad)
st.norm.fit(ref_S2_quad)


Bananas = []
Bananas = [banana(m) for m in M]

BananaParams = [];
BananaParams.append(inv_trans_quad(Bananas[0],2))
for i in range(1,len(Bananas)):
    BananaParams.append(inv_trans_quad(Bananas[i],2))

NewBananas = [banana(m) for m in M]
#ApproxRefs = [eval_S_quad(banana,2,params) for banana, params in zip(NewBananas, BananaParams)]
ApproxRefs = bana(M)
#M=2000
M = np.array([50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000])

p_vals_shapiro = np.zeros((len(M),2))
for i in range(len(M)):
    for j in range(2):
        _, p_vals_shapiro[i,j] = shapiro(ApproxRefs[i][:,j])

acpt_reject = np.zeros((len(M),2))
for i in range(len(M)):
    for j in range(2):
        a = anderson(ApproxRefs[i][:,j])
        if (a.statistic < a.critical_values[2]):
            acpt_reject[i,j] = 0 # Normal
        else:
            acpt_reject[i,j] = 1 # Not normal



#%% Now get the forward map
new_reference_samples = st.norm.rvs(size=1000*2).reshape(1000,2)
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


#%%
new_T1 = T1_quad(new_reference_samples[:,0], param_final_S1_quad)
# Feed in new_T1 into where z1 needs to be in the S2
new_T2 = T2_quad(new_reference_samples[:,1], new_T1, param_final_S2_quad)

#%% The forward map is then
sb.jointplot(new_reference_samples[:,0], new_reference_samples[:,1], kind='scatter', color = 'lightgreen').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
#plt.savefig('bivariate.pdf', bbox_inches = 'tight')
plt.show()


sb.jointplot(new_T1, new_T2, kind='scatter', color = 'pink').plot_joint(sb.kdeplot, n_levels=6, color='red')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
#plt.savefig('banana_quad_forward.pdf', bbox_inches = "tight")
plt.show()


#%% Try this on the 2-Banana problem:
def two_banana(N, theta1, theta2, shift11, shift12, shift21, shift22, prob):
    rot_mat = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    z = st.norm.rvs(size=(N,2))
    b = st.bernoulli.rvs(p=prob, size=N)
    
    z1 = z[np.where(b == 1)]
    z2 = z[np.where(b == 0)]
    
    t11 = 1.5*z1[:,0]+2.5 + shift11
    t12 = np.cos(z1[:,0]) + 0.25 * z1[:,1] + shift12
    
    t21 = 1.5*z2[:,0]+2.5 + shift21
    t22 = np.cos(z2[:,0]) + 0.25 * z2[:,1] + shift22
    
    my_rot_mat1 = rot_mat(theta1*np.pi)
    my_rot_mat2 = rot_mat(theta2*np.pi)
    
    my_rot_z1 = np.dot(np.column_stack((t11, t12)), my_rot_mat1.T)
    my_rot_z2 = np.dot(np.column_stack((t21, t22)), my_rot_mat2.T)
    return np.row_stack((my_rot_z1, my_rot_z2))
    
N=1000
my_bananas = two_banana(N, 3./4, 7./4, 0.5, 1.6, -4.5, 4., 0.3)
#%%
# Just try a bimodal normal in 2d...
def bimod_norm(N, shift11, shift12, shift21, shift22, prob):
    z = st.norm.rvs(size=(N,2))
    b = st.bernoulli.rvs(p = prob, size=N)
    
    z1 = z[np.where(b==1)]
    z2 = z[np.where(b==0)]
    
    # Shift z2 right and up 4 units
    t1 = z2[:,0] + shift21
    t2 = z2[:,1] + shift22
    
    s1 = z1[:,0] + shift11
    s2 = z1[:,1] + shift12
    
    return np.row_stack((np.column_stack((s1,s2)), np.column_stack((t1,t2))))
#%%



sb.jointplot(my_bananas[:,0],my_bananas[:,1],kind='scatter',color='gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.tight_layout()
#plt.savefig('banana.pdf', bbox_inches = "tight")
plt.show()

#%%

#banana_2d = two_banana(1000, 3./4, 7./4, 0.5, 1.6, -4.5, 4., 0.5)

my_bimod_norm = bimod_norm(1000,4,4,-2,-2,0.4)

param_init_S1_quad = np.array([0.2, -0.1, 0.5, -2.0]) # Corresponds to a1, beta0, beta1, beta2
res_S1_quad = minimize(objective_SAA_quad, param_init_S1_quad, args=(my_bimod_norm[:,0], 1)) # Only take the first column since this is for S1

param_final_S1_quad = res_S1_quad.x

#%% Now do the S2 result

param_init_S2_quad = np.array([0.2, -0.1, -0.1, -0.5, -0.7, -0.1, -2.5])
res_S2_quad = minimize(objective_SAA_quad, param_init_S2_quad, args=(my_bimod_norm, 2))

param_final_S2_quad = res_S2_quad.x

#%% Now plot to see what the result is
# Generate some new samples from the target
#new_target_samples = two_banana(2000,  3./4, 7./4, 0.5, 1.6, -4.5, 4., 0.3)
new_target_samples = bimod_norm(2000, 4, 4, -2, -2, 0.4)
ref_S1_quad = S1_quad(new_target_samples[:,0], param_final_S1_quad)
ref_S2_quad = S2_quad(new_target_samples, param_final_S2_quad)

sb.jointplot(ref_S1_quad,ref_S2_quad,kind='scatter',color='lightblue').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig('ref_2D_quad_bimodal.pdf', bbox_inches = "tight")
plt.show()


sb.jointplot(new_target_samples[:,0],new_target_samples[:,1],kind='scatter',color='gold').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.tight_layout()
plt.savefig('bimod_norm_2d.pdf', bbox_inches = "tight")
plt.show()

#% Forward map...
sb.jointplot(new_reference_samples[:,0], new_reference_samples[:,1], kind='scatter', color = 'lightgreen').plot_joint(sb.kdeplot, n_levels=6, color='darkgreen')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.savefig('bivariate_reference.pdf', bbox_inches = 'tight')
plt.show()

new_T1_bimod = T1_quad(new_reference_samples[:,0], param_final_S1_quad)
# Feed in new_T1 into where z1 needs to be in the S2
new_T2_bimod = T2_quad(new_reference_samples[:,1], new_T1_bimod, param_final_S2_quad)
sb.jointplot(new_T1_bimod, new_T2_bimod, kind='scatter', color = 'pink').plot_joint(sb.kdeplot, n_levels=6, color='red')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.savefig('approx_forward_bimod_2d.pdf')
plt.show()