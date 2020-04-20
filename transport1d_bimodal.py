# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:27:40 2019

@author: Gavin
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

#%%
def exp_integrand(w, parameters):
    """Evaluate the exponential integrand by evaluating the polynomial. It is exp(b1(w)). """
    return np.exp(np.polyval(parameters[::-1], w))

def S1(z, parameters):
    """ Returns the 1-D function S1 over an array of inputs z. """
    # Get the polynomial order
    a = parameters[0]
    factors = parameters[1:]
    return np.array([a + integrate.quad(exp_integrand, 0, zz, args=(factors))[0] for zz in z])

def objective_SAA(parameters, z):
    """ Returns the SAA objective function to minimise. Parameters should be fed in
    in the form a1, beta0, beta1, ...
    """
    # z should be a 1d vector of inputs
    return np.mean(0.5*np.power(S1(z, parameters), 2) - np.polyval(parameters[1:][::-1], z))

def optimal_parameters(objective, initial_parameters, target_samples):
    """ Performs the minimisation of the objective SAA to get the parameters 
    a1, beta0, beta1, ...
    """
    result = minimize(objective, initial_parameters, args=(target_samples))
    print(result.message)
    return result.x

def function_for_forward_root(z, parameters, x):
    """ This function defines f(z) = S1(z) - x whose root is z if z = S1^{-1}(x_k). """ 
    return S1(z, parameters) - x 

def T1(x, parameters):
    """ This function uses a Newton root finder to get the forward map given an
    array of samples from the reference. """
    return newton(function_for_forward_root, x0=np.zeros(x.size), args=(parameters, x))

def KL_div(approximate_target_samples, target_dist_parameters, true_target='Gamma'):
    """ This function calculates the approximate KL divergence between 
    a true density function and an approximate one.
    The approximation is done via Gaussian KDE.
    """
#    import pdb; pdb.set_trace()
    if true_target == 'Gamma':
        true_target_pdf = st.gamma.pdf(approximate_target_samples,
                                       a=target_dist_parameters[0],
                                       scale=1.0/target_dist_parameters[1])
    elif true_target == 'Beta':
        true_target_pdf = st.beta.pdf(approximate_target_samples,
                                      a=target_dist_parameters[0],
                                      b=target_dist_parameters[1])
        
    # Now approximate the approximate_target_pdf

    T1_samples_for_kde = approximate_target_samples[:, np.newaxis]
    T1_kde = KernelDensity(kernel='gaussian').fit(T1_samples_for_kde)
    log_density = T1_kde.score_samples(T1_samples_for_kde)
    approximate_target_pdf = np.exp(log_density)
    
    # If the true_target_pdf has any 0 entries, the entropy will blow up
    # Take these samples out
    if np.any(true_target_pdf == 0):
        print('PDF of 0 detected at ' + str(len(np.where(true_target_pdf == 0))) + ' points.')
        print('Removing those points.')
        
    return entropy(approximate_target_pdf[true_target_pdf > 0], true_target_pdf[true_target_pdf > 0])

def transport_1D(target_samples, reference_samples, 
                initial_parameters, target_dist_parameters, true_target='Gamma'):
    """ This function combines the above and packs them in a single function after
    generating target and reference samples (possibly different sizes).
    """
    param_opt = optimal_parameters(objective_SAA, initial_parameters, target_samples)
    
    approximate_reference_samples = S1(target_samples, param_opt)
    approximate_target_samples = T1(reference_samples, param_opt)
    
    kl_div = KL_div(approximate_target_samples, target_dist_parameters, true_target)
    
    transport = {}
    transport['param_opt'] = param_opt
    transport['approx_ref_samps'] = approximate_reference_samples
    transport['approx_target_samps'] = approximate_target_samples
    transport['ref_samps'] = reference_samples
    transport['tar_samps'] = target_samples
    transport['kl'] = kl_div
    
    return transport

def plot_transport(transport_1D,true_target,order):
    """ This is the main plotting function for the transports.
    """
    # 1. Initial inverse map from ref to approximate_target_samples
    fig1, ax1 = plt.subplots()
    sb.distplot(transport_1D['approx_ref_samps'], label=r'Approximate reference samples', ax=ax1)
    sb.distplot(transport_1D['tar_samps'], label=r'Target samples', ax=ax1)
    plt.legend(loc=2)
    plt.title(r'Inverse transport with {0} target'.format(true_target))
    plt.savefig(r'initial_inverse_{0}.pdf'.format(order))
    plt.show()
    
    # 2. The actual maps
    fig2 = plt.subplot()
    if (true_target == 'Gamma'):
        xs = np.linspace(0,5,200)
    elif (true_target == 'Beta'):
        xs = np.linspace(0,1,200)
        
    ys = S1(xs, transport_1D['param_opt'])
    plt.plot(xs,ys,label=r'Inverse map')
    plt.plot(ys,xs,label=r'Forward map')
    plt.title(r'Maps between Normal and {0} distributions'.format(true_target))
    plt.legend()
    plt.xlim=([-4, 5])
    plt.ylim=([-3, 5])
    plt.savefig(r'forward_inverse_{0}_{1}.pdf'.format(order, true_target))
    plt.show()
    # 3. 


#%%
def my_bimodal_normal(N, mu1, mu2, prob):
    b = st.bernoulli.rvs(p = prob, size=N)
    x = st.norm.rvs(loc = mu1, size=N)
    y = st.norm.rvs(loc = mu2, size=N)
    
    return x*(1-b)+y*b
#%% This part runs the functions above for the Gamma distribution
N = 1000 # The inverse map is created on 1000 samples
NN = 2000 # The testing is done on 2000 new samples
mu1 = 4
mu2 = -4
p=0.4
target_distribution_rvs = my_bimodal_normal(N,mu1, mu2,p)
reference_distribution = st.norm()

#%%
# Case 1: Linear function: b1(w) = beta0 + beta1*w 
# Initial guess for the parameters a, beta0, beta1
param_lin_0 = np.array([0.1, 0.4, -0.1])

opt = optimal_parameters(objective_SAA, param_lin_0, target_distribution_rvs)

new_target_samps = my_bimodal_normal(NN,mu1, mu2, p)
sb.distplot(S1(new_target_samps, opt))
#plt.title(r'Inverse transport with bimodal normal target, $\mu_1$ = {0}, $\mu_2$ = {1}, $p$ = {2}'.format(mu1, mu2, p))
plt.title('Approximated reference samples')
plt.savefig('bimodal_fail.pdf')
plt.show()

sb.distplot(new_target_samps)
plt.title('Bimodal normal target')
plt.savefig('bimodal_fail_target.pdf')
plt.show()

xs = np.linspace(-10,10,num=200)
ys = S1(xs,opt)
plt.plot(xs,ys)
plt.show()
#plot_transport(m_transport_1D, 'Bimodal', 'linear')

#%%
# Case 2: Quadratic function: b1(w) = beta0 + beta1*w + beta2*w^2
# Initial guess for the parameters a, beta0, beta1, beta2
param_quad_0 = np.array([0.1, 0.4, -0.1, 0.2])

#target_distribution_rvs = st.norm.rvs(size=N,loc=4.,scale=2.)
opt_quad = optimal_parameters(objective_SAA, param_quad_0, target_distribution_rvs)

new_target_samps = my_bimodal_normal(NN,mu1, mu2, 0.4)
#new_target_samps = st.norm.rvs(size=NN,loc=4.,scale=2.)
sb.distplot(S1(new_target_samps, opt_quad))
plt.title('Approximated reference samples')
plt.savefig('bimodal_fail.pdf')
plt.show()
#plot_transport(m_transport_1D_quad, 'Bimodal', 'quadratic')

xs = np.linspace(-8,8,num=200)
ys = S1(xs,opt_quad)
plt.plot(xs,ys)
#plt.title('Map between bimodal normal target and approximate reference')
#plt.savefig('bimodal_map_1d.pdf')
#plt.plot(ys,xs)
plt.show()

