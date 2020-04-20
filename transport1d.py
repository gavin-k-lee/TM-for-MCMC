# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:24:08 2019

@author: Gavin
"""

#%%
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats as stats
#from datetime import datetime
#startTime = datetime.now()
#fun()
#print(datetime.now() - startTime)
#0:00:17.443896

#%%
ref_samps = np.random.normal(size=(10000))

def analytic_map(samples, k):
    return samples**2/k

tar_samps = analytic_map(ref_samps, 1.4)
sb.distplot(ref_samps)
sb.distplot(tar_samps) #, kde=False, fit=st.gamma)
plt.show()

#%% Inverse
target_samps = np.random.gamma(shape=0.5, scale=1./10, size=500)

def inverse_analytic_map(samples, k):
    return np.sqrt(samples*k)

reference_samps = inverse_analytic_map(target_samps,1.4) # Will be distribution of |Z| not Z~N(0,1) since not bijective
sb.distplot(target_samps)
sb.distplot(reference_samps)
plt.show()

#%%
# Attempt at 'automatically' satisfying the constraints by assuming the form of each component
# Param = [a, alpha, beta] where we assume S(z) to have the form
# S(z) = a + \int_0^{z_1} \exp(b1(w)) dw = a + \int_0^{z_1} \exp(alpha + beta*w)dw
# so we have assumed the form of b1(w) to be linear
def obj(param,z):
    # z should be a 1d vector of inputs
    return np.mean(0.5*(param[0] + 1./param[2] * (np.exp(param[1] + param[2]*z) - np.exp(param[1])))**2 - (param[1] + param[2]*z))

#%%
# This is the target data (Gamma(5.5,3.1))
#PI = np.random.gamma(5.5, 1./3.1, size=1000)
PI = st.gamma.rvs(a=5.5,loc=0,scale=1./3.1,size=1000)
# Starting values a0, alpha0, beta0, chose at random...
param0 = np.array([0.1, 0.4, -0.1])
# Perform the minimisation
from scipy.optimize import minimize
res = minimize(obj, param0, args=(PI))
#%%
# Get the values which minimise the objective
a_star = res.x[0]
alpha_star = res.x[1]
beta_star = res.x[2]
# Define the inverse transport
def S(z,a,alpha,beta):
    return a + 1.0/beta * (np.exp(alpha + beta*z) - np.exp(alpha))
#%%
# Pull back the samples PI~Gamma(5.5,3.1) 
ref_estimate = S(PI,a_star,alpha_star,beta_star)
sb.distplot(ref_estimate,label="Reference (approx N(0,1))")
sb.distplot(PI,label="Target Gamma(5.5,3.1)")
plt.legend()
plt.title("Linear b1(w)")
#plt.savefig('gamma_linear.pdf')
plt.show()
#%%
# Check the estimated parameters of the reference - should be N(0,1)
is_norm = stats.norm.fit(ref_estimate)
#is_norm[0], is_norm[1]

#%% Plot the shape of S
xs = np.linspace(0,5,200)
ys = S(xs, a_star, alpha_star, beta_star)
plt.plot(xs,ys,label="Inverse transport")
plt.plot(ys,xs,label="Forward map")
plt.legend()
plt.title("Gamma target, linear b1(w)")
#plt.savefig('gamma_linear_map.pdf')
plt.show()

#%% Try other distributions as the target: Beta(1.5, 2.3)
B = np.random.beta(1.5, 2.3, size=10000)
param1 = np.array([0.1, 0.4, -0.1]) # This could be random?
res1 = minimize(obj, param1, args=(B))
a_star1 = res1.x[0]
alpha_star1 = res1.x[1]
beta_star1 = res1.x[2]
ref_estimate1 = S(B,a_star1,alpha_star1,beta_star1)
sb.distplot(ref_estimate1,label="Reference (approx N(0,1))")
sb.distplot(B,label="Target Beta(1.5,2.3)")
plt.legend()
plt.title("Linear b1(w)")
#plt.savefig('beta_linear.pdf')
plt.show()
is_norm1 = stats.norm.fit(ref_estimate1)
#is_norm1[0], is_norm1[1]

xs1 = np.linspace(0,1,200)
ys1 = S(xs, a_star1, alpha_star1, beta_star1)
plt.plot(xs1,ys1,label="Inverse transport")
plt.plot(ys1,xs1,label="Forward map")
plt.legend()
plt.title("Beta target, linear b1(w)")
#plt.savefig('beta_linear_map.pdf')
plt.show()

#%% Try b1(w) = alpha+beta*w+gamma*w^2 and see what happens - should be significantly slower
from scipy import integrate

def S_quadratic(z,a,alpha,beta,gamma):
    exp_quad = lambda w, alpha, beta, gamma : np.exp(alpha + beta*w + gamma*w**2)
    return [a + integrate.quad(exp_quad, 0, zz, args=(alpha,beta,gamma))[0] for zz in z]

#%%
def obj_quad(param,z):
    # z should be a 1d vector of inputs
    return np.mean(0.5*np.power(S_quadratic(z,param[0],param[1],param[2],param[3]),2) - (param[1] + param[2]*z + param[3]*z**2))
#%%
#PI = np.random.gamma(5.5, 1./3.1, size=1000)
# Starting values a0, alpha0, beta0, chose at random...
param2 = np.array([0.1, 0.4, -0.1, 0.2])
# Perform the minimisation
res_quad = minimize(obj_quad, param2, args=(PI))
#%%
#S_quadratic(np.array([0.2,0.4,0.5]),0.1,0.3,-0.1,0.2)

#obj_quad(param2,np.array([2.,5.]))

a_star2 = res_quad.x[0]
alpha_star2 = res_quad.x[1]
beta_star2 = res_quad.x[2]
gamma_star2 = res_quad.x[3]

ref_estimate2 = S_quadratic(PI,a_star2,alpha_star2,beta_star2,gamma_star2)
sb.distplot(ref_estimate2,label="Reference (approx N(0,1))")
sb.distplot(PI,label="Target Gamma(5.5,3.1)")
plt.legend()
plt.title("Quadratic b1(w)")
#plt.savefig('gamma_quadratic.pdf')
plt.show()
is_norm2 = stats.norm.fit(ref_estimate2)
#is_norm2[0], is_norm2[1]

xs2 = np.linspace(0,5,400)
ys2 = S_quadratic(xs2, a_star2, alpha_star2, beta_star2, gamma_star2)
plt.plot(xs2,ys2,label="Inverse transport")
plt.plot(ys2,xs2,label="Forward map")
plt.legend()
plt.title("Gamma target, quadratic b1(w)")
#plt.savefig('gamma_quadratic_map.pdf')
plt.show()

#%%
## Get the inverse map from a_star etc
#def S(z,a,alpha,beta):
#    return a + 1.0/beta * (np.exp(alpha + beta*z) - np.exp(alpha)) from before
def func_to_find_root_of(z,a,alpha,beta,x):
    return S(z,a,alpha,beta)-x 

from scipy.optimize import newton
#from scipy.optimize import bisect
from scipy.stats import describe

#%%
# Generate normals
#N = np.random.normal(size=1000)
Ns = np.random.normal(size=1000)
# This is the forward map!!
T_ns = np.array([newton(func_to_find_root_of,args=(a_star,alpha_star,beta_star,N),x0=0) for N in Ns])
# Try to fit the data... should be Gamma(5.5, 1/3.1)
fit_alpha, fit_loc, fit_beta = stats.gamma.fit(T_ns)
print(fit_alpha, fit_loc, fit_beta)

# Generate actual Gamma(5.5,3.1 data)
#PI = np.random.gamma(5.5, 1/3.1,1000)
# Compare the generated PI to the approximate map T_ns
sb.distplot(T_ns,label = 'Est..')
sb.distplot(PI,label='True')
plt.legend()
plt.show()

describe(T_ns)
describe(PI)

#%%

#%%
# Calculate KL divergence after estimating PDFs... for the reference we already have the density so can evaluate the density at the samples
# For the estimated samples, need to estimate PDF and then use that 
#S = entropy(T_ns, PI) # Doesn't work when there are samples which are negative
from scipy.stats import entropy
from sklearn.neighbors.kde import KernelDensity

T_ns_for_kde = T_ns[:,np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(T_ns_for_kde) # Fit the approximate forward map
#T_ns_density_est = np.exp(kde.score_samples(T_ns_for_kde))
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot[:, 0], np.exp(log_dens))

# Now calculate the KL divergence
#PI = stats.gamma.rvs(a=5.5,loc=0,scale=1./3.1,size=1000)
# Real target is gamma(a=5.5, scale=1./3.1)
# Evaluate this at the samples coming from the approximate map T_ns
PI_real = stats.gamma.pdf(T_ns,a=5.5,loc=0.,scale=1/3.1)
PI_est = np.exp(kde.score_samples(T_ns_for_kde))

# Now estimate the KL divergence between gamma real and gamma est
S = entropy(PI_est, PI_real)

#%% Measure the time for running the algorithm