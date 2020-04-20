# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 00:03:19 2019

@author: Gavin

This script shows analytical maps between the
standard normal and Gamma distributions for certain
parameters

ref refers to reference
tar refers to target
"""
#%%
import numpy as np
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

#%% These functions define the forward and inverse
# maps for ref N(0,1) and tar Gamma(alpha,beta)
def analytic_map(samples, rate):
    """ Maps N(0,1) to Gamma(0.5, 1.0/k) """
    return samples**2/rate

def inverse_analytic_map(samples, rate):
    """ Maps Gamma(0.5, 1.0/k) to N(0,1) """
    return np.sqrt(samples*rate)

#%% This is the forward map for Gamma(alpha,beta)
# with alpha = 0.5, beta = k/2
N = 2000
k = 0.4

ref_samples = np.random.normal(size=(N))
tar_samples = analytic_map(ref_samples, k)

x_plot = np.linspace(min(ref_samples), max(ref_samples), N)
plt.plot([], [], ' ', label=r'$N = {:d}$'.format(N))
plt.plot(x_plot, st.norm.pdf(x_plot), label='Reference Normal density')
sb.distplot(tar_samples, label='Target Gamma samples', kde=False, norm_hist=True)
plt.legend()
plt.title(r'Forward map')
plt.xlabel(r'$x$')
plt.xlim([min(ref_samples), 20])
#plt.savefig('analytical_forward.pdf')
plt.show()

#%% This is the inverse map
NN = 2000
kk = 2.0

target_samples = np.random.gamma(shape=0.5, scale=1./kk, size=NN)
# Will be distribution of |Z| not Z~N(0,1) since not bijective
reference_samples = inverse_analytic_map(target_samples, kk)

x_plot = np.linspace(min(reference_samples)+0.03, max(reference_samples), N)
plt.plot([], [], ' ', label=r'$N = {:d}$'.format(N))
plt.plot(x_plot, st.gamma.pdf(x_plot, a=0.5, scale=1./kk), label='Reference Gamma density')
#sb.kdeplot(reference_samples, label='Gamma refrence', clip=(0.0, 5))
sb.distplot(target_samples, label=r'Target Normal samples', kde=False, norm_hist=True)

plt.legend()
plt.xlabel(r'$x$')
plt.title(r'Inverse map')
plt.xlim([-0.1, 2])
plt.savefig('analytical_inverse_check.pdf')
plt.show()
