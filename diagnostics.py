# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:28:40 2019

@author: Gavin
"""

#%% Convergence and diagnostics
import scipy.stats as st
import numpy as np
from scipy.stats import shapiro
from scipy.stats import anderson
from transportQuad_ND import *

#%% Gumbel distribution (known PDF)
def gumbel_D_KL(M):    
#    M = np.array([50, 100])
    Samps = []
    Samps = [st.gumbel_r.rvs(size=m,scale=.5,loc=-2.5) for m in  M] 
    
    Params = []
    Params.append(inv_trans_quad(Samps[0],1))
    for i in range(1,len(Samps)):
        Params.append(inv_trans_quad(Samps[i],1,Params[i-1]))
        
    #Params = [inv_trans_quad(samps,1) for samps in Samps]
    
    detGradSs = []
    detGradSs = [detGradS(samps, params) for samps, params in zip(Samps,Params)]
    
    D_KL = []
    D_KL = np.array([0.5*np.var(np.log(st.gumbel_r.pdf(samps,scale=.5,loc=-2.5)) - np.log(st.norm.pdf(eval_S_quad(samps,1,params)) * np.abs(detGrads)))  for samps, params, detGrads in zip(Samps, Params, detGradSs)])
    return D_KL
#
#%%
M = np.array([50, 100])
D_KL = np.zeros((4,2))
for k in range(4):
    D_KL[k,:] = gumbel_D_KL(M);



plt.errorbar(M,np.mean(D_KL,axis=0),np.std(D_KL,axis=0), fmt='-o',capsize=3)
plt.xlabel("Samples")
plt.ylabel("Estimated KL distance")
plt.tight_layout()
plt.savefig("gumbel_KL_20rds_avged.pdf")


#%%
Params, _ = gumbel_D_KL()
New_Gumbels = [st.gumbel_r.rvs(size=m,scale=.5,loc=-2.5) for m in M]
Approx_Gumbel_refs = [eval_S_quad(g,1,params) for g, params in zip(New_Gumbels,Params)]

p_vals_shapiro_gum = np.zeros((len(M),1))
for i in range(len(M)):
    _, p_vals_shapiro_gum[i] = shapiro(Approx_Gumbel_refs[i])

acpt_reject_gum = np.zeros((len(M),1))
for i in range(len(M)):
#    for j in range(2):
    a = anderson(Approx_Gumbel_refs[i])
    if (a.statistic < a.critical_values[2]):
        acpt_reject_gum[i] = 0 # Normal
    else:
        acpt_reject_gum[i] = 1 # Not normal

#%% Banana (2D) test both dimensions

Bananas = []
Bananas = [banana(m) for m in M]

BananaParams = [];
BananaParams.append(inv_trans_quad(Bananas[0],2))
for i in range(1,len(Bananas)):
    BananaParams.append(inv_trans_quad(Bananas[i],2))

NewBananas = [banana(m) for m in M]
ApproxRefs = [eval_S_quad(banana,2,params) for banana, params in zip(NewBananas, BananaParams)]
M=2000
p_vals_shapiro = np.zeros((1,2))
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

        
#%%

        
        
        