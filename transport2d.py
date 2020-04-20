# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:46:49 2019

@author: Gavin
"""

#%% The banana
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
Z = np.random.normal(size=(500,2))
T = np.zeros((500,2))
T[:,0] = Z[:,0]
T[:,1] = np.cos(1.5*Z[:,0]) + 0.5*Z[:,1]

#sb.jointplot(Z[:,0],Z[:,1],kind="kde")
sb.jointplot(T[:,0], T[:,1], kind="kde")
