#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:33:51 2019

@author: jmadriga
"""
#https://scikit-learn.org/stable/modules/clustering.html#k-means
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from transportQuad_ND import *
from statsmodels.graphics.tsaplots import plot_acf



def obtain_clusters_and_labels(x,plot=False):

    
    N=np.size(x,0)

    
    
    # Compute clustering with MeanShift
    
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=int(0.2*N))
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    
    
    # computes the proportion of samples on each cluster
    
    prop_labels=np.zeros(n_clusters_)
    for i in range(n_clusters_):
        prop_labels[i]=np.sum(labels==labels_unique[i])/N
        
        if(plot==True):
            plt.plot(x[labels==labels_unique[i],0],x[labels==labels_unique[i],1],'.')    
    if(plot==True):
        plt.show()
    
    
    return labels,prop_labels
