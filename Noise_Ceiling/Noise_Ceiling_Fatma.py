#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse


# make all possible combinations of size s from the listed subjects
def get_all_combinations(subjects, s):
    from itertools import combinations 
    comb = combinations(subjects, s) 
    return comb


# get average correlation of predicting the specified target_subject from each source subject (return mean over source subjects)
def get_mean_prediction(target_subject, source_subjects, voxels_dict={}, modality):
    # load all predictions from source to target
    corrs = []
    for source_subject in source_subjects:
        loaded = np.load('./predictions_results/noise_ceiling/'+modality+'/predict_{}_with_{}_{}pcs.npy'.format(target_subject, source_subject, voxels_dict[source_subject]) ,allow_pickle=True)
        corrs.append(np.nan_to_num(loaded))
    mean_subsample_corrs = np.mean(corrs,0)
    
    return mean_subsample_corrs


def exponenial_func(x, v0, t0):
    return v0*(1-np.exp(-x/t0))

def bootstrap_function(x,y):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(exponenial_func, x, y,  maxfev = 6000) #, p0=(1, 1))
    return popt[0]

def bootstrap(samples1, samples2, n_bootstrap=100):
    n_samples = len(samples1)
    
    bootstraps = []
    for b in range(n_bootstrap):
        inds_choice = np.random.choice(range(n_samples),n_samples,replace=True)
        
        metric = bootstrap_function(samples1[inds_choice], samples2[inds_choice])
        bootstraps.append(metric)
        
    return bootstraps


subs = ['02','03','05','07','08','01']
n_voxels = [80350, 72965, 80615, 92970, 79936, 81133]
n_voxels_dict = {}
for s, sub in enumerate(subs):
    n_voxels_dict[sub] = n_voxels[s]

np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    args = parser.parse_args()
    target_subject = args.subjectNum
    modality = args.modality
    source_subjects = [i for i in subs if i != target_subject]
    
    subsample_corrs = []
    subsample_sizes = []
    
    for subsample_size in range(2, len(subs)+1):
        subsamples = get_all_combinations(source_subjects, subsample_size-1) # subtract 1 because we're only doing combs of source subjs

        for subsample in list(subsamples):
            mean_subsample_corrs = get_mean_prediction(target_subject, list(subsample), n_voxels_dict, modality)
            subsample_corrs.append(mean_subsample_corrs)
            subsample_sizes.append(subsample_size)
            
        print('done with subsample size {}'.format(subsample_size))
        
    ceilings = []
    for pc_ind in range(n_voxels_dict[target_subject]):
        bootstraps = bootstrap(np.array(subsample_sizes), np.array(subsample_corrs)[:,pc_ind])
        ceilings.append(np.median(bootstraps))
        
    np.save('predictions_results/noise_ceiling/'+modality+'/subject_{}_kernel_ridge.npy'.format(target_subject),np.array(ceilings))
    
    print('done with subject {}'.format(target_subject))
