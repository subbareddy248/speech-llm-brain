#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
#from utilsnew.utils import run_fmri_pair_subjects_crossval_ridge
from npp import zscore
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV


# make all possible combinations of size s from the listed subjects
def get_all_combinations(subjects, s):
    from itertools import combinations 
    comb = combinations(subjects, s) 
    return comb


import os
import utils1
fdir = './fdir/'

def load_subject_fMRI(modality, subject):
    fname_tr5 = os.path.join(fdir, 'subject{}_{}_fmri_data_trn.hdf'.format(subject, modality))
    trndata5 = utils1.load_data(fname_tr5)
    print(trndata5.keys())

    fname_te5 = os.path.join(fdir, 'subject{}_{}_fmri_data_val.hdf'.format(subject, modality))
    tstdata5 = utils1.load_data(fname_te5)
    print(tstdata5.keys())
    
    trim = 5
    zRresp = np.vstack([zscore(trndata5[story][5+trim:-trim-5]) for story in trndata5.keys()])
    zPresp = np.vstack([zscore(tstdata5[story][0][5+trim:-trim-5]) for story in tstdata5.keys()])
    
    return zRresp, zPresp

subs = ['01','02','03','05','07','08']

save_dir = 'predictions_results/noise_ceiling/'

def pearcorr(actual, predicted):
    corr = []
    for i in range(0, len(actual)):
        corr.append(np.corrcoef(actual[i],predicted[i])[0][1])
    return corr



def kernel_ridge(xtrain, xtest, ytrain, ytest):
    krr = KernelRidge()
    krr.fit(np.nan_to_num(xtrain), np.nan_to_num(ytrain))
    ypred = krr.predict(np.nan_to_num(xtest))
    corr = pearcorr(np.nan_to_num(ytest.T),np.nan_to_num(ypred.T))
    return corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    args = parser.parse_args()
    target_subject = args.subjectNum
    modality = args.modality
    if not os.path.exists(save_dir+modality):
        os.makedirs(save_dir+modality)
    source_subjects = [i for i in subs if i != target_subject]
    for source_subject in source_subjects:
        sourcedata_train, sourcedata_test = load_subject_fMRI(modality, source_subject)
        targetdata_train, targetdata_test = load_subject_fMRI(modality, target_subject)
        corrs_t = kernel_ridge(sourcedata_train, sourcedata_test, targetdata_train, targetdata_test)
        np.save(os.path.join(save_dir, "predict_{}_with_{}_{}pcs.npy".format(target_subject, source_subject, sourcedata_train.shape[1])),corrs_t)
