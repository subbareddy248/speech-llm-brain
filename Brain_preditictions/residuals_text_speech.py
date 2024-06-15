#!/usr/bin/env python
# coding: utf-8

import nltk
import numpy as np
from scipy.io import loadmat
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, accuracy_score
from scipy import stats
from sklearn.decomposition import PCA
import pickle
import os
from joblib import Parallel, delayed
import h5py
from npp import zscore


# Functions to estimate cost for each lambda, by voxel:
from __future__ import division                                              

from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time 
from scipy.stats import zscore
from sklearn.kernel_ridge import KernelRidge

def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def R2r(Pred,Real):
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ridge_sk(X,Y,lmbda):
    rd = Ridge(alpha = lmbda)
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas,solver = 'svd')
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge(X,Y,lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def cross_val_ridge(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
    
    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method]
    
    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
#         print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
#         if icv%3 ==0:
#             print(icv)
#         print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((train_features.shape[1],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])

def GCV_ridge(train_features,train_data,lambdas = np.array([10**i for i in range(-6,10)])):
    
    n_lambdas = lambdas.shape[0]
    n_voxels = train_data.shape[1]
    n_time = train_data.shape[0]
    n_p = train_features.shape[1]

    CVerr = np.zeros((n_lambdas, n_voxels))

    # % If we do an eigendecomp first we can quickly compute the inverse for many different values
    # % of lambda. SVD uses X = UDV' form.
    # % First compute K0 = (X'X + lambda*I) where lambda = 0.
    #K0 = np.dot(train_features,train_features.T)
    print('Running svd',)
    start_time = time.time()
    [U,D,Vt] = svd(train_features,full_matrices=False)
    V = Vt.T
    print(U.shape,D.shape,Vt.shape)
    print('svd time: {}'.format(time.time() - start_time))

    for i,regularizationParam in enumerate(lambdas):
        regularizationParam = lambdas[i]
        print('CVLoop: Testing regularization param: {}'.format(regularizationParam))

        #Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
        dlambda = D**2 + np.eye(n_p)*regularizationParam
        dlambdaInv = np.diag(D / np.diag(dlambda))
        KlambdaInv = V.dot(dlambdaInv).dot(U.T)
        
        # Compute S matrix of Hastie Trick  H = X(XT X + lambdaI)-1XT
        S = np.dot(U, np.diag(D * np.diag(dlambdaInv))).dot(U.T)
        denum = 1-np.trace(S)/n_time
        
        # Solve for weight matrix so we can compute residual
        weightMatrix = KlambdaInv.dot(train_data);


#         Snorm = np.tile(1 - np.diag(S) , (n_voxels, 1)).T
        YdiffMat = (train_data - (train_features.dot(weightMatrix)));
        YdiffMat = YdiffMat / denum;
        CVerr[i,:] = (1/n_time)*np.sum(YdiffMat * YdiffMat,0);


    # try using min of avg err
    minerrIndex = np.argmin(CVerr,axis = 0);
    r=np.zeros((n_voxels));

    for nPar,regularizationParam in enumerate(lambdas):
        ind = np.where(minerrIndex==nPar)[0];
        if len(ind)>0:
            r[ind] = regularizationParam;
            print('{}% of outputs with regularization param: {}'.format(int(len(ind)/n_voxels*100),
                                                                        regularizationParam))
            # got good param, now obtain weights
            dlambda = D**2 + np.eye(n_p)*regularizationParam
            dlambdaInv = np.diag(D / np.diag(dlambda))
            KlambdaInv = V.dot(dlambdaInv).dot(U.T)

            weightMatrix[:,ind] = KlambdaInv.dot(train_data[:,ind]);


    return weightMatrix, r

def kernel_ridge(xtrain, xtest, ytrain, ytest):
    krr = KernelRidge()
    krr.fit(np.nan_to_num(xtrain), np.nan_to_num(ytrain))
    ypred1 = krr.predict(np.nan_to_num(xtest))
    ypred2 = krr.predict(np.nan_to_num(xtrain))
    return ypred2, ypred1

def sub_to_subjectnum(sub):
    for i in range(len(all_subjects)):
        if all_subjects[i] == sub:
            return i+1
    return 0

def residuals_textual(source_train, source_val, target_data, eachfeature): 
    trim = 5 
    np.random.seed(9)
    
    sourcedata_train = np.vstack([zscore(source_train[story][eachfeature][5+trim:-trim]) for story in base_features_train.keys()])
    sourcedata_test = np.vstack([zscore(source_val[story][eachfeature][5+trim:-trim]) for story in base_features_val.keys()])
    
    final_residuals = {}
    all_train = []
    all_test = []
    for eachlayer in np.arange(12):

        split_num = 0
        #for BERT
        y_train = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[:-1]])
        y_test = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[-1:]])

        weights, lbda = cross_val_ridge(sourcedata_train,y_train)        
        y_pred = np.dot(sourcedata_test,weights)

        all_test.append(y_test - y_pred)
        
        y_pred = np.dot(sourcedata_train,weights)
        all_train.append(y_train - y_pred)
    final_residuals[eachfeature] = []
    final_residuals[eachfeature].append(np.array(all_train))
    final_residuals[eachfeature].append(np.array(all_test))
    return final_residuals

def residuals_phones(source_train, source_val, target_data, eachfeature):
    trim = 5  
    np.random.seed(9)
    
    sourcedata_train = np.vstack([zscore(source_train)])
    sourcedata_test = np.vstack([zscore(source_val)])
    
    final_residuals = {}
    all_train = []
    all_test = []
    for eachlayer in np.arange(12):

        split_num = 0
        #for BERT
        y_train = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[:-1]])
        y_test = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[-1:]])

        weights, lbda = cross_val_ridge(np.nan_to_num(sourcedata_train),np.nan_to_num(y_train))        
        y_pred = np.dot(np.nan_to_num(sourcedata_test),weights)

        all_test.append(np.nan_to_num(y_test) - y_pred)
        
        y_pred = np.dot(np.nan_to_num(sourcedata_train),weights)
        all_train.append(np.nan_to_num(y_train) - y_pred)
    final_residuals[eachfeature] = []
    final_residuals[eachfeature].append(np.array(all_train))
    final_residuals[eachfeature].append(np.array(all_test))
    return final_residuals

def residuals_visual(source_train, source_val, target_data, eachfeature):
    trim = 5  
    np.random.seed(9)
    
    sourcedata_train = np.vstack([zscore(source_train)])
    sourcedata_test = np.vstack([zscore(source_val)])
    print(sourcedata_train.shape)
    print(sourcedata_test.shape)
    
    final_residuals = {}
    all_train = []
    all_test = []
    for eachlayer in np.arange(12):

        split_num = 0
        #for BERT
        y_train = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[:-1]])
        y_test = np.vstack([zscore(target_data.item()[story][eachlayer][5+trim:-trim]) for story in list(target_data.item().keys())[-1:]])
            
        print(y_train.shape)
        print(y_test.shape)

        #weights, lbda = cross_val_ridge(np.nan_to_num(sourcedata_train),np.nan_to_num(y_train))        
        y_pred1, y_pred2 = kernel_ridge(np.nan_to_num(sourcedata_train), np.nan_to_num(sourcedata_test), y_train, y_test)

        all_test.append(np.nan_to_num(y_test) - y_pred2)
        
        #y_pred = np.dot(np.nan_to_num(sourcedata_train),weights)
        all_train.append(np.nan_to_num(y_train) - y_pred1)
    final_residuals[eachfeature] = []
    final_residuals[eachfeature].append(np.array(all_train))
    final_residuals[eachfeature].append(np.array(all_test))
    return final_residuals