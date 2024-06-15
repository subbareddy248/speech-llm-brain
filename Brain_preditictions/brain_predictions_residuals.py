#!/usr/bin/env python

# This cell imports libraries that you will need
# Run this.
from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
import os
import utils1
from npp import zscore
from ridge_utils.ridge import bootstrap_ridge
from residuals_text_speech import *
logging.basicConfig(level=logging.DEBUG)



#These files contains low-level textual and speech features
def load_low_level_textual_features():
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File('../features_trn_NEW.hdf','r+')
    base_features_val = h5py.File('../features_val_NEW.hdf','r+')
    return base_features_train, base_features_val

def load_low_level_speech_features(lowlevelfeature):
    # 'diphone', 'powspec', 'triphone'
    if lowlevelfeature in ['diphone', 'powspec', 'triphone']:
        df = h5py.File('./features_matrix.hdf')
        base_features_train = df[lowlevelfeature+'_train']
        base_features_val = df[lowlevelfeature+'_test']
    elif lowlevelfeature in 'articulation':
        base_features_train = np.load('./articulation_train.npy')
        base_features_val = np.load('./articulation_test.npy')
    return base_features_train, base_features_val

def load_low_level_visual_features():
    stimulus_data_file = np.load('m_ll.npz', allow_pickle=True)
    stimulus_data_file = {key: stimulus_data_file[key].item() for key in stimulus_data_file}
    test_matrix = stimulus_data_file['test']['7']  #  (291, 6555) matrix of (TRs, feature_dims) for test story
    train_matrix = stimulus_data_file['train']['7']  # (3737, 6555) matrix of (TRs, feature_dims) for train stories (train stories ordered in alphabetical order)
    return train_matrix, test_matrix


trim = 5
fdir = './fdir/'
def load_subject_fMRI(subject, modality):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("featurename", help="Choose feature", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("dirname", help="Choose Directory", type = str)
    parser.add_argument("layernum", help="Choose Layer Num", type = int)
    parser.add_argument("lowlevelfeature", help="Choose low-level feature name", type = str)
    args = parser.parse_args()

    stimulus_features = np.load(args.featurename, allow_pickle=True) # This file contains already downsampled data

    if args.lowlevelfeature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
        base_features_train, base_features_val = load_low_level_textual_features()
        residual_features = residuals_textual(base_features_train, base_features_val, stimulus_features, args.lowlevelfeature)
    elif args.lowlevelfeature in ['powspec', 'diphone', 'triphone','articulation']:
        base_features_train, base_features_val = load_low_level_speech_features(args.lowlevelfeature)
        residual_features = residuals_phones(base_features_train, base_features_val, stimulus_features, args.lowlevelfeature)
    elif args.lowlevelfeature in ['motion']:
        base_features_train, base_features_val = load_low_level_visual_features()
        residual_features = residuals_visual(base_features_train, base_features_val, stimulus_features, args.lowlevelfeature)

    # Delay stimuli
    from util import make_delayed
    ndelays = 6
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)

    delRstim = []
    for eachlayer in np.arange(12):
        delRstim.append(make_delayed(np.array(residual_features.item()[args.lowlevelfeature][0][eachlayer]), delays))
        
    delPstim = []
    for eachlayer in np.arange(12):
        delPstim.append(make_delayed(np.array(residual_features.item()[args.lowlevelfeature][1][eachlayer]), delays))

    # Print the sizes of these matrices
    print ("delRstim shape: ", delRstim[0].shape)
    print ("delPstim shape: ", delPstim[0].shape)

    subject = '0'+str(args.subjectNum)

    nboots = 5 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    main_dir = args.dirname+'/'+args.modality+'/'+subject
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    for eachlayer in np.arange(args.layernum,12):
        zRresp, zPresp = load_subject_fMRI(subject, args.modality)
        alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
        all_corrs = []
        save_dir = str(eachlayer)
        if not os.path.exists(main_dir+'/'+save_dir):
            os.mkdir(main_dir+'/'+save_dir)
        wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(delRstim[eachlayer]), zRresp, np.nan_to_num(delPstim[eachlayer]), zPresp,
                                                             alphas, nboots, chunklen, nchunks,
                                                             singcutoff=1e-10, single_alpha=True)
        pred = np.dot(np.nan_to_num(delPstim[eachlayer]), wt)

        print ("pred has shape: ", pred.shape)
        voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
        print (voxcorrs)

        np.save(os.path.join(main_dir+'/'+save_dir, "layer_"+str(eachlayer)),voxcorrs)