#!/usr/bin/env python

# This cell imports libraries that you will need
# Run this.
from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
from stimulus_utils import load_grids_for_stories, load_generic_trfiles
from dsutils import make_word_ds, make_phoneme_ds
from dsutils import make_semantic_model
from SemanticModel import SemanticModel
from pathlib import Path
from interpdata import lanczosinterp2D
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("featurename", help="Choose feature", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("dirname", help="Choose Directory", type = str)
    parser.add_argument("numlayers", help="Number of Layers", type = int)
    args = parser.parse_args()

    chunk_sz, context_sz = 0.1, 32.0
    model = args.featurename

    base_features_path = Path(f"encoding-model-scaling-laws/features_cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}/{model}")
    num_layers = args.numlayers

    Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                'life', 'myfirstdaywiththeyankees', 'naked',
                'odetostepfather', 'souls', 'undertheinfluence']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']

    allstories = Rstories + Pstories

    grids = load_grids_for_stories(allstories)

    # Load TRfiles
    trfiles = load_generic_trfiles(allstories)

    # Make word and phoneme datasequences
    wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
    phonseqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}


    from npp import zscore
    trim = 5
    Rstim = {}
    Pstim = {}
    for eachlayer in np.arange(num_layers):
        Rstim[eachlayer] = []
        downsampled_semanticseqs = dict() # dictionary to hold downsampled stimuli
        for story in allstories:
            downsampled_semanticseqs[story] = []
            times = np.load(base_features_path / f"{story}_times.npz")['times'][:,1] # shape: (time,)
            features = np.load(base_features_path / f"layer.{eachlayer}" / f"{story}.npz")['features'] # shape: (time, model dim.)
            # you will need `wordseqs` from the notebook
            downsampled_features = lanczosinterp2D(features, times, wordseqs[story].tr_times)
            downsampled_semanticseqs[story].append(downsampled_features)
        Rstim[eachlayer].append(np.vstack([zscore(downsampled_semanticseqs[story][0][5+trim:-trim]) for story in Rstories]))

        Pstim[eachlayer] = []
        Pstim[eachlayer].append(np.vstack([zscore(downsampled_semanticseqs[story][0][5+trim:-trim]) for story in Pstories]))

    # Delay stimuli
    from util import make_delayed
    ndelays = 8
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)
    print(np.array(Rstim[0]).shape)
    delRstim = []
    for eachlayer in np.arange(num_layers):
        delRstim.append(make_delayed(np.array(Rstim[eachlayer])[0], delays))
        
    delPstim = []
    for eachlayer in np.arange(num_layers):
        delPstim.append(make_delayed(np.array(Pstim[eachlayer])[0], delays))


    # Print the sizes of these matrices
    print ("delRstim shape: ", delRstim[0].shape)
    print ("delPstim shape: ", delPstim[0].shape)


    import os
    import utils1
    from npp import zscore
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

    subject = '0'+str(args.subjectNum)
    # Run regression
    from ridge_utils.ridge import bootstrap_ridge

    nboots = 5 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    main_dir = args.dirname+'/'+args.modality+'/'+subject
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    for eachlayer in np.arange(5,num_layers):
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
        np.save(os.path.join(main_dir+'/'+save_dir, "test_"+str(eachlayer)),zPresp)
        np.save(os.path.join(main_dir+'/'+save_dir, "pred_"+str(eachlayer)),pred)
        print ("pred has shape: ", pred.shape)
        voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
        print (voxcorrs)

        np.save(os.path.join(main_dir+'/'+save_dir, "layer_"+str(eachlayer)),voxcorrs)
