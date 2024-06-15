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
from npp import zscore
import os
import utils1
from npp import zscore
from ridge_utils.ridge import bootstrap_ridge

logging.basicConfig(level=logging.DEBUG)

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
    zPresp = np.vstack([zscore(tstdata5[story][1][5+trim:-trim-5]) for story in tstdata5.keys()])
    
    return zRresp, zPresp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("featurename", help="Choose feature", type = str)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("dirname", help="Choose Directory", type = str)
    parser.add_argument("layers", help="Choose layers", type = int)
    args = parser.parse_args()

    stimul_features = np.load(args.featurename, allow_pickle=True)
    print(stimul_features.item().keys())

    Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                'life', 'myfirstdaywiththeyankees', 'naked',
                'odetostepfather', 'souls', 'undertheinfluence']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']
    num_layers = args.layers

    allstories = Rstories + Pstories

    grids = load_grids_for_stories(allstories)

    # Load TRfiles
    trfiles = load_generic_trfiles(allstories)

    # Make word and phoneme datasequences
    wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
    phonseqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}

    eng1000 = SemanticModel.load("./english1000sm.hf5")
    semanticseqs = dict() # dictionary to hold projected stimuli {story name : projected DataSequence}
    for story in allstories:
        semanticseqs[story] = make_semantic_model(wordseqs[story], eng1000)

    storie_filenames = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                'life', 'myfirstdaywiththeyankees', 'naked',
                'odetostepfather', 'souls', 'undertheinfluence','wheretheressmoke'] 
    semanticseqs = dict()
    for i in np.arange(len(allstories)):
        print(allstories[i])
        semanticseqs[allstories[i]] = []
        for eachlayer in np.arange(num_layers):
            temp = make_semantic_model(wordseqs[allstories[i]], eng1000)
            temp.data = np.nan_to_num(stimul_features.item()[storie_filenames[i]][eachlayer])
            semanticseqs[allstories[i]].append(temp)

    # Downsample stimuli
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter
    #num_layers = 12
    downsampled_semanticseqs = dict() # dictionary to hold downsampled stimuli
    for story in allstories:
        downsampled_semanticseqs[story] = []
        for eachlayer in np.arange(num_layers):
            temp = semanticseqs[story][eachlayer].chunksums(interptype, window=window)
            downsampled_semanticseqs[story].append(temp)

    trim = 5
    Rstim = {}
    Pstim = {}
    for eachlayer in np.arange(num_layers):
        Rstim[eachlayer] = []
        Rstim[eachlayer].append(np.vstack([zscore(downsampled_semanticseqs[story][eachlayer][5+trim:-trim]) for story in Rstories]))

    for eachlayer in np.arange(num_layers):
        Pstim[eachlayer] = []
        Pstim[eachlayer].append(np.vstack([zscore(downsampled_semanticseqs[story][eachlayer][5+trim:-trim]) for story in Pstories]))
    storylens = [len(downsampled_semanticseqs[story][0][5+trim:-trim]) for story in Rstories]
    print(storylens)

    # Delay stimuli
    from util import make_delayed
    ndelays = 6
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

    subject = '0'+str(args.subjectNum)
    # Run regression
    nboots = 1 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    main_dir = args.dirname+'/'+args.modality+'/'+subject
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    for eachlayer in np.arange(num_layers):
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
        #np.save(os.path.join(main_dir+'/'+save_dir, "test_"+str(eachlayer)),zPresp)
        #np.save(os.path.join(main_dir+'/'+save_dir, "pred_"+str(eachlayer)),pred)
        voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
        print (voxcorrs)

        np.save(os.path.join(main_dir+'/'+save_dir, "layer_"+str(eachlayer)),voxcorrs)