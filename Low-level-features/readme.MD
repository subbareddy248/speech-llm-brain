**How to load low-level stimuli features**
- 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'

#low-level textual features
```
def load_low_level_textual_features():
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File('../features_trn_NEW.hdf','r+')
    base_features_val = h5py.File('../features_val_NEW.hdf','r+')
    return base_features_train, base_features_val
```

#low-level speech features
```
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
```

#low-level speech features
```
def load_low_level_visual_features():
    stimulus_data_file = np.load('m_ll.npz', allow_pickle=True)
    stimulus_data_file = {key: stimulus_data_file[key].item() for key in stimulus_data_file}
    test_matrix = stimulus_data_file['test']['7']  #  (291, 6555) matrix of (TRs, feature_dims) for test story
    train_matrix = stimulus_data_file['train']['7']  # (3737, 6555) matrix of (TRs, feature_dims) for train stories (train stories ordered in alphabetical order)
    return train_matrix, test_matrix
```
