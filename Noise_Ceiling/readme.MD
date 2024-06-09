## Estimate cross-subject prediction accuracy ##
- To estimate cross-subject prediction accuracy for each brain dataset, we need to perform two steps:
1. After subsampling of all the participants, predict target participant using each source participant
```
python brain_predictions-noise-ceiling 1 reading
```
2. Estimate cross-subject prediction accuracy for each voxel 
```
python Noise_Ceiling_Fatma.py 1 reading
```

