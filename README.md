# Speech language models lack important brain relevant semantics

[Speech language models lack important brain relevant semantics](https://arxiv.org/pdf/2311.04664), Subba Reddy Oota, Emin Çelik, Fatma Deniz and Mariya Toneva, ACL-2024

![screenshot](speechlm.PNG)

**Subset-Moth-Radio-Hour dataset statistics:**

Link: https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri
- 6 subjects
- fMRI brain recordings
- 11 stories (10 stories for training + 1 story for testing)
- TR = 2.0045 secs
- subjects: ['01', '02', '03', '05', '07', '08']

**Predict brain recordings using text-based language model representations (Reading and Listening):**

- sub_number [1, 2, 3, 5, 7, 8]
- stimulus vector [bert, gpt-2, FLAN]
- modality [reading, listening]
- output_dir
- #num of layers [12, 24]
```
python brain_predictions_subset_mothradio.py 1 bert-subset-moth-radio.npy reading bert-predictions 12
python brain_predictions_subset_mothradio.py 1 bert-subset-moth-radio.npy listening bert-predictions 12
```
