**How to download 21<sup>st</sup> year dataset**

- For Whisper-small model
```bash
python extract_speech_features.py --stimulus_dir stories_wav --model whisper-small --chunksz 100 --contextsz 16000 --use_featext --batchsz 64
```
- For Wav2vec2.0-base model
```bash
python extract_speech_features.py --stimulus_dir stories_wav --model wav2vec2-base --chunksz 100 --contextsz 16000 --use_featext --batchsz 64
```
