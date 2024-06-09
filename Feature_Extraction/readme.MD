**How to extract speech-based model features**

- stimulus_dir: directory consists of .wav files
- model name: refer json file model name
- chunk size: stride for window (10, 100 ms)
- context size: window length (16 secs, 32 secs, 64 secs)

*For Whisper-small model*
```bash
python extract_speech_features.py --stimulus_dir stories_wav --model whisper-small --chunksz 100 --contextsz 16000 --use_featext --batchsz 64
```
*For Wav2vec2.0-base model*
```bash
python extract_speech_features.py --stimulus_dir stories_wav --model wav2vec2-base --chunksz 100 --contextsz 16000 --use_featext --batchsz 64
```
