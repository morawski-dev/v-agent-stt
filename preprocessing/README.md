# Audio Preprocessor - User Guide

## Installation

### Requirements
- Python 3.7 or newer
- pip

### Install Dependencies

```bash
pip install librosa noisereduce soundfile scipy numpy
```

Or use the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Processing a Single File

```bash
# Basic usage
python audio_preprocessor.py -i recording.wav

# With output file specified
python audio_preprocessor.py -i recording.wav -o clean_recording.wav

# With custom sample rate
python audio_preprocessor.py -i recording.wav -sr 22050
```

### 2. Processing Multiple Files

```bash
# Process all files in directory
python audio_preprocessor.py -d ./recordings

# With output directory specified
python audio_preprocessor.py -d ./recordings -od ./processed
```

### 3. Customizing Preprocessing

```bash
# Disable noise reduction
python audio_preprocessor.py -i recording.wav --no-noise-reduce

# Disable normalization and speech enhancement
python audio_preprocessor.py -i recording.wav --no-normalize --no-enhance

# Disable silence trimming
python audio_preprocessor.py -i recording.wav --no-trim

# Disable high-pass filter
python audio_preprocessor.py -i recording.wav --no-highpass
```

## Preprocessing Features

The script performs the following operations:

1. **Trim Silence** - Removes silence from the beginning and end of recording
2. **High-pass Filter** - High-pass filter (removes low frequencies <80Hz)
3. **Noise Reduction** - Background noise reduction
4. **Speech Enhancement** - Enhances speech frequency range (300-3400Hz)
5. **Normalization** - Volume normalization to -20dB

## Supported Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)

## Advanced Parameters

### Noise Reduction
```python
audio = preprocessor.reduce_noise(
    audio, 
    sr, 
    stationary=True,      # Whether noise is stationary
    prop_decrease=1.0     # Reduction level (0.0-1.0)
)
```

### Normalization
```python
audio = preprocessor.normalize_audio(
    audio,
    target_level=-20.0    # Target level in dB
)
```

### High-pass Filter
```python
audio = preprocessor.apply_highpass_filter(
    audio,
    sr,
    cutoff=80.0,         # Cutoff frequency (Hz)
    order=5              # Filter order
)
```

### Band-pass Filter
```python
audio = preprocessor.apply_bandpass_filter(
    audio,
    sr,
    low_cutoff=300.0,    # Lower frequency
    high_cutoff=3400.0,  # Upper frequency
    order=5
)
```

### Silence Trimming
```python
audio = preprocessor.trim_silence(
    audio,
    sr,
    top_db=30,           # Silence threshold in dB
    frame_length=2048,
    hop_length=512
)
```

## Troubleshooting

### Issue: Slow processing
- Reduce sample rate: `-sr 8000`
- Disable some features: `--no-enhance --no-trim`

## Logs

All operations are logged to:
- Console (stdout)
- File: `audio_preprocessing.log`

## Tips

1. **For speech recognition**: Keep all options enabled, set `-sr 16000`
