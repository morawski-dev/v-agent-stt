# v-agent-stt

A proof-of-concept project for Automatic Speech Recognition (ASR) quality evaluation. Uses OpenAI's Whisper Large V3 model for transcription on AWS EC2 GPU instances, with utilities for audio preprocessing and Word Error Rate (WER) calculation.

## Features

- Audio preprocessing pipeline (noise reduction, normalization, silence trimming)
- GPU-accelerated transcription using Whisper Large V3
- LLM-based post-processing for domain-specific error correction (Polish banking)
- WER calculation with detailed error breakdown
- Batch processing support
- Multi-language support (Polish, English, Ukrainian)

## Project Structure

```
v-agent-stt/
├── preprocessing/     # Audio cleaning and preparation
├── whisper-ec2/       # AWS EC2 transcription scripts
├── postprocessing/    # LLM-based error correction
├── benchmark/         # WER calculation tools
├── transcription/     # Ground truth guidelines
└── recordings/        # Audio data storage
```

## Requirements

- Python 3.7+
- ffmpeg
- CUDA 12.1+ (for GPU acceleration)

### Python Dependencies

```
librosa
soundfile
noisereduce
scipy
numpy
torch
transformers
accelerate
openai
jiwer
```

## Usage

### 1. Audio Preprocessing

Prepare audio files for optimal transcription quality:

```bash
# Single file
python preprocessing/audio_preprocessor.py -i input.wav -o output.wav

# Batch processing
python preprocessing/audio_preprocessor.py -d ./recordings -od ./processed
```

Available options:
- `--no-noise-reduce` - Disable noise reduction
- `--no-normalize` - Disable volume normalization
- `--no-enhance` - Disable speech enhancement
- `--no-trim` - Disable silence trimming
- `--no-highpass` - Disable high-pass filter
- `-sr, --sample-rate` - Custom sample rate (default: 16000)

### 2. Transcription

Run transcription on AWS EC2 GPU instance:

```bash
# Single file
python3 whisper-ec2/transcribe.py audio.wav pl

# Batch processing
python3 whisper-ec2/batch_transcribe.py ./audio_files/ pl

# Verify environment
python3 whisper-ec2/quick_test.py
```

Supported languages: `pl` (Polish), `en` (English), `uk` (Ukrainian)

### 3. LLM Post-Processing

Correct ASR errors using OpenAI GPT (Polish banking domain):

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Single file
python postprocessing/llm_postprocessor.py -i recording_transcription.txt

# Batch processing
python postprocessing/llm_postprocessor.py -d ./transcriptions -od ./corrected

# Preview without saving
python postprocessing/llm_postprocessor.py -i recording_transcription.txt --dry-run
```

Example corrections:
- "kredyt potoczny" -> "kredyt hipoteczny"
- "karta kretowa" -> "karta kredytowa"
- "erso" -> "RRSO"

### 4. WER Evaluation

Calculate Word Error Rate between reference and hypothesis:

```bash
python benchmark/wer_calculator.py
```

Output includes:
- WER score (0.0 = perfect, 1.0 = completely wrong)
- Substitutions, deletions, insertions, and hits breakdown

## AWS EC2 Setup

Recommended configuration:
- Instance type: g5.xlarge (4 vCPU, 16GB RAM, NVIDIA A10G)
- AMI: Deep Learning AMI GPU PyTorch 2.1.0
- Storage: 100GB gp3

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate soundfile librosa
```

## Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A

## License

This project is for internal use.
