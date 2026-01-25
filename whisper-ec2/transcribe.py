#!/usr/bin/env python3
"""
Script for audio transcription using Whisper Large V3
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

def transcribe_audio(audio_file, language=None):
    """
    Transcribes audio file using Whisper Large V3

    Args:
        audio_file: Path to audio file
        language: Language code (e.g., 'pl', 'en') or None for autodetection
    """

    print("=" * 80)
    print("WHISPER LARGE V3 - AUDIO TRANSCRIPTION")
    print("=" * 80)

    # Check GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Model ID
    model_id = "openai/whisper-large-v3"

    print(f"\nLoading model {model_id}...")
    print("   (First run may take several minutes - model is ~3GB)")

    start_time = time.time()

    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds\n")

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Transcription
    print(f"Transcribing file: {audio_file}")
    print("Processing...")

    transcribe_start = time.time()

    # Run transcription
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language

    result = pipe(audio_file, generate_kwargs=generate_kwargs)

    transcribe_time = time.time() - transcribe_start

    # Results
    print("\n" + "=" * 80)
    print("TRANSCRIPTION RESULT")
    print("=" * 80)
    print(f"\n{result['text']}\n")
    print("=" * 80)
    print(f"Transcription time: {transcribe_time:.2f} seconds")
    print("=" * 80)

    # Save to file
    output_file = audio_file.replace('.wav', '_transcription.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    print(f"\nTranscription saved to: {output_file}")

    return result


if __name__ == "__main__":
    import sys

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe.py <audio_file.wav> [language]")
        print("\nExamples:")
        print("  python3 transcribe.py test_audio.wav")
        print("  python3 transcribe.py test_audio.wav pl")
        print("  python3 transcribe.py test_audio.wav en")
        sys.exit(1)

    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if file exists
    import os
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} does not exist!")
        sys.exit(1)

    # Run transcription
    transcribe_audio(audio_file, language)
