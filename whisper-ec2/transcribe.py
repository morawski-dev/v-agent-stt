#!/usr/bin/env python3
"""
Script for audio transcription using Whisper Large V3
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from jiwer import wer, process_words

def normalize_text(text: str) -> str:
    """Simple text normalization for WER calculation."""
    text = text.lower()
    text = text.replace("—", " ").replace("-", " ")
    chars_to_remove = [",", ".", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]"]
    for ch in chars_to_remove:
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    return text


def calculate_wer(reference_text, hypothesis_text):
    """Calculate and print WER between reference and hypothesis."""
    ref_n = normalize_text(reference_text)
    hyp_n = normalize_text(hypothesis_text)

    print("\n" + "=" * 80)
    print("WER ANALYSIS")
    print("=" * 80)

    w = wer(ref_n, hyp_n)
    print(f"\nWER: {w:.4f} ({w*100:.2f}%)")

    out = process_words(ref_n, hyp_n)
    print(f"\nError breakdown:")
    print(f"  - Substitutions (S): {out.substitutions}")
    print(f"  - Deletions (D):     {out.deletions}")
    print(f"  - Insertions (I):    {out.insertions}")
    print(f"  - Hits (H):          {out.hits}")

    total_words = len(ref_n.split())
    print(f"\nTotal words in reference: {total_words}")
    print("=" * 80)

    return w


def transcribe_audio(audio_file, language=None, reference_file=None):
    """
    Transcribes audio file using Whisper Large V3

    Args:
        audio_file: Path to audio file
        language: Language code (e.g., 'pl', 'en') or None for autodetection
        reference_file: Path to ground truth TXT file for WER calculation
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
    # Note: chunk_length_s is intentionally omitted - Whisper has its own long-form
    # transcription mechanism (paper section 3.8) activated by return_timestamps=True.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=300,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Transcription
    print(f"Transcribing file: {audio_file}")
    print("Processing...")

    transcribe_start = time.time()

    _initial_prompt = (
        "Transkrypcja rozmowy z hurtownią mięsa. Numer klienta zapisuj 123 45 67."
        "Zamówienie: filet z kurczaka 30 kg, karkówka 15 kg, szynka kulka 50 kg"
        "Transkrypcja liczb nie słownie a w notacji liczbowej: dwa to 2, pięć to 5, dwanaście to 12."
        "Ceny zapisuj z przecinkiem: 21,50, nie 21.50."
        "Firmy: XZY, XYZ, XYZ, XYZ, XYZ, XYZ, XYZ, XYZ."
    )

    prompt_ids = processor.get_prompt_ids(_initial_prompt, return_tensors="pt").to(device)

    generate_kwargs = {
        "language": language if language else "pl",
        "task": "transcribe",
        "prompt_ids": prompt_ids,
        "no_repeat_ngram_size": 5,
    }

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

    # WER calculation if reference file provided
    if reference_file:
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
        calculate_wer(reference_text, result['text'])

    return result


if __name__ == "__main__":
    import sys

    import argparse
    import os

    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper Large V3")
    parser.add_argument("audio_file", help="Path to audio file (WAV/MP3/FLAC/OGG/M4A)")
    parser.add_argument("language", nargs="?", default=None, help="Language code (e.g., 'pl', 'en')")
    parser.add_argument("--ref", dest="reference_file", default=None,
                        help="Path to ground truth TXT file for WER calculation")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: File {args.audio_file} does not exist!")
        sys.exit(1)

    if args.reference_file and not os.path.exists(args.reference_file):
        print(f"Error: Reference file {args.reference_file} does not exist!")
        sys.exit(1)

    # Run transcription
    transcribe_audio(args.audio_file, args.language, args.reference_file)
