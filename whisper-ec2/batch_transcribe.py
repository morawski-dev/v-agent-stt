#!/usr/bin/env python3
"""
Script for transcribing multiple audio files
"""

import os
import sys
from pathlib import Path
from transcribe import transcribe_audio

def batch_transcribe(directory, language=None, extensions=['.wav', '.mp3', '.m4a', '.flac']):
    """
    Transcribes all audio files in directory
    """
    directory = Path(directory)

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.glob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {directory}")
        return

    print(f"Found {len(audio_files)} files to transcribe\n")

    # Transcribe each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*80}")
        print(f"File {i}/{len(audio_files)}: {audio_file.name}")
        print(f"{'='*80}\n")

        try:
            transcribe_audio(str(audio_file), language)
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue

    print(f"\nCompleted transcription of {len(audio_files)} files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 batch_transcribe.py <directory> [language]")
        sys.exit(1)

    directory = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None

    batch_transcribe(directory, language)
