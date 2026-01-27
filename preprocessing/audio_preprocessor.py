#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Preprocessing Script
Processes audio files: noise reduction, normalization, filtering
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    # Set encoding for stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # Set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Required installations:
# pip install librosa noisereduce soundfile scipy numpy

import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

# Logging configuration with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Class for audio preprocessing"""

    def __init__(self, target_sr: int = 16000):
        """
        Initialize preprocessor

        Args:
            target_sr: Target sample rate (Hz)
        """
        self.target_sr = target_sr

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and resample audio file

        Args:
            file_path: Path to audio file

        Returns:
            Tuple[audio_data, sample_rate]
        """
        try:
            logger.info(f"Loading: {file_path}")
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            duration = len(audio) / sr
            logger.info(f"Loaded: duration={duration:.2f}s, sr={sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def reduce_noise(self, audio: np.ndarray, sr: int,
                     stationary: bool = True,
                     prop_decrease: float = 1.0) -> np.ndarray:
        """
        Apply noise reduction

        Args:
            audio: Audio data
            sr: Sample rate
            stationary: Whether noise is stationary
            prop_decrease: Reduction level (0.0-1.0)

        Returns:
            Audio after noise reduction
        """
        logger.info("Noise reduction...")
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease
        )

    def normalize_audio(self, audio: np.ndarray,
                        target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio volume

        Args:
            audio: Audio data
            target_level: Target level in dB

        Returns:
            Normalized audio
        """
        logger.info("Normalizing volume...")

        # Normalize to range [-1, 1]
        audio_normalized = librosa.util.normalize(audio)

        # Optionally: adjust to target_level
        current_level = 20 * np.log10(np.sqrt(np.mean(audio_normalized**2)))
        gain = 10 ** ((target_level - current_level) / 20)
        audio_normalized = audio_normalized * gain

        # Ensure no clipping
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        return audio_normalized

    def apply_highpass_filter(self, audio: np.ndarray, sr: int,
                              cutoff: float = 80.0,
                              order: int = 5) -> np.ndarray:
        """
        Apply high-pass filter (removes low frequencies)

        Args:
            audio: Audio data
            sr: Sample rate
            cutoff: Cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered audio
        """
        logger.info(f"High-pass filter (cutoff={cutoff}Hz)...")
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='high')
        return filtfilt(b, a, audio)

    def apply_bandpass_filter(self, audio: np.ndarray, sr: int,
                              low_cutoff: float = 300.0,
                              high_cutoff: float = 3400.0,
                              order: int = 5) -> np.ndarray:
        """
        Apply band-pass filter (enhances speech range)

        Args:
            audio: Audio data
            sr: Sample rate
            low_cutoff: Lower cutoff frequency (Hz)
            high_cutoff: Upper cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered audio
        """
        logger.info(f"Band-pass filter ({low_cutoff}-{high_cutoff}Hz)...")
        nyquist = sr / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, audio)

    def enhance_speech(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance speech (dynamic compression + EQ)

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            Enhanced audio
        """
        logger.info("Enhancing speech...")

        # Band-pass filter for speech range (300-3400 Hz)
        audio = self.apply_bandpass_filter(audio, sr, 300, 3400)

        return audio

    def trim_silence(self, audio: np.ndarray, sr: int,
                     top_db: int = 30,
                     frame_length: int = 2048,
                     hop_length: int = 512) -> np.ndarray:
        """
        Remove silence from beginning and end

        Args:
            audio: Audio data
            sr: Sample rate
            top_db: Silence threshold in dB
            frame_length: Frame length
            hop_length: Hop between frames

        Returns:
            Audio without silence
        """
        logger.info("Trimming silence...")
        audio_trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return audio_trimmed

    def preprocess_full_pipeline(self,
                                 audio_path: str,
                                 output_path: Optional[str] = None,
                                 noise_reduce: bool = True,
                                 normalize: bool = True,
                                 enhance: bool = True,
                                 trim: bool = True,
                                 highpass: bool = True) -> str:
        """
        Apply full preprocessing pipeline

        Args:
            audio_path: Input file path
            output_path: Output file path (optional)
            noise_reduce: Whether to apply noise reduction
            normalize: Whether to normalize volume
            enhance: Whether to enhance speech
            trim: Whether to remove silence
            highpass: Whether to apply high-pass filter

        Returns:
            Path to processed file
        """
        logger.info("=" * 60)
        logger.info(f"Processing: {audio_path}")
        logger.info("=" * 60)

        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Apply preprocessing
        if trim:
            audio = self.trim_silence(audio, sr)

        if highpass:
            audio = self.apply_highpass_filter(audio, sr, cutoff=80)

        if noise_reduce:
            audio = self.reduce_noise(audio, sr, prop_decrease=1.0)

        if enhance:
            audio = self.enhance_speech(audio, sr)

        if normalize:
            audio = self.normalize_audio(audio, target_level=-20.0)

        # Determine output path
        if output_path is None:
            base_name = Path(audio_path).stem
            output_dir = Path(audio_path).parent / "preprocessed"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{base_name}_preprocessed.wav"

        # Save processed audio
        sf.write(str(output_path), audio, sr)
        logger.info(f"Saved: {output_path}")
        logger.info("=" * 60)

        return str(output_path)

    def batch_preprocess(self,
                         input_dir: str,
                         output_dir: Optional[str] = None,
                         extensions: List[str] = None,
                         **kwargs) -> List[str]:
        """
        Process multiple files at once

        Args:
            input_dir: Directory with input files
            output_dir: Output directory
            extensions: List of extensions to process
            **kwargs: Additional arguments for preprocess_full_pipeline

        Returns:
            List of paths to processed files
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path / "preprocessed"
            output_path.mkdir(exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))

        logger.info(f"Found {len(audio_files)} files to process")

        # Process each file
        processed_files = []
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}]")
            try:
                output_file = output_path / f"{audio_file.stem}_preprocessed.wav"
                result = self.preprocess_full_pipeline(
                    str(audio_file),
                    str(output_file),
                    **kwargs
                )
                processed_files.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Processed {len(processed_files)}/{len(audio_files)} files")
        logger.info(f"{'='*60}")

        return processed_files


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Audio file preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python audio_preprocessor.py -i audio.wav -o audio_clean.wav
  
  # Process all files in directory
  python audio_preprocessor.py -d ./recordings -od ./processed
  
  # Process with selected options
  python audio_preprocessor.py -i audio.wav --no-enhance --no-trim
  
  # Set sample rate
  python audio_preprocessor.py -i audio.wav -sr 22050
        """
    )

    parser.add_argument('-i', '--input', type=str,
                        help='Input file path')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file path')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory with files to process')
    parser.add_argument('-od', '--output-directory', type=str,
                        help='Output directory for processed files')
    parser.add_argument('-sr', '--sample-rate', type=int, default=16000,
                        help='Sample rate (Hz, default: 16000)')

    # Preprocessing options
    parser.add_argument('--no-noise-reduce', action='store_true',
                        help='Disable noise reduction')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable normalization')
    parser.add_argument('--no-enhance', action='store_true',
                        help='Disable speech enhancement')
    parser.add_argument('--no-trim', action='store_true',
                        help='Disable silence trimming')
    parser.add_argument('--no-highpass', action='store_true',
                        help='Disable high-pass filter')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.directory:
        parser.error("Required: -i/--input or -d/--directory argument")

    # Create preprocessor
    preprocessor = AudioPreprocessor(target_sr=args.sample_rate)

    # Preprocessing options
    preprocess_options = {
        'noise_reduce': not args.no_noise_reduce,
        'normalize': not args.no_normalize,
        'enhance': not args.no_enhance,
        'trim': not args.no_trim,
        'highpass': not args.no_highpass
    }

    try:
        if args.input:
            # Process single file
            result = preprocessor.preprocess_full_pipeline(
                args.input,
                args.output,
                **preprocess_options
            )
            logger.info(f"\nSuccess! File saved: {result}")

        elif args.directory:
            # Process directory
            results = preprocessor.batch_preprocess(
                args.directory,
                args.output_directory,
                **preprocess_options
            )
            logger.info(f"\nSuccess! Processed {len(results)} files")

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
