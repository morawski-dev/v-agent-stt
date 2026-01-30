#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Post-Processing Script for ASR Transcription Correction
Corrects domain-specific errors in Polish banking transcriptions using OpenAI GPT
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Windows UTF-8 encoding support
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_postprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionCorrector:
    """Class for LLM-based transcription correction using OpenAI GPT"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize the corrector

        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client

    def correct_transcription(self, text: str) -> Dict:
        """
        Correct a single transcription using GPT

        Args:
            text: ASR transcription text

        Returns:
            Dict with original, corrected text, and metadata
        """
        from polish_banking_vocab import build_correction_prompt, get_system_prompt

        prompt = build_correction_prompt(text)
        system_prompt = get_system_prompt()

        try:
            logger.info(f"Sending to {self.model} for correction...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.1
            )

            corrected = response.choices[0].message.content.strip()

            return {
                "original": text,
                "corrected": corrected,
                "model": self.model,
                "success": True,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return {
                "original": text,
                "corrected": text,  # Fallback: return original
                "model": self.model,
                "success": False,
                "error": str(e)
            }

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     dry_run: bool = False) -> Dict:
        """
        Process a single transcription file

        Args:
            input_path: Path to input transcription file
            output_path: Path for output file (auto-generated if None)
            dry_run: If True, show result but don't save

        Returns:
            Dict with processing result
        """
        input_path = Path(input_path)

        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        if input_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {input_path}")

        # Read input
        logger.info(f"Reading: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"Input length: {len(text)} characters")

        # Correct
        result = self.correct_transcription(text)

        if not result['success']:
            logger.warning(f"Correction failed, using original text")
            return result

        # Determine output path
        if output_path is None:
            if '_transcription.txt' in input_path.name:
                output_path = input_path.parent / input_path.name.replace(
                    '_transcription.txt', '_corrected.txt')
            else:
                output_path = input_path.parent / f"{input_path.stem}_corrected.txt"
        else:
            output_path = Path(output_path)

        # Show diff
        if result['original'] != result['corrected']:
            logger.info("Changes detected in correction")
        else:
            logger.info("No changes needed")

        # Save output
        if dry_run:
            logger.info(f"[DRY RUN] Would save to: {output_path}")
            print("\n" + "=" * 60)
            print("CORRECTED TEXT:")
            print("=" * 60)
            print(result['corrected'])
            print("=" * 60 + "\n")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['corrected'])
            logger.info(f"Saved: {output_path}")

        result['output_path'] = str(output_path)
        return result

    def batch_process(self, input_dir: str, output_dir: Optional[str] = None,
                      dry_run: bool = False) -> List[Dict]:
        """
        Process all transcription files in directory

        Args:
            input_dir: Directory with transcription files
            output_dir: Output directory (same as input if None)
            dry_run: If True, show results but don't save

        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path

        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # Find transcription files
        transcription_files = list(input_path.glob("*_transcription.txt"))

        if not transcription_files:
            logger.warning(f"No *_transcription.txt files found in {input_dir}")
            return []

        logger.info(f"Found {len(transcription_files)} transcription files")
        logger.info("=" * 60)

        results = []
        for i, tf in enumerate(transcription_files, 1):
            logger.info(f"\n[{i}/{len(transcription_files)}] Processing: {tf.name}")

            try:
                out_file = output_path / tf.name.replace(
                    '_transcription.txt', '_corrected.txt')
                result = self.process_file(str(tf), str(out_file), dry_run)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {tf.name}: {e}")
                results.append({
                    "original_file": str(tf),
                    "success": False,
                    "error": str(e)
                })

        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        logger.info("\n" + "=" * 60)
        logger.info(f"BATCH COMPLETE: {successful}/{len(results)} files processed successfully")

        return results


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='LLM-based ASR transcription correction for Polish banking domain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python llm_postprocessor.py -i recording_transcription.txt

  # Process with custom output path
  python llm_postprocessor.py -i recording_transcription.txt -o corrected.txt

  # Use different model
  python llm_postprocessor.py -i recording_transcription.txt --model gpt-4o

  # Batch processing
  python llm_postprocessor.py -d ./transcriptions -od ./corrected

  # Dry run (show corrections without saving)
  python llm_postprocessor.py -i recording_transcription.txt --dry-run

Environment:
  OPENAI_API_KEY    OpenAI API key (required)
        """
    )

    # Input/Output arguments
    parser.add_argument('-i', '--input', type=str,
                        help='Input transcription file (*_transcription.txt)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output corrected file (*_corrected.txt)')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory with transcription files to process')
    parser.add_argument('-od', '--output-directory', type=str,
                        help='Output directory for corrected files')

    # Model arguments
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model name (default: gpt-4o-mini)')

    # Processing options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show corrections without saving to file')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.directory:
        parser.error("Required: -i/--input or -d/--directory argument")

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error("OPENAI_API_KEY environment variable not set")

    # Process
    try:
        corrector = TranscriptionCorrector(model=args.model)

        if args.input:
            result = corrector.process_file(
                args.input,
                args.output,
                dry_run=args.dry_run
            )
            if result['success']:
                logger.info(f"\nSuccess! Corrected file: {result.get('output_path', 'N/A')}")
                if result.get('tokens_used'):
                    logger.info(f"Tokens used: {result['tokens_used']}")
            else:
                logger.error(f"\nFailed: {result.get('error', 'Unknown error')}")
                return 1

        elif args.directory:
            results = corrector.batch_process(
                args.directory,
                args.output_directory,
                dry_run=args.dry_run
            )
            successful = sum(1 for r in results if r.get('success', False))
            if successful < len(results):
                return 1

    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
