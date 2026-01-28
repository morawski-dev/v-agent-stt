# WER Calculator

A Python script for calculating Word Error Rate (WER) between reference transcriptions and speech recognition hypotheses.

## Requirements

- Python 3.x
- jiwer library

```bash
pip install jiwer
```

## Usage

Edit the `ref` (reference/ground truth) and `hyp` (hypothesis/ASR output) variables in the script, then run:

```bash
python wer_calculator.py
```

## Features

- **Text normalization**: Converts text to lowercase, removes punctuation, and normalizes whitespace
- **WER calculation**: Returns error rate as a value between 0 (perfect) and 1 (completely wrong)
- **Error breakdown**: Shows detailed counts of:
  - Substitutions (S): words replaced with different words
  - Deletions (D): words missing from hypothesis
  - Insertions (I): extra words in hypothesis
  - Hits (H): correctly recognized words

## Output Example

```
WER: 0.7500 (75.00%)

Error breakdown:
  - Substitutions (S): 10
  - Deletions (D):     5
  - Insertions (I):    3
  - Hits (H):          20

Total words in reference: 35
```

## Customization

The `normalize_text()` function can be extended for domain-specific preprocessing needs.
