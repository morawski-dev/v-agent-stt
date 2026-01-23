#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WER (Word Error Rate) Calculator
Requirements: pip install jiwer
"""

from jiwer import wer, process_words

def normalize_text(text: str) -> str:
    """
    Simple text normalization.
    You can extend this for specific needs (e.g., domain-specific preprocessing).
    """
    text = text.lower()
    text = text.replace("—", " ").replace("-", " ")
    # Remove punctuation marks
    chars_to_remove = [",", ".", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]"]
    for ch in chars_to_remove:
        text = text.replace(ch, " ")
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def main():
    # Reference text (ground truth)
    ref = "Good morning, I have 12 thousand dollars in my account."

    # Hypothesis text (from speech recognition)
    hyp = "Good morning I have twelve thousand dollars in my account"

    # Normalization
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)

    print("=" * 60)
    print("WER ANALYSIS")
    print("=" * 60)
    print(f"\nReference (normalized): {ref_n}")
    print(f"Hypothesis (normalized): {hyp_n}")
    print()

    # WER as a number 0..1 (0 = perfect match, 1 = completely wrong)
    w = wer(ref_n, hyp_n)
    print(f"WER: {w:.4f} ({w*100:.2f}%)")
    print()

    # Detailed components: S/D/I
    out = process_words(ref_n, hyp_n)
    print("Error breakdown:")
    print(f"  - Substitutions (S): {out.substitutions}")
    print(f"  - Deletions (D):     {out.deletions}")
    print(f"  - Insertions (I):    {out.insertions}")
    print(f"  - Hits (H):          {out.hits}")
    print()

    total_words = len(ref_n.split())
    print(f"Total words in reference: {total_words}")
    print("=" * 60)

if __name__ == "__main__":
    main()
