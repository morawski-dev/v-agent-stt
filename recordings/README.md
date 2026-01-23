# ASR POC – Data & Evaluation Requirements

This document summarizes the requirements and open questions regarding audio data, transcriptions, and evaluation methodology for the ASR Proof of Concept (POC).

## 1. WER (Word Error Rate) Evaluation

To calculate **WER (Word Error Rate)** — the standard metric for ASR quality — we need:

- **ASR output** (generated automatically by the system under evaluation)
- **Reference transcription (ground truth)** — a correct, human-verified transcript of what was actually said

Without high-quality reference transcriptions, WER **cannot be calculated**.

### Preferred Scenario
- Customer delivers **audio recordings together with accurate transcriptions**

### Alternative (if customer does not provide transcriptions)
We will need to manually create a **gold set** of high-quality transcriptions.

- Recommended size: **30–50 calls initially**
- The gold set must be **representative**, diversified across:
    - Short vs long calls
    - Different consultants
    - Different audio quality:
        - clean
        - noisy
        - overlapping speech
    - Different topics:
        - cards
        - loans
        - e-banking
        - etc.

> ⚠️ WER can vary significantly between calls.  
> A small or biased sample may produce misleading results (too easy / too hard cases).

- Over time, we may need to **extend the gold set** by another **20–30 calls** to stabilize results.

### Without Reference Transcriptions
We will only be able to provide:
- subjective assessments (“sounds good / bad”)
- partial qualitative checks (“looks OK on selected fragments”)

These are **proxy metrics**, not a hard WER suitable for a **go / no-go decision**.

---

## 4. Audio Channel Configuration

- Are the recordings:
    - mono?
    - stereo / dual-channel?

> Some Contact Centers provide separate channels for:
> - customer
> - consultant  
    > This is **extremely helpful** for ASR quality and analysis.

---

## 5. Audio Format & Sampling

- File format:
    - WAV / MP3
- Sampling rate:
    - 8 kHz or 16 kHz
- Bitrate (if applicable)

---

## 6. Call Metadata

Is metadata available for each call, such as:
- topic
- consultant ID
- date
- call duration
- tags / labels (if any)

---

## 7. Consents & Anonymization

- Is the audio already anonymized?
- Are we allowed to process the data in the cloud (AWS) as part of the POC?

---

## 8. Language Characteristics

- Is the language **100% Polish**?
- Do other languages appear?
    - English
    - Ukrainian
    - Russian
- Are there:
    - proper names
    - spelling out words
    - dictation of numbers (IDs, card numbers, etc.)

### Known Challenge
- The biggest ASR issues usually come from **regional dialects**, e.g.:
    - Silesian
    - Kashubian

---

## Summary

High-quality reference transcriptions are **critical** for reliable ASR evaluation.  
Without them, objective metrics like WER cannot be computed, and POC conclusions will be inherently limited.
