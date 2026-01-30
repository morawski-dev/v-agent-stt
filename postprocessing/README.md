# LLM Post-Processor for ASR Transcription Correction

Corrects ASR transcription errors using OpenAI GPT with Polish banking domain specialization.

## Installation

```bash
pip install openai
```

## Configuration

Set OpenAI API key as environment variable:

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-...
```

## Usage

### Single File

```bash
python llm_postprocessor.py -i recording_transcription.txt
```

Output: `recording_corrected.txt`

### Batch Processing

```bash
python llm_postprocessor.py -d ./transcriptions -od ./corrected
```

Processes all `*_transcription.txt` files in the directory.

### Dry Run

Preview corrections without saving:

```bash
python llm_postprocessor.py -i recording_transcription.txt --dry-run
```

### Custom Model

```bash
python llm_postprocessor.py -i recording_transcription.txt --model gpt-4o
```

Available models:
- `gpt-4o-mini` (default) - Fast, cost-effective
- `gpt-4o` - Higher quality, more expensive

## Pipeline Integration

```
Audio -> [preprocessing] -> [whisper-ec2] -> *_transcription.txt
    -> [llm_postprocessor.py] -> *_corrected.txt -> [wer_calculator]
```

## Example Corrections

| ASR Output | Corrected |
|------------|-----------|
| kredyt potoczny | kredyt hipoteczny |
| karta kretowa | karta kredytowa |
| erso | RRSO |
| bik | BIK |
| przelew sepa | przelew SEPA |

## Domain Vocabulary

The script includes Polish banking terminology:
- Credit products (kredyt hipoteczny, gotówkowy, konsumpcyjny)
- Cards (karta kredytowa, debetowa)
- Accounts (rachunek bieżący, oszczędnościowy)
- Financial terms (RRSO, BIK, KRD, marża, prowizja)
- Banking operations (przelew, wpłata, wypłata)

## Output

- Input: `*_transcription.txt`
- Output: `*_corrected.txt`
- Logs: `llm_postprocessing.log`
