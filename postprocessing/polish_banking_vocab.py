#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polish Banking Domain Vocabulary for ASR Correction
Contains common terms, abbreviations, and typical ASR errors
"""

# Common Polish banking terms (for context injection in prompts)
BANKING_TERMS = [
    # Credit products
    "kredyt hipoteczny",
    "kredyt gotówkowy",
    "kredyt konsumpcyjny",
    "kredyt obrotowy",
    "kredyt inwestycyjny",
    "kredyt konsolidacyjny",
    "kredyt refinansowy",
    "pożyczka",

    # Cards
    "karta kredytowa",
    "karta debetowa",
    "karta płatnicza",
    "karta wielowalutowa",

    # Accounts
    "rachunek bieżący",
    "rachunek oszczędnościowy",
    "rachunek firmowy",
    "konto osobiste",
    "konto walutowe",

    # Savings and investments
    "lokata terminowa",
    "depozyt",
    "fundusz inwestycyjny",
    "obligacje",
    "akcje",

    # Insurance
    "polisa ubezpieczeniowa",
    "ubezpieczenie na życie",
    "ubezpieczenie kredytu",

    # Operations
    "przelew",
    "przelew natychmiastowy",
    "przelew zagraniczny",
    "przelew SEPA",
    "zlecenie stałe",
    "polecenie zapłaty",
    "wpłata",
    "wypłata",
    "przewalutowanie",
    "spłata rat",
    "wcześniejsza spłata",
    "nadpłata",
    "zaległość",

    # Financial terms
    "oprocentowanie",
    "RRSO",
    "marża",
    "prowizja",
    "opłata",
    "saldo",
    "limit kredytowy",
    "zdolność kredytowa",
    "historia kredytowa",
    "BIK",
    "KRD",
    "zabezpieczenie",
    "poręczenie",
    "hipoteka",
    "weksel",
    "rata",
    "kapitał",
    "odsetki",

    # Documents
    "umowa",
    "aneks",
    "regulamin",
    "tabela opłat",
    "harmonogram spłat",
    "zaświadczenie",
    "wyciąg",
    "potwierdzenie",

    # Authentication
    "PIN",
    "hasło",
    "kod autoryzacyjny",
    "SMS kod",
    "token",
    "podpis elektroniczny",

    # Customer service
    "reklamacja",
    "dyspozycja",
    "wniosek",
    "zgłoszenie",
    "doradca",
    "konsultant",
]

# Common ASR errors in Polish banking context
# Format: (incorrect, correct, description)
COMMON_ASR_ERRORS = [
    ("kredyt potoczny", "kredyt hipoteczny", "phonetically similar"),
    ("kredyt hipotyczny", "kredyt hipoteczny", "spelling error"),
    ("kredyt gotowy", "kredyt gotówkowy", "truncated word"),
    ("karta kretowa", "karta kredytowa", "missing syllable"),
    ("karta kreditowa", "karta kredytowa", "spelling error"),
    ("erso", "RRSO", "acronym misheard"),
    ("ERSO", "RRSO", "wrong first letter"),
    ("r r s o", "RRSO", "spelled out"),
    ("bik", "BIK", "lowercase acronym"),
    ("krde", "KRD", "phonetic error"),
    ("ka er de", "KRD", "spelled out"),
    ("rachunek biznesowy", "rachunek bieżący", "similar meaning confusion"),
    ("lokata terminy", "lokata terminowa", "grammar error"),
    ("przelew sepa", "przelew SEPA", "lowercase acronym"),
    ("pin kod", "PIN", "redundant word"),
]


def get_vocabulary_context() -> str:
    """Generate context string with banking vocabulary for prompt"""
    terms_list = "\n".join(f"- {term}" for term in BANKING_TERMS[:40])
    return f"""Polish banking terminology reference:
{terms_list}
"""


def get_error_examples() -> str:
    """Generate examples of common corrections for few-shot prompting"""
    examples = [
        ("Chciałbym wziąć kredyt potoczny na mieszkanie",
         "Chciałbym wziąć kredyt hipoteczny na mieszkanie"),
        ("Mam pytanie o kartę kretową visa",
         "Mam pytanie o kartę kredytową visa"),
        ("Jakie jest erso tego kredytu",
         "Jakie jest RRSO tego kredytu"),
        ("Chcę sprawdzić swój bik",
         "Chcę sprawdzić swój BIK"),
        ("Potrzebuję przelew sepa do Niemiec",
         "Potrzebuję przelew SEPA do Niemiec"),
    ]

    result = []
    for orig, corr in examples:
        result.append(f'Input: "{orig}"\nOutput: "{corr}"')

    return "\n\n".join(result)


def get_system_prompt() -> str:
    """Return system prompt for the LLM"""
    return """You are an expert in Polish banking terminology and speech recognition error correction.

Your task is to correct ASR (Automatic Speech Recognition) transcription errors in Polish banking call center recordings.

RULES:
1. Only fix clear transcription errors - do not change correctly transcribed words
2. Focus on domain-specific banking terms that are commonly misrecognized
3. Preserve the original meaning and sentence structure
4. Keep conversational elements (filler words, hesitations) if they appear natural
5. Correct acronyms to uppercase (RRSO, BIK, KRD, SEPA, PIN)
6. Do not add or remove content beyond corrections
7. Return ONLY the corrected text without any explanations or comments"""


def build_correction_prompt(transcription: str) -> str:
    """
    Build the user prompt for transcription correction

    Args:
        transcription: Raw ASR transcription text

    Returns:
        Formatted prompt for LLM
    """
    vocab_context = get_vocabulary_context()
    error_examples = get_error_examples()

    prompt = f"""Correct the following Polish banking call transcription.

{vocab_context}

CORRECTION EXAMPLES:
{error_examples}

TRANSCRIPTION TO CORRECT:
{transcription}

CORRECTED TRANSCRIPTION:"""

    return prompt
