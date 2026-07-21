import re
from typing import List, Tuple, Set
import torch
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

class PhoneticUnificationEngine:
    """
    Stage 1: Converts code-switched text (Latin ASCII + Brahmic Devanagari) into a
    Universal Phonetic Representation (IPA) bottleneck while mapping language-switch
    boundary indices for downstream attention entropy regularization.
    """
    def __init__(self, language_map: dict = None):
        if language_map is None:
            self.language_map = {'en': 'en-us', 'hi': 'hi'}
        self.backend_en = EspeakBackend(self.language_map['en'], preserve_punctuation=True)
        self.backend_hi = EspeakBackend(self.language_map['hi'], preserve_punctuation=True)
        
        # Unicode ranges for Brahmic script (Devanagari) and ASCII Latin
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.latin_pattern = re.compile(r'[a-zA-Z]+')

    def _detect_script(self, word: str) -> str:
        """Identifies whether a token belongs to Hindi (Devanagari) or English (Latin)."""
        if self.devanagari_pattern.search(word):
            return 'hi'
        elif self.latin_pattern.search(word):
            return 'en'
        return 'punct'

    def process_text(self, raw_text: str) -> Tuple[List[str], Set[int], List[int]]:
        """
        Processes intra-sentential code-switched text into IPA tokens and calculates
        boundary indices B where language switching occurs.
        
        Returns:
            ipa_tokens: List of individual IPA phoneme symbols.
            boundary_indices: Set of token indices corresponding to script transitions (Beta).
            token_ids: Numerical IDs mapped from an internal phoneme vocabulary.
        """
        words = raw_text.strip().split()
        if not words:
            raise ValueError("Input raw_text cannot be empty.")

        ipa_sequence = []
        boundary_indices = set()
        current_lang = None
        
        for word_idx, word in enumerate(words):
            detected_lang = self._detect_script(word)
            
            # Skip punctuation from triggering language switch boundaries
            if detected_lang != 'punct':
                if current_lang is not None and detected_lang != current_lang:
                    # Mark the current transition index in the IPA sequence
                    boundary_indices.add(len(ipa_sequence))
                current_lang = detected_lang
            
            # Perform automated Grapheme-to-Phoneme conversion via eSpeak-NG backend
            backend = self.backend_hi if detected_lang == 'hi' else self.backend_en
            phonemes = backend.phonemize([word], strip=True)[0]
            
            # Character-level token density mapping
            for char in phonemes:
                if char != ' ':
                    ipa_sequence.append(char)
            ipa_sequence.append(' ') # Word separator

        if ipa_sequence and ipa_sequence[-1] == ' ':
            ipa_sequence.pop() # Clean trailing space
            
        return ipa_sequence, boundary_indices
