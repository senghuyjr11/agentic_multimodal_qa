"""
translation_agent.py - Translation with deep-translator

Features:
✅ FREE (uses deep-translator)
✅ NO HTTPX CONFLICTS (uses requests instead)
✅ RELIABLE (actively maintained)
✅ FAST (skips English input/output)

Installation:
pip install deep-translator
"""

from deep_translator import GoogleTranslator
import re
from typing import Dict


class TranslationAgent:
    """
    Smart translation agent that only translates when needed.

    Optimization: Skips translation for English input/output!
    """

    def __init__(self):
        print("✓ Translation Agent initialized (deep-translator)")

    def detect_language(self, text: str) -> str:
        """
        Detect language using pattern matching + Unicode ranges.

        Args:
            text: Text to detect language from

        Returns:
            Language code ('en', 'es', 'fr', 'ja', 'ko', 'km', etc.)
        """
        try:
            text_lower = text.lower().strip()

            # Common English phrases (short text)
            common_english = [
                'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
                'yes', 'no', 'good', 'great', 'nice', 'bad', 'good job',
                'well done', 'awesome', 'cool', 'wow', 'please', 'sorry',
                'bye', 'goodbye', 'help', 'what', 'why', 'how', 'when',
                'where', 'who', 'tell me', 'show me', 'explain', 'i see',
                'got it', 'understood', 'appreciate it'
            ]

            if text_lower in common_english:
                print("[Translation] Short English phrase detected")
                return 'en'

            # Check for common English sentence patterns
            english_patterns = [
                r'^my name is ',
                r'^i am ',
                r"^what('s| is)",
                r'^(do you|did you|can you|will you|are you|is there|is this)',
                r'^(the|a|an) ',
                r'^this is ',
                r'^that is ',
                r'^these are ',
                r'^those are ',
                r' is | are | was | were ',
                r' have | has | had ',
                r' can | could | should | would | will ',
                r'^(show|tell|explain|describe|find|search)',
            ]

            if any(re.search(pattern, text_lower) for pattern in english_patterns):
                print("[Translation] English sentence pattern detected")
                return 'en'

            # Check if text too short
            if len(text.strip()) < 3:
                print("[Translation] Text too short, assuming English")
                return 'en'

            # Check for Latin script with diacritics (Spanish, French, Portuguese, etc.)
            has_diacritics = any(c in text for c in 'áéíóúñü¿¡àèìòùâêîôûäëïöüçÁÉÍÓÚÑÜÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÇ')

            if has_diacritics:
                print("[Translation] Latin script with diacritics detected → needs translation")
                if '¿' in text or '¡' in text or 'ñ' in text:
                    return 'es'  # Spanish
                elif any(c in text for c in 'àèéêëôûç'):
                    return 'fr'  # French
                elif any(c in text for c in 'ãõ'):
                    return 'pt'  # Portuguese
                else:
                    return 'es'  # Default to Spanish

            # Check for non-Latin scripts
            # Khmer Unicode range: \u1780-\u17FF
            if any('\u1780' <= c <= '\u17FF' for c in text):
                print("[Translation] Detected Khmer by Unicode range")
                return 'km'

            # Japanese: Hiragana, Katakana, Kanji
            if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text):
                print("[Translation] Detected Japanese by Unicode range")
                return 'ja'

            # Korean: Hangul
            if any('\uAC00' <= c <= '\uD7AF' for c in text):
                print("[Translation] Detected Korean by Unicode range")
                return 'ko'

            # Chinese: Simplified/Traditional
            if any('\u4E00' <= c <= '\u9FFF' for c in text):
                print("[Translation] Detected Chinese by Unicode range")
                return 'zh-CN'

            # Thai
            if any('\u0E00' <= c <= '\u0E7F' for c in text):
                print("[Translation] Detected Thai by Unicode range")
                return 'th'

            # Arabic
            if any('\u0600' <= c <= '\u06FF' for c in text):
                print("[Translation] Detected Arabic by Unicode range")
                return 'ar'

            # Check if mostly basic ASCII (English alphabet only)
            basic_ascii_ratio = sum(c.isascii() and c.isalnum() for c in text) / len(text) if text else 0

            if basic_ascii_ratio > 0.9:
                print("[Translation] Text is mostly basic ASCII → English")
                return 'en'

            # Default to English
            print("[Translation] Could not detect language → assuming English")
            return 'en'

        except Exception as e:
            print(f"[Translation] Error in detection: {e}, assuming English")
            return 'en'

    def process_input(self, question: str) -> Dict:
        """
        Process input question with smart translation.
        """
        source_lang = self.detect_language(question)

        print(f"\n[Translation Input]")
        print(f"  Detected language: {source_lang}")

        if source_lang == 'en':
            print(f"  Input is English → Skipping translation ✓")
            return {
                "english_question": question,
                "source_language": "en",
                "output_language": "en",
                "needs_translation": False
            }

        print(f"  Translating from {source_lang} to English...")

        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            english_question = translator.translate(question)
            print(f"  ✓ Translated: {english_question[:50]}...")
        except Exception as e:
            print(f"  ✗ Translation failed: {e}")
            print(f"  Using original text as fallback")
            english_question = question
            source_lang = 'en'

        return {
            "english_question": english_question,
            "source_language": source_lang,
            "output_language": source_lang,
            "needs_translation": source_lang != 'en'
        }

    def process_output(self, response: str, output_language: str) -> str:
        """
        Translate response back to user's language.
        """
        print(f"\n[Translation Output]")
        print(f"  Target language: {output_language}")

        if output_language == 'en':
            print(f"  Output is English → Skipping translation ✓")
            return response

        print(f"  Translating from English to {output_language}...")

        try:
            translator = GoogleTranslator(source='en', target=output_language)
            translated = translator.translate(response)
            print(f"  ✓ Translated: {translated[:50]}...")
            return translated
        except Exception as e:
            print(f"  ✗ Translation failed: {e}")
            print(f"  Returning original English text")
            return response