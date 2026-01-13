"""
translation_agent.py - Optimized Translation with deep-translator

Features:
✅ FREE (uses deep-translator)
✅ FAST (skips English input/output)
✅ RELIABLE (more maintained than googletrans)
✅ LOCAL detection (no API calls for language detection)

Installation:
pip install deep-translator langdetect
"""

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
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
        Detect language using langdetect (FREE, LOCAL, FAST!)

        Args:
            text: Text to detect language from

        Returns:
            Language code ('en', 'es', 'fr', 'ja', 'ko', 'km', etc.)
        """
        try:
            # For very short text, langdetect is unreliable
            # Check for common English phrases first
            text_lower = text.lower().strip()

            # Common English phrases (short text)
            common_english = [
                'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
                'yes', 'no', 'good', 'great', 'nice', 'bad', 'good job',
                'well done', 'awesome', 'cool', 'wow', 'please', 'sorry',
                'bye', 'goodbye', 'help', 'what', 'why', 'how', 'when',
                'where', 'who', 'tell me', 'show me', 'explain'
            ]

            if text_lower in common_english:
                print("[Translation] Short English phrase detected")
                return 'en'

            # Need at least a few characters for langdetect
            if len(text.strip()) < 3:
                print("[Translation] Text too short, assuming English")
                return 'en'

            # Try langdetect
            lang = detect(text)

            # langdetect sometimes returns 'km' or 'lo' for Khmer
            # Normalize to 'km' for Khmer
            if lang in ['km', 'lo']:
                lang = 'km'

            return lang

        except LangDetectException as e:
            # If detection fails, try to detect by character range
            # Khmer Unicode range: \u1780-\u17FF
            if any('\u1780' <= c <= '\u17FF' for c in text):
                print("[Translation] Detected Khmer by Unicode range")
                return 'km'

            # Default to English if can't detect
            print(f"[Translation] Language detection failed: {e}, assuming English")
            return 'en'
        except Exception as e:
            print(f"[Translation] Unexpected error in detection: {e}, assuming English")
            return 'en'

    def process_input(self, question: str) -> Dict:
        """
        Process input question with smart translation.

        OPTIMIZATION: Only translates if NOT English!

        Args:
            question: User's question in any language

        Returns:
            {
                "english_question": str,      # Question in English
                "source_language": str,       # Detected language code
                "output_language": str,       # Language for final output
                "needs_translation": bool     # Was translation needed?
            }
        """
        # Step 1: Detect language (FREE, local, instant!)
        source_lang = self.detect_language(question)

        print(f"\n[Translation Input]")
        print(f"  Detected language: {source_lang}")

        # Step 2: Check if translation needed
        if source_lang == 'en':
            # OPTIMIZATION: Skip translation! ⚡
            print(f"  Input is English → Skipping translation ✓")
            return {
                "english_question": question,
                "source_language": "en",
                "output_language": "en",
                "needs_translation": False
            }

        # Step 3: Translate to English (only if needed!)
        print(f"  Translating from {source_lang} to English...")

        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            english_question = translator.translate(question)
            print(f"  ✓ Translated: {english_question[:50]}...")
        except Exception as e:
            print(f"  ✗ Translation failed: {e}")
            print(f"  Using original text as fallback")
            english_question = question

        return {
            "english_question": english_question,
            "source_language": source_lang,
            "output_language": source_lang,
            "needs_translation": True
        }

    def process_output(self, response: str, output_language: str) -> str:
        """
        Translate response back to user's language.

        OPTIMIZATION: Only translates if NOT English!

        Args:
            response: Response in English
            output_language: Target language code

        Returns:
            Translated response (or original if English)
        """
        print(f"\n[Translation Output]")
        print(f"  Target language: {output_language}")

        # Check if translation needed
        if output_language == 'en':
            # OPTIMIZATION: Skip translation! ⚡
            print(f"  Output is English → Skipping translation ✓")
            return response

        # Translate to target language (only if needed!)
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


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    agent = TranslationAgent()

    print("=" * 70)
    print("TEST 1: English Input (should skip translation)")
    print("=" * 70)

    result = agent.process_input("What is diabetes?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")  # False!

    output = agent.process_output("Diabetes is a chronic disease.", "en")
    print(f"\nOutput: {output}")

    # ============================================

    print("\n" + "=" * 70)
    print("TEST 2: Spanish Input (should translate)")
    print("=" * 70)

    result = agent.process_input("¿Qué es la diabetes?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")  # True!

    output = agent.process_output(
        "Diabetes is a chronic disease that affects blood sugar.",
        "es"
    )
    print(f"\nOutput: {output}")

    # ============================================

    print("\n" + "=" * 70)
    print("TEST 3: Japanese Input (should translate)")
    print("=" * 70)

    result = agent.process_input("糖尿病とは何ですか？")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")  # True!

    output = agent.process_output(
        "Diabetes is a chronic metabolic disease.",
        "ja"
    )
    print(f"\nOutput: {output}")

    # ============================================

    print("\n" + "=" * 70)
    print("TEST 4: Korean Input (should translate)")
    print("=" * 70)

    result = agent.process_input("당뇨병이란 무엇인가요?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")  # True!

    output = agent.process_output(
        "Diabetes is a condition where blood sugar levels are too high.",
        "ko"
    )
    print(f"\nOutput: {output}")