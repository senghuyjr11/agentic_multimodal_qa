"""
translation_agent.py - Hybrid Translation Agent

Features:
✅ FREE language detection (pattern matching + Unicode)
✅ HIGH-QUALITY translation (Gemma 3 12B)
✅ FAST (skips English input/output)
✅ CONTEXT-AWARE (LLM understands medical terminology)

Installation:
pip install google-generativeai
"""

import google.generativeai as genai
import re
from typing import Dict


class TranslationAgent:
    """
    Smart translation agent using Gemma 3 12B for translation.

    Optimization: Skips translation for English input/output!
    """

    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        print("✓ Translation Agent initialized (Gemini 2.0 Flash)")

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
                'got it', 'understood', 'appreciate it', 'huh'
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

            # Thai - BLOCKED
            if any('\u0E00' <= c <= '\u0E7F' for c in text):
                print("[Translation] Detected Thai - BLOCKED")
                return 'th-BLOCKED'

            # Arabic
            if any('\u0600' <= c <= '\u06FF' for c in text):
                print("[Translation] Detected Arabic by Unicode range")
                return 'ar'

            # Vietnamese (with diacritics)
            if any(c in text for c in 'ăâđêôơưĂÂĐÊÔƠƯ'):
                print("[Translation] Detected Vietnamese")
                return 'vi'

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

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        lang_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'pt': 'Portuguese',
            'km': 'Khmer',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-CN': 'Chinese',
            'ar': 'Arabic',
            'vi': 'Vietnamese',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'hi': 'Hindi',
            'th-BLOCKED': 'Thai (Blocked)'
        }
        return lang_map.get(lang_code, lang_code)

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
        # Step 1: Detect language
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

        # Step 2.5: Block Thai language
        if source_lang == 'th-BLOCKED':
            print(f"  Thai language detected → BLOCKED ✗")
            return {
                "english_question": "THAI_LANGUAGE_BLOCKED",
                "source_language": "th-BLOCKED",
                "output_language": "en",
                "needs_translation": False
            }

        # Step 3: Translate to English using Gemma 3 12B
        print(f"  Translating from {self.get_language_name(source_lang)} to English...")

        try:
            prompt = f"""Translate the following text from {self.get_language_name(source_lang)} to English.

IMPORTANT RULES:
- Translate accurately while preserving medical terminology
- Keep the meaning and tone of the original
- If it's a medical question, keep medical terms accurate
- Output ONLY the English translation, no explanations

{self.get_language_name(source_lang)} text: {question}

English translation:"""

            response = self.model.generate_content(prompt)
            english_question = response.text.strip()

            # Clean up any quotes or extra formatting
            english_question = english_question.strip('"').strip("'").strip()

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
        Translate response back to user's language using Gemma 3 12B.

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

        # Block Thai language output
        if output_language == 'th-BLOCKED':
            print(f"  Thai language output → BLOCKED ✗")
            return "Sorry, this system does not support Thai language. Thai language and Thai people are banned from using this medical system. Please use English or another supported language."

        # Translate to target language using Gemma 3 12B
        print(f"  Translating from English to {self.get_language_name(output_language)}...")

        try:
            prompt = f"""Translate the following English medical response to {self.get_language_name(output_language)}.

IMPORTANT RULES:
- Translate accurately while preserving medical terminology
- Keep medical terms clear and understandable
- Preserve any citations like [1], [2], [3] exactly as they are
- Preserve any markdown formatting (**, *, links)
- Keep URLs unchanged
- Output ONLY the {self.get_language_name(output_language)} translation, no explanations

English text:
{response}

{self.get_language_name(output_language)} translation:"""

            result = self.model.generate_content(prompt)
            translated = result.text.strip()

            # Clean up any quotes or extra formatting
            translated = translated.strip('"').strip("'").strip()

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
    import os

    agent = TranslationAgent(google_api_key=os.getenv("GOOGLE_API_KEY"))

    print("=" * 70)
    print("TEST 1: English Input (should skip translation)")
    print("=" * 70)

    result = agent.process_input("What is diabetes?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")

    print("\n" + "=" * 70)
    print("TEST 2: Spanish Input (should translate with Gemma)")
    print("=" * 70)

    result = agent.process_input("¿Qué es la diabetes?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")
    print(f"  Needs translation: {result['needs_translation']}")

    output = agent.process_output(
        "Diabetes is a chronic disease that affects how your body regulates blood sugar.",
        "es"
    )
    print(f"\nSpanish output: {output}")

    print("\n" + "=" * 70)
    print("TEST 3: Korean Input (should translate with Gemma)")
    print("=" * 70)

    result = agent.process_input("당뇨병이란 무엇인가요?")
    print(f"\nResult:")
    print(f"  English question: {result['english_question']}")
    print(f"  Source language: {result['source_language']}")

    output = agent.process_output(
        "Diabetes is a condition where blood sugar levels are too high.",
        "ko"
    )
    print(f"\nKorean output: {output}")