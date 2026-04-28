"""
translation_agent.py - Hybrid Translation Agent

Features:
✅ FREE language detection (langdetect library + Unicode checks)
✅ HIGH-QUALITY translation (Gemini)
✅ FAST (skips English input/output)
✅ Configurable supported languages via SUPPORTED_LANGUAGES
✅ Coming-soon languages handled gracefully with stored chat message
"""

from langdetect import detect, LangDetectException
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI


class TranslationAgent:
    """
    Translation agent using Gemini.
    Skips translation for English input/output.

    To add a new language:
      1. Add its code to SUPPORTED_LANGUAGES
      2. Ensure detect_language can return its code
      3. Add its name in get_language_name()

    To mark a language as coming soon (detected but not supported yet):
      1. Keep its detection logic in detect_language()
      2. Add its code to COMING_SOON_LANGUAGES
    """

    # ── Languages actively supported (add/remove to control) ────────────────
    # Total: 18 languages
    #   European  : English, Spanish, French, Portuguese, German, Italian, Dutch, Polish, Russian
    #   Middle East: Arabic
    #   South Asia : Hindi, Bengali
    #   Southeast  : Vietnamese, Indonesian, Malay
    #   East Asia  : Japanese, Korean, Chinese (Simplified)
    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'pt', 'de', 'it', 'nl', 'pl',
        'ru', 'ar', 'hi', 'bn', 'vi', 'id', 'ms',
        'ja', 'ko', 'zh-CN', 'km',
    }

    # ── Detected but not supported yet — shows "coming soon" message ─────────
    COMING_SOON_LANGUAGES = set()

    def __init__(self, google_api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key
        )
        print("Translation model: Gemini (gemini-2.0-flash)")

    def detect_language(self, text: str) -> str:
        """
        Detect language using Unicode checks + langdetect library.
        Returns language code, 'th-BLOCKED' for Thai, or code from COMING_SOON_LANGUAGES.
        """
        try:
            if len(text.strip()) < 2:
                return 'en'

            # Step 1: Unicode checks for scripts langdetect doesn't handle well
            if any('\u1780' <= c <= '\u17FF' for c in text):
                print("[Translation] Detected Khmer by Unicode range")
                return 'km'

            if any('\u0E00' <= c <= '\u0E7F' for c in text):
                print("[Translation] Detected Thai - BLOCKED")
                return 'th-BLOCKED'

            # Step 2: Pure ASCII text → always English, no exception
            # Every real non-English Latin-script language (French, Spanish, German,
            # Portuguese, Italian, Dutch, Polish...) uses accented characters.
            # A genuine non-English user WILL have é, ñ, ü, ç, etc. in their text.
            # ASCII-only text that "looks" French/Spanish to langdetect is always wrong.
            if all(c.isascii() for c in text):
                print("[Translation] Pure ASCII text → English")
                return 'en'

            # Step 3: Distinctive character checks before langdetect
            # ¿ and ¡ are unique to Spanish — langdetect often confuses short Spanish with French
            if any(c in text for c in '¿¡'):
                print("[Translation] Spanish characters detected → es")
                return 'es'

            # Step 4: Non-ASCII text → use langdetect
            lang = detect(text)
            print(f"[Translation] langdetect detected: {lang}")

            # Normalize codes
            code_map = {'zh-cn': 'zh-CN', 'zh-tw': 'zh-CN'}
            lang = code_map.get(lang, lang)

            # Allow supported + coming-soon (both are known languages)
            known = self.SUPPORTED_LANGUAGES | self.COMING_SOON_LANGUAGES
            if lang not in known:
                print(f"[Translation] Unsupported language '{lang}' → notifying user")
                return f'unsupported:{lang}'

            return lang

        except LangDetectException:
            print("[Translation] langdetect failed → assuming English")
            return 'en'
        except Exception as e:
            print(f"[Translation] Error in detection: {e} → assuming English")
            return 'en'

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        lang_map = {
            'en':         'English',
            'es':         'Spanish',
            'fr':         'French',
            'pt':         'Portuguese',
            'de':         'German',
            'it':         'Italian',
            'nl':         'Dutch',
            'pl':         'Polish',
            'ru':         'Russian',
            'ar':         'Arabic',
            'hi':         'Hindi',
            'bn':         'Bengali',
            'vi':         'Vietnamese',
            'id':         'Indonesian',
            'ms':         'Malay',
            'ja':         'Japanese',
            'ko':         'Korean',
            'zh-CN':      'Chinese',
            'km':         'Khmer',
            'th-BLOCKED': 'Thai (Blocked)',
        }
        return lang_map.get(lang_code, lang_code)

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text using Gemini."""
        src_name = self.get_language_name(src_lang)
        tgt_name = self.get_language_name(tgt_lang)
        prompt = (
            f"Translate the following medical text from {src_name} to {tgt_name}.\n"
            "Preserve medical meaning exactly. Keep names, numbers, and units unchanged.\n"
            "Return only the translated text.\n\n"
            f"Text:\n{text}"
        )
        raw = self.llm.invoke(prompt)
        translated = str(raw.content if hasattr(raw, "content") else raw).strip()
        return translated or text

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
            print(f"  Thai language detected → BLOCKED")
            return {
                "english_question": "THAI_LANGUAGE_BLOCKED",
                "source_language": "th-BLOCKED",
                "output_language": "en",
                "needs_translation": False
            }

        # Step 2.6: Unsupported language detected
        if source_lang.startswith('unsupported:'):
            detected_code = source_lang.split(':')[1]
            print(f"  Unsupported language '{detected_code}' → notifying user")
            supported_names = ', '.join(
                sorted(self.get_language_name(c) for c in self.SUPPORTED_LANGUAGES if c != 'en')
            )
            return {
                "english_question": f"UNSUPPORTED_LANGUAGE:{detected_code}",
                "source_language": detected_code,
                "output_language": "en",
                "needs_translation": False,
                "supported_languages": supported_names
            }

        # Step 2.7: Coming-soon language (e.g. Khmer)
        if source_lang in self.COMING_SOON_LANGUAGES:
            print(f"  {self.get_language_name(source_lang)} detected → COMING SOON")
            return {
                "english_question": f"COMING_SOON_LANGUAGE:{source_lang}",
                "source_language": source_lang,
                "output_language": "en",
                "needs_translation": False
            }

        # Step 3: Translate to English using Gemini
        print(f"  Translating from {self.get_language_name(source_lang)} to English...")

        try:
            english_question = self._translate(question, src_lang=source_lang, tgt_lang='en')
            print(f"  Translated: {english_question[:50]}...")

        except Exception as e:
            print(f"  Translation failed: {e}")
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
            return "Sorry, Thai language is not supported. Please use English or another supported language."

        # Translate to target language using Gemini
        print(f"  Translating from English to {self.get_language_name(output_language)}...")

        try:
            translated = self._translate(response, src_lang='en', tgt_lang=output_language)
            print(f"  Translated: {translated[:50]}...")
            return translated

        except Exception as e:
            print(f"  Translation failed: {e}")
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
