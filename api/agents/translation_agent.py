"""
translation_agent.py - Hybrid Translation Agent

Features:
✅ FREE language detection (langdetect library + Unicode checks)
✅ HIGH-QUALITY translation (facebook/nllb-200-distilled-1.3B on GPU)
✅ FAST (skips English input/output)
✅ Configurable supported languages via SUPPORTED_LANGUAGES
✅ Coming-soon languages handled gracefully with stored chat message
"""

import torch
from langdetect import detect, LangDetectException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict


class TranslationAgent:
    """
    Translation agent using facebook/nllb-200-distilled-1.3B on GPU.
    Skips translation for English input/output.

    To add a new language:
      1. Add its code → NLLB Flores-200 code in NLLB_CODES
      2. Add its code to SUPPORTED_LANGUAGES
      3. Add its name in get_language_name()

    To mark a language as coming soon (detected but not supported yet):
      1. Keep its detection logic in detect_language()
      2. Add its code to COMING_SOON_LANGUAGES
    """

    # ── All languages we CAN translate (NLLB Flores-200 codes) ──────────────
    NLLB_CODES = {
        'en':    'eng_Latn',
        'es':    'spa_Latn',
        'fr':    'fra_Latn',
        'pt':    'por_Latn',
        'de':    'deu_Latn',
        'it':    'ita_Latn',
        'nl':    'nld_Latn',
        'pl':    'pol_Latn',
        'ru':    'rus_Cyrl',
        'ar':    'arb_Arab',
        'hi':    'hin_Deva',
        'bn':    'ben_Beng',
        'vi':    'vie_Latn',
        'id':    'ind_Latn',
        'ms':    'zsm_Latn',
        'ja':    'jpn_Jpan',
        'ko':    'kor_Hang',
        'zh-CN': 'zho_Hans',
        'km':    'khm_Khmr',   # kept for coming-soon use
    }

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
        'ja', 'ko', 'zh-CN',
    }

    # ── Detected but not supported yet — shows "coming soon" message ─────────
    COMING_SOON_LANGUAGES = {
        'km',           # Khmer — NLLB accuracy too low for medical terms
    }

    def __init__(self, google_api_key: str = None):
        model_name = "facebook/nllb-200-distilled-1.3B"
        print(f"Loading translation model: {model_name}...")

        # Load on CPU to reserve all GPU VRAM for VQA models (Qwen2-VL-7B + Qwen3-VL-2B)
        # Translation runs once per message so CPU speed is acceptable
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()

        print("Translation model loaded on CPU (GPU reserved for VQA models)")

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
        """Translate text using NLLB-200."""
        src_nllb = self.NLLB_CODES.get(src_lang)
        tgt_nllb = self.NLLB_CODES.get(tgt_lang)

        if not src_nllb or not tgt_nllb:
            raise ValueError(f"Unsupported language pair: {src_lang} → {tgt_lang}")

        # NLLB requires src_lang set on tokenizer before encoding
        self.tokenizer.src_lang = src_nllb

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        target_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512,
                num_beams=4,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

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

        # Step 3: Translate to English using NLLB-200
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
            return "Sorry, this system does not support Thai language. Thai language and Thai people are banned from using this medical system. Please use English or another supported language."

        # Translate to target language using NLLB-200
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