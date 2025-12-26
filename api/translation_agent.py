"""
translation_agent.py - Front layer for language handling
"""
import re

import google.generativeai as genai
from langdetect import detect


class TranslationAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.translate_model = genai.GenerativeModel("gemma-3-12b-it")
        self.detect_model = genai.GenerativeModel("gemma-3-1b-it")  # or 4b-it

    # ---------- Output-language preference (instruction) ----------
    def detect_output_language_preference(self, text: str) -> str | None:
        t = text.lower()

        patterns = {
            "Khmer": r"\b(explain|answer|reply|respond|write)\b.*\b(khmer|cambodian)\b",
            "Korean": r"\b(explain|answer|reply|respond|write)\b.*\b(korean|한국어)\b",
            "Japanese": r"\b(explain|answer|reply|respond|write)\b.*\b(japanese|日本語)\b",
            "Chinese": r"\b(explain|answer|reply|respond|write)\b.*\b(chinese|中文)\b",
            "English": r"\b(explain|answer|reply|respond|write)\b.*\b(english)\b",
        }

        for lang, pat in patterns.items():
            if re.search(pat, t):
                return lang
        return None

    # ---------- Source-language detection (what language the text is written in) ----------
    def detect_source_language(self, text: str) -> str:
        try:
            # Khmer script check
            if any('\u1780' <= c <= '\u17FF' for c in text):
                return "Khmer"

            code = detect(text)
            lang_map = {
                "km": "Khmer",
                "ko": "Korean",
                "en": "English",
                "ja": "Japanese",
                "zh-cn": "Chinese",
                "zh-tw": "Chinese",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "vi": "Vietnamese",
                "ar": "Arabic",
                "ur" : "Urdu",
                "sw":"Swahili"
            }
            return lang_map.get(code, "English")
        except Exception:
            return "English"

    def to_english(self, text: str, source_language: str) -> str:
        if source_language == "English":
            return text
        prompt = f"""Translate this {source_language} text to English.
Only return the translation.

Text: {text}"""
        return self.translate_model.generate_content(prompt).text.strip()

    def from_english(self, text: str, target_language: str) -> str:
        if target_language == "English":
            return text
        prompt = f"""Translate this English text to {target_language}.
Keep medical terms accurate. Only return the translation.

Text: {text}"""
        return self.translate_model.generate_content(prompt).text.strip()

    def process_input(self, question: str) -> dict:
        source_language = self.detect_source_language(question)
        output_language = self.detect_output_language_preference(question) or source_language
        english_question = self.to_english(question, source_language)

        return {
            "original_question": question,
            "english_question": english_question,

            # New fields (recommended)
            "source_language": source_language,
            "output_language": output_language,

            # Backward compatibility (prevents KeyError in existing code)
            "detected_language": output_language
        }

    def process_output(self, english_response: str, output_language: str) -> str:
        return self.from_english(english_response, output_language)
