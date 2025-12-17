"""
translation_agent.py - Front layer for language handling
"""
import google.generativeai as genai
from langdetect import detect


class TranslationAgent:
    """Handles all language translation as the front layer."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemma-3-12b-it")

    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            # Check for Khmer characters first (Unicode range)
            if any('\u1780' <= char <= '\u17FF' for char in text):
                return 'Khmer'

            lang_code = detect(text)
            # Map common codes to full names
            lang_map = {
                'km': 'Khmer',
                'ko': 'Korean',
                'en': 'English',
                'ja': 'Japanese',
                'zh-cn': 'Chinese',
                'zh-tw': 'Chinese',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'vi': 'Vietnamese',
                'ar': 'Arabic'
            }
            return lang_map.get(lang_code, 'English')
        except:
            return 'English'

    def to_english(self, text: str, source_language: str) -> str:
        """Translate text to English if needed."""
        if source_language == 'English':
            return text

        prompt = f"""Translate this {source_language} text to English. 
Only return the translation, no explanations.

Text: {text}"""

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def from_english(self, text: str, target_language: str) -> str:
        """Translate English text back to target language."""
        if target_language == 'English':
            return text

        prompt = f"""Translate this English text to {target_language}.
Keep medical terms accurate. Only return the translation.

Text: {text}"""

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def process_input(self, question: str) -> dict:
        """Process user input - detect and translate to English."""
        original_language = self.detect_language(question)
        english_question = self.to_english(question, original_language)

        print(f"Detected language: {original_language}")
        if original_language != 'English':
            print(f"Translated to English: {english_question}")

        return {
            "original_question": question,
            "english_question": english_question,
            "detected_language": original_language
        }

    def process_output(self, english_response: str, target_language: str) -> str:
        """Translate final response back to user's language."""
        if target_language == 'English':
            return english_response

        print(f"Translating response back to {target_language}...")
        return self.from_english(english_response, target_language)