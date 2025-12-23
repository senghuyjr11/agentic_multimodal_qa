"""
text_only_agent.py - Handles text-only questions (no image)
Uses LLM for intelligent classification instead of hardcoded rules
"""
import google.generativeai as genai
import json
import re


class TextOnlyAgent:
    """Routes text-only questions using LLM-based classification.
    - Casual questions: direct response
    - Medical questions: flag for PubMed search (no direct medical answers)
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemma-3-4b-it")

    def _extract_json(self, text: str) -> dict | None:
        """Safely extract JSON from LLM output."""
        if not text:
            return None

        # Try to find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def classify(self, question: str, conversation_context: str = "") -> str:
        """Use LLM to classify question type: 'casual' or 'medical'.

        Args:
            question: User's question
            conversation_context: Previous conversation history

        Returns:
            'casual' or 'medical'
        """

        context_section = ""
        if conversation_context:
            # Truncate context to avoid token limits
            context_section = f"""
Previous conversation context:
{conversation_context[-1000:]}  
"""

        prompt = f"""{context_section}
You are classifying user questions for a Medical VQA Assistant.

Current user question: "{question}"

Classify this question as either:
- "casual": Greetings, thank you, who are you, what can you do, general chitchat
- "medical": Any question about diseases, symptoms, treatments, medical images, diagnoses, health conditions, or follow-up questions about medical topics discussed previously

IMPORTANT RULES:
1. If the question contains medical terms (disease, symptom, treatment, diagnosis, etc.) → ALWAYS "medical"
2. If the question uses pronouns (this, that, it) and the context shows a medical discussion → "medical"
3. If the question is a follow-up to a medical topic → "medical"
4. Only greetings and meta questions about the assistant are "casual"
5. When in doubt → "medical" (safer to search than guess)

Return ONLY a JSON object:
{{"type": "casual"}} or {{"type": "medical"}}

Examples:

Question: "hello"
Context: (empty)
Output: {{"type": "casual"}}

Question: "who are you"
Context: (empty)
Output: {{"type": "casual"}}

Question: "What disease does this relate to?"
Context: Previous discussion about aortic valve endocarditis image
Output: {{"type": "medical"}}

Question: "What causes it?"
Context: Previous discussion about pneumonia
Output: {{"type": "medical"}}

Question: "What are the symptoms of pneumonia?"
Context: (empty)
Output: {{"type": "medical"}}

Now classify the current question above:"""

        try:
            response = self.model.generate_content(prompt)
            data = self._extract_json(response.text)

            if data and "type" in data:
                classification = data["type"]
                print(f"[LLM Classification] '{question}' → {classification}")
                return classification

            # Fallback: if JSON extraction fails, default to medical for safety
            print(f"[LLM Classification] Failed to extract JSON, defaulting to 'medical'")
            return "medical"

        except Exception as e:
            print(f"[LLM Classification] Error: {e}, defaulting to 'medical'")
            return "medical"

    def respond(self, question: str, conversation_context: str = "") -> dict:
        """Route question based on LLM classification.

        Args:
            question: Question in English (already translated by TranslationAgent)
            conversation_context: Previous conversation history (for context-aware responses)

        Returns:
            dict with question_type and response (only for casual questions)
            For medical questions, returns flag to use PubMed pipeline
        """
        print(f"\n[TextOnlyAgent] Processing: '{question}'")
        print(f"[TextOnlyAgent] Has context: {bool(conversation_context)}")

        question_type = self.classify(question, conversation_context)

        print(f"[TextOnlyAgent] Final classification: {question_type}\n")

        if question_type == "casual":
            # Generate casual response with context awareness
            context_prompt = ""
            if conversation_context:
                # Include recent context for continuity
                context_prompt = f"\nPrevious conversation:\n{conversation_context[-500:]}\n\n"

            prompt = f"""{context_prompt}You are a friendly Medical VQA Assistant. Answer this casual question briefly and naturally.

Question: {question}

Guidelines:
- Keep response short and friendly (1-2 sentences max)
- If asked who you are: "I'm a Medical VQA Assistant that helps analyze medical images and answer health-related questions."
- If asked what you can do: "I can analyze medical images, answer questions about diseases and treatments, and provide information backed by scientific research."
- Be warm and conversational but professional
- No medical advice for casual greetings

Response:"""

            response = self.model.generate_content(prompt)

            return {
                "question": question,
                "question_type": "casual",
                "response": response.text.strip(),
            }

        else:  # medical
            print("⚠️ Medical question detected - routing to PubMed pipeline")

            return {
                "question": question,
                "question_type": "medical"
            }