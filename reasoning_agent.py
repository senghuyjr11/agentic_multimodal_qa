"""
reasoning_agent.py - Final agent using Gemma 3 12B
"""
import google.generativeai as genai


class ReasoningAgent:
    """Explains VQA answers in simple language with multi-language support.
    model:
    models/gemma-3-1b-it
    models/gemma-3-4b-it
    models/gemma-3-12b-it
    models/gemma-3-27b-it
    models/gemma-3n-e4b-it
    models/gemma-3n-e2b-it
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_fast = genai.GenerativeModel("gemma-3-4b-it")  # English
        self.model_multilang = genai.GenerativeModel("gemini-2.5-flash-lite")  # Other languages

    def generate_response(
            self,
            question: str,
            vqa_answer: str,
            pubmed_articles: str,
            language: str
    ) -> dict:

        # Select model based on language
        if language.lower() == "english":
            model = self.model_fast
            model_name = "gemma-3-4b-it"
        else:
            model = self.model_multilang
            model_name = "gemini-2.5-flash-lite"

        print(f"Using model: {model_name} for {language}")

        prompt = f"""You are a medical AI assistant. Based on the following information, provide a clear and helpful response.

    QUESTION: {question}
    VQA MODEL ANSWER: {vqa_answer}

    RELATED MEDICAL LITERATURE:
    {pubmed_articles}

    Please provide your response in {language} with the following format:

    1. ANSWER: State the answer clearly
    2. SIMPLE EXPLANATION: Explain what this means in plain language that anyone can understand
    3. CLINICAL CONTEXT: Provide relevant medical context based on the literature
    4. SUMMARY: A brief 1-2 sentence summary

    Keep the explanation friendly and easy to understand. Avoid complex medical jargon unless necessary."""

        response = model.generate_content(prompt)

        return {
            "original_answer": vqa_answer,
            "original_question": question,
            "enhanced_response": response.text,
            "language": language,
            "model_used": model_name
        }


if __name__ == "__main__":
    agent = ReasoningAgent(api_key="GOOGLE_API_KEY_REMOVED")

    result = agent.generate_response(
        question="how are the histone subunits charged?",
        vqa_answer="positively charged",
        pubmed_articles="""
Related Medical Literature:

[1] Mechanistic insights into DNA binding by BD1 of the TAF1 tandem bromodomain module.
    Transcription initiation factor TFIID subunit 1 (TAF1) is a pivotal component of the TFIID complex, critical for RNA polymerase II-mediated transcription initiation. However, the molecular basis by which TAF1 recognizes and associates with chromatin remains incompletely understood. Here, we report that the tandem bromodomain module of TAF1 engages nucleosomal DNA through a distinct positively charged surface patch on the first bromodomain (BD1). Electrostatic potential mapping and molecular dock...
    Link: https://pubmed.ncbi.nlm.nih.gov/40928734/

[2] Structural Basis for the Interaction Between Yeast Chromatin Assembly Factor 1 and Proliferating Cell Nuclear Antigen.
    Proliferating cell nuclear antigen (PCNA), the homotrimeric eukaryotic sliding clamp protein, recruits and coordinates the activities of a multitude of proteins that function on DNA at the replication fork. Chromatin assembly factor 1 (CAF-1), one such protein, is a histone chaperone that deposits histone proteins onto DNA immediately following replication. The interaction between CAF-1 and PCNA is essential for proper nucleosome assembly at silenced genomic regions. Most proteins that bind PCNA...
    Link: https://pubmed.ncbi.nlm.nih.gov/38969056/

[3] Molecular insight into the SETD1A/B N-terminal region and its interaction with WDR82.
    SETD1A and SETD1B originate from Set1, the sole H3K4 methyltransferase in yeast, and they play important roles in active gene transcription. Here, we present the crystal structures of the RRM domains of human SETD1A and SETD1B. Although both RRM domains adopt a canonical RRM fold, their structural features are different from that of the yeast Set1 RRM domain, their yeast homolog. By using an ITC binding assay, we found an intrinsically disordered region in SETD1A/B binds WDR82. The structural an...
    Link: https://pubmed.ncbi.nlm.nih.gov/37030068/
""",
        language="English"
    )

    print(result["enhanced_response"])