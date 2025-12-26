import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
REQUIRED_KEYS = ["GOOGLE_API_KEY", "NCBI_EMAIL", "NCBI_API_KEY"]

missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

for model in genai.list_models():
    if "gemma" in model.name.lower():
        print(model.name)