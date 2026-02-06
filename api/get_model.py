from google import genai as genai_client
from google.genai import types as genai_types
import os
from dotenv import load_dotenv

load_dotenv()

# Validate required keys
REQUIRED_KEYS = ["GOOGLE_API_KEY", "NCBI_EMAIL", "NCBI_API_KEY"]
missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

# Test embedding
client = genai_client.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# List all available models
for model in client.models.list():
    if "embed" in model.name.lower():
        print(f"Available: {model.name}")