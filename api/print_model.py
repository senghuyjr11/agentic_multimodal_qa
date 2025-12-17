import google.generativeai as genai

genai.configure(api_key="GOOGLE_API_KEY_REMOVED")

for model in genai.list_models():
    if "gemma" in model.name.lower():
        print(model.name)