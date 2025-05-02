import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCLwWkDW03zjzVKUQf3ui5wgcreVJdsMbw"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")