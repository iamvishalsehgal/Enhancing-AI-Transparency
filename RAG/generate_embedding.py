import google.generativeai as genai
import json

genai.configure(api_key="AIzaSyCLwWkDW03zjzVKUQf3ui5wgcreVJdsMbw")

# Generate embedding
result = genai.embed_content(
    model="models/embedding-001",
    content="machine learning",
    task_type="retrieval_document"
)

# Save to file
with open("RAG/query_embedding.json", "w") as f:
    json.dump(result["embedding"], f)