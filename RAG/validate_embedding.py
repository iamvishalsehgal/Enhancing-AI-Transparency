import json

# Load and validate
with open("RAG/query_embedding.json", "r") as f:
    data = json.load(f)

# Ensure data is a flat list of floats
assert isinstance(data, list), "Not a list"
assert all(isinstance(x, (int, float)) for x in data), "Contains non-numeric values"
assert len(data) == 768, "Length is not 768"

# Re-save as strict JSON
with open("RAG/query_embedding.json", "w") as f:
    json.dump(data, f, ensure_ascii=True)