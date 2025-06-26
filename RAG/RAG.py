import json
import re
from pathlib import Path

# --- CONFIG ---
INPUT_JSON_PATH = Path("Ontology_mapper/Output/1/triples.json")
OUTPUT_CYPHER_FILE = Path("import_graph.cypher")

# --- FUNCTION TO CLEAN STRINGS ---
def clean_latex_string(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\\url\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'[{}]', '', text)
    text = text.replace('"', '\\"')  # Escape quotes for Cypher compatibility
    return text.strip()

# --- LOAD TRIPLES ---
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    triples = json.load(f)

nodes = {}
relationships = []

for triple in triples:
    s, p, o = triple["s"], triple["p"], triple["o"]

    # Handle rdf:type for labeling
    if p == "rdf:type":
        if s not in nodes:
            nodes[s] = {"labels": set(), "properties": {}}
        nodes[s]["labels"].add(o.split(":")[-1])
    elif o.startswith(("mcro:", "obo:", "prov:")):
        relationships.append((s, p, o))
        for uri in (s, o):
            if uri not in nodes:
                nodes[uri] = {"labels": set(), "properties": {}}
    else:
        if s not in nodes:
            nodes[s] = {"labels": set(), "properties": {}}
        key = p.split(":")[-1]
        val = clean_latex_string(o)
        nodes[s]["properties"][key] = val

# --- GENERATE CYPHER ---
cypher_lines = ["// AUTO-GENERATED CYPHER IMPORT"]

def safe_id(uri):
    # Replace any invalid Cypher variable characters with underscores
    return re.sub(r'[^a-zA-Z0-9_]', '_', uri).strip('_')

# 1. CREATE NODES
for uri, data in nodes.items():
    node_id = safe_id(uri)
    labels = ":".join(data["labels"]) if data["labels"] else "Entity"
    props = ', '.join([f'{k}: "{v}"' for k, v in data["properties"].items()])
    cypher = f'CREATE ({node_id}:{labels} {{id: "{uri}"' + (f', {props}' if props else '') + '});'
    cypher_lines.append(cypher)

# 2. CREATE RELATIONSHIPS
for s, p, o in relationships:
    s_id = safe_id(s)
    o_id = safe_id(o)
    rel_type = p.split(":")[-1].upper()
    cypher = f'MATCH (a {{id: "{s}"}}), (b {{id: "{o}"}}) CREATE (a)-[:{rel_type}]->(b);'
    cypher_lines.append(cypher)

# --- SAVE FILE ---
with open(OUTPUT_CYPHER_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(cypher_lines))

print(f"✅ Cypher script saved to {OUTPUT_CYPHER_FILE}")