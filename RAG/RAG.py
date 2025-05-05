from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, OWL
import csv
import re
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FILE = "extracted_triples.ttl"
BASE_DIR = Path("RAG/Nodes-Rel")
OUTPUT_DIR = BASE_DIR / str(max([int(d.name) for d in BASE_DIR.iterdir() if d.is_dir() and d.name.isdigit()], default=0) + 1)

# Ensure output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- NAMESPACES ---
MCRO = Namespace("http://purl.obolibrary.org/obo/MCRO_")
OBO = Namespace("http://purl.obolibrary.org/obo/")
PROV = Namespace("https://www.w3.org/ns/prov#")

# --- LOAD GRAPH ---
graph = Graph()
graph.parse(INPUT_FILE, format="turtle")

# Bind namespaces for readability
graph.bind("mcro", MCRO)
graph.bind("obo", OBO)
graph.bind("prov", PROV)

# --- CLEAN URI HANDLING ---
def clean_uri(uri):
    uri_str = str(uri)
    
    # Handle MCRO namespace
    if uri_str.startswith(str(MCRO)):
        return f"mcro_{uri_str.split('MCRO_')[-1]}"
    
    # Handle OBO namespace
    if uri_str.startswith(str(OBO)):
        return f"obo_{uri_str.split('obo:')[-1].replace(':', '_')}"
    
    # Handle PROV namespace
    if uri_str.startswith(str(PROV)):
        return f"prov_{uri_str.split('#')[-1]}"
    
    # Fallback: extract last segment
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    elif "/" in uri_str:
        return uri_str.split("/")[-1]
    return re.sub(r"[^\w-]", "", uri_str)

# --- DATA STRUCTURES ---
nodes = {}
relationships = []

# --- PROCESS TRIPLES ---
for s, p, o in graph:
    s_clean = clean_uri(s)
    p_clean = clean_uri(p)

    # Initialize subject node
    if s_clean not in nodes:
        nodes[s_clean] = {"labels": set(), "properties": {}}

    # Handle types (RDF.type)
    if p == RDF.type:
        o_clean = clean_uri(o)
        nodes[s_clean]["labels"].add(o_clean)
    
    # Handle literals (e.g., prov:hasTextValue)
    elif isinstance(o, Literal):
        prop_key = p_clean
        value = str(o).strip()
        
        # Skip empty values
        if not value:
            continue

        # Append to list if multiple values
        if prop_key in nodes[s_clean]["properties"]:
            if not isinstance(nodes[s_clean]["properties"][prop_key], list):
                nodes[s_clean]["properties"][prop_key] = [nodes[s_clean]["properties"][prop_key]]
            nodes[s_clean]["properties"][prop_key].append(value)
        else:
            nodes[s_clean]["properties"][prop_key] = value
    
    # Handle object references (non-literals)
    else:
        o_clean = clean_uri(o)
        relationships.append((s_clean, p_clean, o_clean))
        if o_clean not in nodes:
            nodes[o_clean] = {"labels": set(), "properties": {}}

# --- CLEAN EMPTY NODES ---
# Remove nodes without any relationships or properties
valid_node_ids = {s for s, _, _ in relationships} | {o for _, _, o in relationships}
nodes = {nid: data for nid, data in nodes.items() if nid in valid_node_ids or data["properties"]}

# --- WRITE NODES CSV ---
nodes_path = OUTPUT_DIR / "nodes.csv"
with open(nodes_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([":ID", ":LABEL", "properties"])

    for node_id, data in nodes.items():
        labels = ";".join(sorted(data["labels"])) if data["labels"] else "Undefined"
        props = {}

        for k, v in data["properties"].items():
            if isinstance(v, list):
                props[k] = "; ".join(str(x).strip() for x in v if x.strip())
            else:
                props[k] = str(v).strip() if v else "N/A"

        writer.writerow([node_id, labels, str(props)])

# --- WRITE RELATIONSHIPS CSV ---
rels_path = OUTPUT_DIR / "relationships.csv"
with open(rels_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([":START_ID", "TYPE", ":END_ID"])
    
    # Filter out invalid relationships
    valid_rels = []
    for s, p, o in relationships:
        if s in nodes and o in nodes:
            valid_rels.append((s, p, o))
        else:
            print(f"⚠️ Skipping invalid relationship: {s} → {p} → {o}")
    
    writer.writerows(valid_rels)

print(f"✅ Files saved to: {OUTPUT_DIR}")