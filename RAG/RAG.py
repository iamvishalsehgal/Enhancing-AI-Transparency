from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF
import csv
import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

INPUT_FILE = "Ontology_mapper/Output/5/triples.ttl"
OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH"))
OUTPUT_DIR = OUTPUT_BASE_PATH / str(
    max([int(d.name) for d in OUTPUT_BASE_PATH.iterdir() if d.is_dir() and d.name.isdigit()], default=0) + 1
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MCRO = Namespace("http://purl.obolibrary.org/obo/MCRO_")
OBO = Namespace("http://purl.obolibrary.org/obo/")
PROV = Namespace("https://www.w3.org/ns/prov#")

graph = Graph()
graph.parse(INPUT_FILE, format="turtle")

graph.bind("mcro", MCRO)
graph.bind("obo", OBO)
graph.bind("prov", PROV)

def clean_uri(uri):
    uri_str = str(uri)
    if uri_str.startswith(str(MCRO)):
        return f"mcro_{uri_str.split('MCRO_')[-1]}"
    if uri_str.startswith(str(OBO)):
        return f"obo_{uri_str.split('obo:')[-1].replace(':', '_')}"
    if uri_str.startswith(str(PROV)):
        return f"prov_{uri_str.split('#')[-1]}"
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    if "/" in uri_str:
        return uri_str.split("/")[-1]
    return re.sub(r"[^\w-]", "", uri_str)

nodes = {}
relationships = []

for s, p, o in graph:
    s_clean = clean_uri(s)
    p_clean = clean_uri(p)

    if s_clean not in nodes:
        nodes[s_clean] = {"labels": set(), "properties": {}}

    if p == RDF.type:
        nodes[s_clean]["labels"].add(clean_uri(o))
    elif isinstance(o, Literal):
        val = str(o).strip()
        if not val:
            continue
        if p_clean in nodes[s_clean]["properties"]:
            existing = nodes[s_clean]["properties"][p_clean]
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(val)
            nodes[s_clean]["properties"][p_clean] = existing
        else:
            nodes[s_clean]["properties"][p_clean] = val
    else:
        o_clean = clean_uri(o)
        relationships.append((s_clean, p_clean, o_clean))
        if o_clean not in nodes:
            nodes[o_clean] = {"labels": set(), "properties": {}}

valid_node_ids = {s for s, _, _ in relationships} | {o for _, _, o in relationships}
nodes = {nid: data for nid, data in nodes.items() if nid in valid_node_ids or data["properties"]}

nodes_path = OUTPUT_DIR / "nodes.csv"
with open(nodes_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([":ID", ":LABEL", "properties"])

    for node_id, data in nodes.items():
        labels = ";".join(sorted(data["labels"])) if data["labels"] else "Undefined"
        props = {
            k: "; ".join(v) if isinstance(v, list) else str(v).strip()
            for k, v in data["properties"].items()
        }
        writer.writerow([node_id, labels, str(props)])

rels_path = OUTPUT_DIR / "relationships.csv"
with open(rels_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([":START_ID", "TYPE", ":END_ID"])

    valid_rels = [(s, p, o) for s, p, o in relationships if s in nodes and o in nodes]
    writer.writerows(valid_rels)

print(f"Files saved to: {OUTPUT_DIR}")

