from rdflib import Graph, Namespace
import csv
import re
import os

def clean_uri(uri):
    uri_str = str(uri.split("/")[-1])
    return re.sub(r"[^a-zA-Z0-9_]", "", uri_str)

def export_to_csv(rdf_file="extracted_triples.ttl"):
    # Load RDF
    graph = Graph()
    graph.parse(rdf_file, format="turtle")
    
    # Collect all nodes and relationships
    nodes = set()
    triples = []
    
    for s, p, o in graph:
        s_clean = clean_uri(s)
        p_clean = clean_uri(p)
        o_clean = clean_uri(o)
        
        nodes.add(s_clean)
        nodes.add(o_clean)
        triples.append((s_clean, p_clean, o_clean))
    
    # Write nodes.csv
    with open("nodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id:ID", "name"])
        for node in nodes:
            writer.writerow([node, node])
    
    # Write relationships.csv
    with open("relationships.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID", "type", ":END_ID"])
        for s, p, o in triples:
            writer.writerow([s, p, o])

if __name__ == "__main__":
    export_to_csv()