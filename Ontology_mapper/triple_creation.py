from huggingface_hub import HfApi, ModelCard
import json
import re
import google.generativeai as genai
from rdflib import Graph, Namespace, URIRef, Literal, RDF
from dotenv import load_dotenv
import os 

load_dotenv()

GTOKEN = os.getenv("GEMINI_API_KEY")
HFTOKEN = os.getenv("HFTOKEN")
MCRO_TTL_PATH = "Ontology_mapper/Base_ontology/mcro.ttl"


genai.configure(api_key=GTOKEN)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

prefix_map = {
    "mcro": "http://purl.obolibrary.org/obo/MCRO_",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "prov": "https://www.w3.org/ns/prov#",
    "obo": "http://purl.obolibrary.org/obo/"
}

def clean_identifier(text):
    """Generate safe identifier"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text).replace(" ", ""))[:50]

def upload_mcro_ontology():
    """Upload .ttl ontology to Gemini"""
    try:
        print("Uploading MCRO ontology...")
        mcro_file = genai.upload_file(path=MCRO_TTL_PATH)
        print(f"Ontology uploaded: {mcro_file.name}")
        return mcro_file
    except Exception as e:
        print(f"File upload failed: {e}")
        raise

def get_mapped_triples(model_card_text, mcro_file, model_id):
    prompt = f"""Using the attached Model Card Ontology (MCRO) file ({mcro_file.uri}), analyze this Hugging Face model card text and return only RDF triples in JSON format. Follow these strict rules:

 Rules for Mapping
1. Only use terms defined in the MCRO ontology.
2. Always map metadata fields to appropriate MCRO concepts **by their CURIEs**, such as:
   - license → mcro:LicenseInformationSection
   - dataset → mcro:DatasetInformationSection
   - model architecture → mcro:ModelArchitectureInformationSection
   - citation → mcro:CitationInformationSection
   - intended use case → mcro:UseCaseInformationSection
3. Use proper relationships:
   - `rdf:type` for types
   - `prov:hasTextValue` for textual values (like "mit", "CNN", "ImageNet")
   - Appropriate `mcro:hasX` properties for linking model to its sections
4. Never assign `rdf:type` to abstract IAO classes like `obo:IAO_*`.
5. Never directly type instances with `obo:MCRO_0000004`, `obo:MCRO_0000016`, etc. — always use CURIEs like `mcro:CitationInformationSection`, `mcro:LicenseInformationSection`.
6. Only the root model instance (e.g., `mcro:{clean_identifier(model_id)}`) should be assigned `rdf:type mcro:Model`.
   - Do NOT assign `rdf:type mcro:Model` to supporting entities like `mcro:{clean_identifier(model_id)}-ModelDetail`, `-ModelDetailSection`, or other sections.
   - Instead, assign their appropriate type such as `mcro:ModelArchitectureInformationSection`, `mcro:UseCaseInformationSection`, etc.

Sample Output Format:
[
  {{
    "s": "mcro:{clean_identifier(model_id)}",
    "p": "rdf:type",
    "o": "mcro:Model"
  }},
  {{
    "s": "mcro:{clean_identifier(model_id)}",
    "p": "mcro:hasLicense",
    "o": "mcro:{clean_identifier(model_id)}-License"
  }},
  {{
    "s": "mcro:{clean_identifier(model_id)}-License",
    "p": "rdf:type",
    "o": "mcro:LicenseInformationSection"
  }},
  {{
    "s": "mcro:{clean_identifier(model_id)}-License",
    "p": "prov:hasTextValue",
    "o": "mit"
  }}
]
Important: Return ONLY the JSON array. No explanation. No markdown.

Input Text:
{model_card_text}
"""

    try:
        response = gemini_model.generate_content(
            contents=[prompt, mcro_file],
            request_options={"timeout": 60}
        )

        json_str = response.text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()

        return json.loads(json_str)
    
    except Exception as e:
        print(f"Error: {e}")
        return []

def process_huggingface_models(limit):
    """Main pipeline with single-prompt mapping"""
    mcro_file = upload_mcro_ontology()
    api = HfApi(token=HFTOKEN)
    models = list(api.list_models(sort="downloads", direction=-1, limit=limit))
    all_triples = []

    for idx, model_info in enumerate(models):
        try:
            card = ModelCard.load(model_info.id, token=HFTOKEN)
            triples = get_mapped_triples(card.text, mcro_file, model_info.id)
            
            if triples:
                all_triples.extend(triples)
                print(f"Processed {idx+1}/{len(models)}: {model_info.id}")
                print(f"Generated {len(triples)} triples")
            else:
                print(f"No triples returned for {model_info.id}")

        except Exception as e:
            print(f"Error processing {model_info.id}: {str(e)}")

    return all_triples

def get_next_output_directory(base_dir="Ontology_mapper/Output"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
    next_index = max(existing, default=0) + 1
    next_dir = os.path.join(base_dir, str(next_index))
    os.makedirs(next_dir, exist_ok=True)
    return next_dir

def convert_json_triples_to_turtle(json_triples_path, turtle_output_path, prefix_map):
    """Convert JSON triples to Turtle format and save to file"""
    g = Graph()

    for prefix, uri in prefix_map.items():
        g.bind(prefix, Namespace(uri))

    with open(json_triples_path, "r") as f:
        triples = json.load(f)

    def expand(curie):
        if ":" in curie:
            prefix, local = curie.split(":", 1)
            if prefix in prefix_map:
                return URIRef(prefix_map[prefix] + local)
            elif prefix == "obo":
                return URIRef(f"http://purl.obolibrary.org/obo/{local}")
        return URIRef(curie)

    for t in triples:
        s = expand(t["s"])
        p = expand(t["p"])
        o = t["o"]

        if str(p) == prefix_map["prov"] + "hasTextValue":
            g.add((s, p, Literal(o)))
        else:
            g.add((s, p, expand(o)))

    turtle_data = g.serialize(format="turtle", encoding="utf-8").decode("utf-8")
    with open(turtle_output_path, "w") as f:
        f.write(turtle_data)

    print(f"Turtle file saved to {turtle_output_path}")

if __name__ == "__main__":
    print(" Ontology-aware triple generation started ")
    
    output_dir = get_next_output_directory()
    json_path = os.path.join(output_dir, "triples.json")
    ttl_path = os.path.join(output_dir, "triples.ttl")
    
    all_triples = process_huggingface_models(limit=10)

    with open(json_path, "w") as f:
        json.dump(all_triples, f, indent=2)
    print(f"Saved triples to {json_path}")
    
    print("Converting to Turtle...")
    convert_json_triples_to_turtle(json_path, ttl_path, prefix_map)
    print("Completed.")
