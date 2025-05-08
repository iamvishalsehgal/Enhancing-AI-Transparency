import re
import os
import requests
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class Config:
    GRAPHDB_ENDPOINT = os.getenv("GRAPHDB_ENDPOINT")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

    PREFIXES = """
PREFIX mcro: <http://purl.obolibrary.org/obo/MCRO_>
PREFIX prov1: <https://www.w3.org/ns/prov#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

genai.configure(api_key=Config.GEMINI_API_KEY)
gemini = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)


class KnowledgeGraphQuerySystem:
    def __init__(self):
        self.session = requests.Session()

    def get_schema_context(self) -> str:
        """Return known schema structure and ontology design."""
        return """
Known Model Properties
mcro:hasUseCase - links to use case information
mcro:hasTrainingData - links to training data details
mcro:hasModelArchitecture - links to architecture information
mcro:hasLimitation - links to limitations

Common Text Value Pattern
?section prov1:hasTextValue ?text

Example:
mcro:Falconsainsfwimagedetection a mcro:Model ;
    mcro:hasUseCase mcro:Falconsainsfwimagedetection-UseCase .

mcro:Falconsainsfwimagedetection-UseCase prov1:hasTextValue "NSFW Image Classification" .
"""

    def generate_sparql(self, question: str) -> str:
        """Generate a SPARQL query from a natural language question."""
        prompt = f"""{Config.PREFIXES}

Convert the following question into a SPARQL query using the knowledge graph schema:

{self.get_schema_context()}

Rules:
1. Always include the PREFIX declarations.
2. Use path patterns like:
   ?model mcro:hasUseCase ?section .
   ?section prov1:hasTextValue ?value .

Question: {question}

SPARQL:
"""

        for _ in range(3):
            try:
                response = gemini.generate_content(prompt)
                match = re.search(r"```sparql(.*?)```", response.text, re.DOTALL)
                if match:
                    query = match.group(1).strip()
                    if not query.startswith("PREFIX"):
                        query = Config.PREFIXES + "\n" + query
                    return query
            except Exception as e:
                print(f"Gemini Error: {e}")
        return ""

    def _execute_query(self, query: str) -> dict:
        """Execute SPARQL query and return JSON results."""
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": "Gemini-KG-Query-System"
        }

        try:
            response = self.session.post(
                Config.GRAPHDB_ENDPOINT,
                headers=headers,
                data=query,
                timeout=15
            )
            return response.json() if response.ok else {"error": f"{response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def format_results(self, results: dict) -> str:
        """Format the SPARQL query results."""
        if "error" in results:
            return f"Error: {results['error']}"

        bindings = results.get('results', {}).get('bindings', [])
        if not bindings:
            return "No results found."

        primary_var = next(iter(bindings[0].keys())) if bindings else None
        output = []

        for row in bindings:
            if primary_var:
                entity_uri = row[primary_var]['value']
                entity = entity_uri.split('/')[-1] if '/' in entity_uri else entity_uri

                details = [
                    f"{k}: {v['value'].split('#')[-1]}"
                    for k, v in row.items()
                    if k != primary_var and 'value' in v
                ]
                line = f"{entity}"
                if details:
                    line += f" ({', '.join(details)})"
                output.append(line)

        return f"Found {len(output)} result(s):\n" + "\n".join(output)

    def interactive_query(self):
        """Interactive mode for asking natural language questions."""
        print("Knowledge Graph Query System (type 'exit' to quit)")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['exit', 'quit']:
                break

            print("Generating SPARQL query")
            sparql = self.generate_sparql(question)

            if not sparql:
                print("Failed to generate a valid SPARQL query.")
                continue

            print("\nGenerated SPARQL Query:")
            print(sparql)

            print("\nExecuting query")
            results = self._execute_query(sparql)
            print(self.format_results(results))


if __name__ == "__main__":
    kg_system = KnowledgeGraphQuerySystem()
    kg_system.interactive_query()
