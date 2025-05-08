"""
This script connects to a Neo4j database and uses Gemini Model
(via LangChain) to generate Cypher queries for querying a knowledge graph.
"""

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

llm = GoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

CYPHER_GENERATION_TEMPLATE = """
You are a Cypher expert working with a Neo4j knowledge graph.

All entities are represented as nodes with the label 'Node'. Each node has:
- id: a unique model identifier (e.g., "mcro_resnet50a1in1k")
- label: a semicolon-separated string of type tags (e.g., "NamedIndividual;mcro_Model")
- embedding: a vector for similarity search

Important: 
- The `label` property is crucial for identifying the type of node.
- Model nodes are identified when the `label` contains the string 'mcro_Model'.
- Do not use a fixed label like `:Model` in your queries.

To retrieve all models, use the following structure:
MATCH (n:Node) WHERE n.label CONTAINS 'mcro_Model' RETURN n.id AS model

Given a question, write the correct Cypher query using this structure.
Question: {query}
Cypher Query:
"""


prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_prompt=prompt,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True
)

__all__ = ['chain']
