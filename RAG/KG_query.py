# Change the prompt file path below
# This script sets up a connection to a Neo4j database and uses the Google Generative AI model to generate Cypher queries for knowledge graph querying.
# It uses the LangChain library to facilitate the interaction between the LLM and the Neo4j graph database.


import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# Connect to Neo4j
graph = Neo4jGraph(
    uri=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Load prompt template from file
prompt_path = Path("RAG/prompts/prompt2.txt")
with open(prompt_path, "r") as f:
    CYPHER_GENERATION_TEMPLATE = f.read()

# Create prompt template
prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Create LangChain chain
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_prompt=prompt,
    llm=llm,
    verbose=True
)