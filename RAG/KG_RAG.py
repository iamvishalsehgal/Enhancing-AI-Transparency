"""
RAG pipeline setup using Gemini LLM, Neo4j vector search, and Cypher-based graph queries.
"""

from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import warnings
from KG_query import chain
from KG_query import chain as cypher_qa_chain

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm = GoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME"),
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    node_label="*",  # Target all node labels in the graph
    text_node_properties=["*"],  # Use all node properties for embedding
    embedding_node_property="embedding"  # Store the embedding in this property
)

def retrieve_context(query):
    return vector_index.similarity_search(query, k=2)

tools = [
    Tool(
        name="GraphQuery",
        func=lambda q: chain.run({"query": q}),
        description="Answer structured questions using Neo4j graph queries"
    ),
    Tool(
        name="VectorSearch",
        func=retrieve_context,
        description="Find unstructured information from model cards"
    )
]


rag_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
