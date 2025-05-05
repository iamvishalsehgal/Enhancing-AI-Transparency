from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI  # For LLM
from langchain.chains import LLMChain  # For chain
from langchain.prompts import PromptTemplate  # For prompt
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Define a prompt template for the graph query chain
prompt = PromptTemplate.from_template("""
Answer the following question based on the knowledge graph:
{query}
""")

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Create vector store with updated parameters
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),  # ✅ New parameter
    password=os.getenv("NEO4J_PASSWORD"),  # ✅ New parameter
    node_label="Node",
    text_node_properties=["properties"],
    embedding_node_property="embedding"
)

# Retrieve context from graph
def retrieve_context(query):
    return vector_index.similarity_search(query, k=2)

# Define tools
tools = [
    Tool(
        name="GraphQuery",
        func=chain.invoke,  # Now chain is defined
        description="Answer questions about model architectures, use cases, and citations"
    ),
    Tool(
        name="VectorSearch",
        func=retrieve_context,
        description="Find unstructured information from the knowledge base"
    )
]

# Hybrid reasoning agent
rag_agent = initialize_agent(
    tools,
    llm,  # Now llm is defined
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)