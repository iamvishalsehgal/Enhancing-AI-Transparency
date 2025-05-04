# This code is used to generate embeddings for each node's description

import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import google.generativeai as genai
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Neo4j Connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def create_vector_index():
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 768,
              `vector.similarity_function`: 'cosine'
            }}
            """)

def get_embedding(text: str) -> list:
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return None

def update_embeddings(batch_size=100):
    create_vector_index()
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) WHERE n.description IS NOT NULL RETURN n.id AS id, n.description AS desc")
        batch = []
        for record in result:
            embedding = get_embedding(record["desc"])
            if embedding:
                batch.append({
                    "id": record["id"],
                    "embedding": embedding
                })
                if len(batch) >= batch_size:
                    session.run(
                        """UNWIND $batch AS row
                        MATCH (n:Entity {id: row.id})
                        SET n.embedding = row.embedding""",
                        {"batch": batch}
                    )
                    batch = []
        
        if batch:  # Process remaining records
            session.run(
                """UNWIND $batch AS row
                MATCH (n:Entity {id: row.id})
                SET n.embedding = row.embedding""",
                {"batch": batch}
            )

if __name__ == "__main__":
    try:
        update_embeddings()
    finally:
        driver.close()