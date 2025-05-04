#main script for combining:  Vector search (Gemini embeddings),  Graph traversal (Neo4j relationships), LLM generation (Gemini)

import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import google.generativeai as genai

load_dotenv()

# Configure connections
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

class VectorGraphRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # Get query embedding
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result["embedding"]
            
            # Fixed Cypher query (removed comment)
            cypher_query = """
            CALL db.index.vector.queryNodes('index_86f244aa', $k, $embedding)
            YIELD node, score
            OPTIONAL MATCH path = (node)-[*1..2]-(related)
            RETURN 
                coalesce(related.description, related.name, node.description, node.name) AS text,
                score,
                labels(node) AS labels,
                [r IN relationships(path) | type(r)] AS relationships
            ORDER BY score DESC
            LIMIT $k
            """
            
            results = graph.query(
                cypher_query,
                params={"embedding": query_embedding, "k": 5}
            )
            
            return [
                Document(
                    page_content=r["text"],
                    metadata={
                        "score": r["score"],
                        "labels": r["labels"],
                        "relationships": r["relationships"]
                    }
                ) for r in results if r["text"] is not None
            ]
        except Exception as e:
            print(f"Retrieval failed: {str(e)}")
            return []

# Updated LLM initialization with latest model
llm = GoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=VectorGraphRetriever(),
    return_source_documents=True
)

def query_system(question: str):
    try:
        response = qa_chain.invoke({"query": question})
        if response["result"]:
            print(f"Answer: {response['result']}\n")
            print("Sources:")
            for doc in response["source_documents"]:
                print(f"- {doc.page_content} (Score: {doc.metadata['score']:.2f})")
        else:
            print("No relevant results found")
    except Exception as e:
        print(f"Query failed: {str(e)}")

if __name__ == "__main__":
    while True:
        question = input("\nEnter your question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        query_system(question)