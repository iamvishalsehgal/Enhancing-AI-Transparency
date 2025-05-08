"""
Interactive CLI for querying the RAG agent.
"""

from KG_RAG import rag_agent

print("RAG + Neo4j Query System (type 'exit' to quit)")

while True:
    question = input("\nQuestion: ")
    if question.strip().lower() in {"exit", "quit"}:
        print("Exiting.")
        break

    print("\nGenerating Cypher query")
    try:
        response = rag_agent.run(question)
        print("\nResponse:\n")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred: {e}")