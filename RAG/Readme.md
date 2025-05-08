Still in working the problem is that the agent is generating the wrong query format or missing some input keys. [In short prompt issue]

First run the `RAG.py` [Make sure to change the file path of extracted_triples.ttl]
Then Nodes and relationships will store in `RAG/Nodes-Rel` folder.
Run `Validate_relationship.py` to verify if all relationships have nodes and viceversa.
Then copy those extracted nodes and relationship.csv files to neo4j's import folder.

# Knowledge Graph + Gemini Agent

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` with your Gemini API key and Neo4j credentials
3. Place `nodes.csv` and `relationships.csv` in `data/` folder

## Run
```bash
python sample_query_runner.py

- KG_query.py
- KG_RAG.py
- Nodes-Rel/ 
- Prompts/
- Validate_relationships.py
- Readme.md
- RAG.py
- query_runner.py
