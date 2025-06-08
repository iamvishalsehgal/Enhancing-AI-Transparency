# Knowledge Graph Query System

This repository contains a Python script that enables natural language querying of a knowledge graph using the [Model Card Report Ontology (MCRO)](https://github.com/UTHealth-Ontology/MCRO). It is a subcomponent of a larger thesis project focused on enhancing transparency and accessibility in machine learning model documentation through knowledge graph technologies. The script translates natural language questions into SPARQL queries, executes them against a GraphDB endpoint, and formats the results for user-friendly output.

Note: This is a subdirectory of the main thesis project repository. For additional context, resources, or related components, please refer to the main project repository.

## Table of Contents

- [Purpose](#purpose)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Purpose

This code facilitates intuitive access to structured data in a knowledge graph by allowing users to ask questions in natural language. It leverages the Gemini generative AI model to convert questions into SPARQL queries, which are executed against a GraphDB instance containing MCRO-based model card data. This contributes to the thesis project's goal of making AI model documentation more accessible and interoperable for non-technical users.

## Dependencies

To run the script, install the following dependencies:

- Python 3.8+
- `requests`
- `python-dotenv`
- `google-generativeai`

Install them using pip:

```bash
pip install requests python-dotenv google-generativeai
```

or

```bash
pip install -r requirements.txt
```

You will also need:
- A GraphDB instance with a repo and knowledge graph schema.
- A Gemini API key for query generation.

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/iamvishalsehgal/Enhancing-AI-Transparency.git
   cd Enhancing-AI-Transparency
   ```

2. **Rename `.env.copy` file to `.env`** in the root directory and add your API keys:

   ```
   GRAPHDB_ENDPOINT=your_graphdb_endpoint
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL_NAME=your_gemini_model_name
   ```

   Replace `your_graphdb_endpoint`, `your_gemini_api_key`, and `your_gemini_model_name` with your actual values.

3. **Ensure the GraphDB instance** is running and accessible with MCRO-based data loaded.

**Note**: You must have access to the Google Gemini API and comply with its terms of use, which may involve setting up billing or adhering to usage limits.

## Usage

Run the script to start the interactive query system:

```bash
python KGQA.py
```

The script will:
1. Prompt for a natural language question (e.g., "What models have NSFW image classification use cases?").
2. Use Gemini to generate a SPARQL query based on the question and MCRO schema.
3. Execute the query against the GraphDB endpoint.
4. Display the generated SPARQL query and formatted results.
5. Continue prompting until the user types `exit` or `quit`.

## Output

The script outputs:
- The generated SPARQL query for each question.
- Formatted query results, showing entities and their properties (e.g., model names and use case values).
- Error messages if query generation or execution fails.

### Actual Output by using [Ontology_mapper](Ontology_mapper) prompt-1.

```
Question: What are the use cases for the Falconsainsfwimagedetection model?
Generating SPARQL query

Generated SPARQL Query:
PREFIX mcro: <http://purl.obolibrary.org/obo/MCRO_>
PREFIX prov1: <https://www.w3.org/ns/prov#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?useCaseText
WHERE {
  mcro:Falconsainsfwimagedetection mcro:hasUseCase ?useCase .
  ?useCase prov1:hasTextValue ?useCaseText .
}

Executing query
Found 2 result(s):
NSFW Image Classification
- **NSFW Image Classification**: The primary intended use of this model is for the classification of NSFW (Not Safe for Work) images. It has been fine-tuned for this purpose, making it suitable for filtering explicit or inappropriate content in various applications.

```

## Contributing

As this is part of a thesis project, contributions are limited but welcome. If you have suggestions for improvements or identify issues, please:
- Open an issue on the [main project repository](https://github.com/iamvishalsehgal/Enhancing-AI-Transparency/issues).
- Follow the project's coding style and conventions.
- Provide clear and descriptive commit messages.
- Update documentation as necessary.

## License

This code is licensed under the [MIT License](license).

## Contact Information

For questions, issues, or further inquiries, please open an issue on the [main project repository](https://github.com/iamvishalsehgal/Enhancing-AI-Transparency/issues).