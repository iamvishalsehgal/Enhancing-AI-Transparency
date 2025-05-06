# MCRO Ontology Mapper

This repository contains a Python script that utilizes the [Model Card Report Ontology (MCRO)](https://github.com/UTHealth-Ontology/MCRO) to process Hugging Face model cards and generate structured RDF data. t is part of a larger thesis project focused on enhancing transparency and interoperability in machine learning model documentation through ontological frameworks. The script maps model card content to MCRO ontology concepts, producing RDF triples that can be used for further analysis or integration with knowledge graphs.

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

This code automates the extraction and structuring of information from Hugging Face model cards using the MCRO ontology. It generates standardized RDF data, enabling interoperability and facilitating the comparison, analysis and integration of model card information, contributing to the thesis project's goal of promoting transparency and accountability in AI systems.

## Dependencies

To run the script, install the following dependencies:

- Python 3.x
- `huggingface_hub`
- `rdflib`
- `google-generativeai`
- `python-dotenv`

Install them using pip:

```bash
pip install huggingface_hub rdflib google-generativeai python-dotenv
```

or 

```bash
pip install requirements.txt
```
You will also need API keys for Hugging Face and Google Gemini.

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/iamvishalsehgal/Enhancing-AI-Transparency.git
   cd Enhancing-AI-Transparency
   ```

2. **Rename `.env.copy` file to `.env` ** in the root directory and add your API keys:

   ```
   GEMINI_API_KEY=your_gemini_api_key
   HFTOKEN=your_huggingface_token
   ```

   Replace `your_gemini_api_key` and `your_huggingface_token` with your actual keys.

3. **Ensure the MCRO ontology file** (`mcro.ttl`) is in the `Ontology_mapper/Base_ontology/` directory. If not, download it from the [MCRO repository](https://github.com/UTHealth-Ontology/MCRO/tree/main) and place it there.

**Note**: You must have access to the Google Gemini API and comply with its terms of use, which may involve setting up billing or adhering to usage limits.

## Usage

Run the script to process the top 10 Hugging Face models by downloads:

```bash
python triple_creation.py
```

The script will:
1. Load the MCRO ontology into Google Gemini.
2. Retrieve model cards for the Hugging Face models by Downloads.
3. Generate RDF triples mapping model card content to MCRO concepts using Gemini.
4. Save the triples in JSON format to `Ontology_mapper/Output/<run_number>/triples.json`.
5. Convert the JSON triples to Turtle format and save to `Ontology_mapper/Output/<run_number>/triples.ttl`.

Note: To process a different number of models, modify the `all_triples = process_huggingface_models(limit=10)` parameter in the `process_huggingface_models` function.

## Output

Each run produces two files:
- `triples.json`: RDF triples in JSON format.
- `triples.ttl`: RDF triples in Turtle format.

These are saved in a new directory under `Ontology_mapper/Output/`, numbered incrementally (e.g., `1`, `2`, etc.).

### Loading the Turtle File

The Turtle file (triples.ttl) can be loaded into tools like [Protégé](https://protege.stanford.edu/) or [GraphDB](https://graphdb.ontotext.com) for analysis, querying or visualisations.

## Contributing

As this is part of a thesis project, contributions are limited but welcome. If you have suggestions for improvements or identify issues, please:
- Open an issue on the [main project repository](https://github.com/iamvishalsehgal/Enhancing-AI-Transparency.git).
- Follow the project's coding style and conventions.
- Provide clear and descriptive commit messages.
- Update documentation as necessary.

## License

This code is licensed under the [MIT License](LICENSE).

## Contact Information

For questions, issues, or further inquiries, please open an issue on the [main project repository](https://github.com/iamvishalsehgal/Enhancing-AI-Transparency.git).


