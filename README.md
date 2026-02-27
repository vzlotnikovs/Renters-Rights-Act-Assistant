Turing College AI Engineering Sprint 2 (Building Applications with LangChain & RAGs) Project

# Renters' Rights Act Assistant

The new Renters' Rights Act in England represents the biggest change to the English private rental sector in over 30 years. The Act aims to modernise the sector, giving greater security to tenants while also providing landlords with reliable, long-term occupancy of their property investment. These changes will apply to both new and existing tenancies.

This chatbot is designed to help both tenants & landlors navigate the new Act and answer any questions they may have.

## Workflow: 

1. **User Interaction**: Users ask questions about the Renters' Rights Act via the chatbot web interface (powered by Gradio).
2. **Agent Processing**: A LangChain-based agent receives the query and maintains conversation context using an in-memory checkpointer.
3. **Tool Execution**: The agent has three tools at its disposal to respond to the user's query:
   - *retrieve_context*: Searches a local Chroma vector database containing parsed text from the official Renters' Rights Act web guide and a PDF briefing by JLL, a leading property management agency.
   - *extract_notice_period*: Detects and parses notice periods (e.g., "2 months", "4 weeks", etc.) from the retrieved context.
   - *calculate_effective_date*: Computes effective action dates based on an initial date and the required notice period.
4. **Response Generation**: The agent synthesizes the tool outputs and presents a coherent, accurate response to the user.

## Features

- **Retrieval-Augmented Generation (RAG)**: Leverages OpenAI embeddings and Chroma DB to retrieve accurate, domain-specific context from the Renters' Rights Act.
- **Custom LangChain Tools**: Specialized functions relevant to the Renters' Rights Act.
- **Conversational Memory**: Retains session history to allow for follow-up questions and contextual dialogue.
- **Interactive UI**: Clean, easy-to-use chat interface built with Gradio.
- **Robust Error Handling**: Gracefully handles missing documents, network errors, and invalid user inputs.

## Validations

- **Environment Verification**: The application checks for the presence of required environment variables (like `OPENAI_API_KEY`) without exposing the actual values.
- **Document Loading Validations**: Verifies that web sources are reachable and PDF documents exist before attempting to process them.
- **Tool Input Validation**: The date calculation tool validates date string formats and ensures notice periods are valid integers.

## Optional Tasks Completed

- **Source citations in responses**: The agent includes citations to the sources of the information it provides. This is explicitly requested in the system prompt.

## Installation

### Prerequisites / Dependencies

- Python 3.11 - this Python version fully supports the Chroma database.
- An OpenAI API key. 
- See pyproject.toml for full list of dependencies.

### Steps

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install .
   ```

   Or if you are using `uv`:

   ```bash
   uv sync
   ```

3. Rename .env.example to .env and fill with your OpenAI API key.

## Usage

To start the application, run:

```bash
python main.py
```

Then click on the Gradio link generated in your terminal.

## Running Tests

To run the unit tests, execute:

```bash
pytest tests
```

## Type Checking

To run the type checking, execute:

```bash
mypy .
```

## Code Formatting

```bash
ruff format .
```
