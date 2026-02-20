# Renters' Rights Act Assistant

The upcoming introduction of the Renters' Rights Act sees the biggest shake-up to the private rental sector in England for almost 40 years. It aims to modernise and balance the sector giving greater security to tenants, and in turn provide landlords with reliable, long-term occupancy of their property investment. These changes will apply to both new and existing tenancies.

This chatbot is designed to help both tenants & landlors navigate the new act and answer any questions they may have.

## Workflow: 

TBD

## Features

TBD

## Validations
TBD

## Optional Tasks Completed

- TBC

## Installation

### Prerequisites / Dependencies

- Python 3.11 - full support of Chroma database.
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