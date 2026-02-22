"""Retrieval-Augmented Generation (RAG) module for the Renters' Rights assistant."""

import os
from typing import Union, Any, List, Tuple, Dict
from constants import (
    USER_AGENT,
    DOTENV_PATH,
    URL,
    PDF_FILENAME,
    SUB_DIR,
    LLM_MODEL,
    TAG,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_MODEL,
    COLLECTION_NAME,
    PERSIST_DIR,
    K_CONSTANT,
    CHATBOT_PROMPT,
)

os.environ["USER_AGENT"] = USER_AGENT

from dotenv import load_dotenv
from pathlib import Path
import bs4
import re
from datetime import date, timedelta
from dateutil import parser
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.runnables.config import RunnableConfig

load_dotenv(dotenv_path=DOTENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")


def load_source_content(
    SUB_DIR: str, PDF_FILENAME: str, URL: str, TAG: str
) -> List[Any]:
    """Load source content from web and PDF documents.

    Loads content from a specified URL and a PDF file, adding metadata to each
    source indicating its origin.

    Args:
        SUB_DIR (str): The subdirectory where the PDF is located.
        PDF_FILENAME (str): The filename of the PDF to load.
        URL (str): The URL of the web page to load.
        TAG (str): The HTML tag to extract from the web page.

    Returns:
        list: A list of Document objects loaded from the sources.

    Raises:
        RuntimeError: If document loading fails or network request fails.
    """
    try:
        BASE_DIR = Path(__file__).resolve().parent
        PDF_PATH = BASE_DIR / SUB_DIR / PDF_FILENAME

        bs4_strainer = bs4.SoupStrainer(TAG)
        web_loader = WebBaseLoader(
            web_paths=(URL,),
            bs_kwargs={"parse_only": bs4_strainer},
        )

        pdf_loader = PyPDFLoader(str(PDF_PATH))

        web_sources = web_loader.load()
        if not web_sources:
            raise RuntimeError("No web documents loaded")

        pdf_sources = pdf_loader.load()
        if not pdf_sources:
            raise RuntimeError("No PDF documents loaded")

        for source in web_sources:
            source.metadata["source"] = URL

        for source in pdf_sources:
            source.metadata["source"] = str(PDF_FILENAME)

        sources = web_sources + pdf_sources
        if len(sources) == 0:
            raise RuntimeError("Error loading source content")
        else:
            print("Source content loaded successfully.")
        return sources
    except ConnectionError as e:
        raise RuntimeError(f"[WARNING] Could not reach {URL}: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Request failed for {URL}: {e}")


sources = load_source_content(SUB_DIR, PDF_FILENAME, URL, TAG)


def create_vector_store(
    CHUNK_SIZE: int,
    CHUNK_OVERLAP: int,
    EMBEDDINGS_MODEL: str,
    COLLECTION_NAME: str,
    PERSIST_DIR: str,
) -> Chroma:
    """Create and persist a Chroma vector store from loaded documents.

    Splits the globally loaded source documents into chunks and stores them
    in a Chroma vector database using OpenAI embeddings.

    Args:
        CHUNK_SIZE (int): The maximum size of each text chunk.
        CHUNK_OVERLAP (int): The overlap size between chunks.
        EMBEDDINGS_MODEL (str): The name of the OpenAI embeddings model to use.
        COLLECTION_NAME (str): The name of the Chroma collection.
        PERSIST_DIR (str): The directory where the vector store will be saved.

    Returns:
        Chroma: The initialized and populated Chroma vector store.

    Raises:
        RuntimeError: If an error occurs during vector store creation.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(sources)

        embed_model = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL,
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embed_model,
            persist_directory=PERSIST_DIR,
        )

        vector_store.add_documents(documents=all_splits)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error creating vector store: {e}")


vector_store = create_vector_store(
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_MODEL, COLLECTION_NAME, PERSIST_DIR
)


@tool
def retrieve_context(query: str) -> str:
    """Retrieve information to help answer a query.

    Args:
        query (str): The user query.

    Returns:
        str: A formatted string containing retrieved documents with their sources.
    """
    retrieved_docs = vector_store.similarity_search(query, k=K_CONSTANT)
    bullet_points = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown")
        content = doc.page_content.replace("\n", " ")
        bullet_points.append(f"Source: {src}\n {content}")
    return "\n".join(bullet_points)


@tool
def extract_notice_period(query: str) -> str:
    """Extract all notice periods (in days) from retrieved Renters' Rights Act context.

    Automatically detects time periods like "2 months", "4 weeks", "120 days", etc.
    and converts them to days. Searches through retrieved documents from the vector
    store and extracts all matching notice periods grouped by source.

    Args:
        query (str): The user query.

    Returns:
        str: A formatted string containing extracted notice periods grouped by source,
            or "No notice periods found in retrieved context." if none are found.
            Format: "Extracted notice periods:\n\n- **{source}**: {periods}\n"
    """
    retrieved_sources = vector_store.similarity_search(query, k=K_CONSTANT)
    periods_by_source = {}

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:days?|day)",
        r"(\d+(?:\.\d+)?)\s*(?:weeks?|week)s?",
        r"(\d+(?:\.\d+)?)\s*(?:months?|month)s?",
    ]

    for source in retrieved_sources:
        src = source.metadata.get("source", "Unknown")
        text = source.page_content.lower()
        matches = []

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                num_str = match.group(1)
                num = int(num_str)

                days = 0
                full_match = match.group(0)

                if "day" in full_match:
                    days = num
                elif "week" in full_match:
                    days = num * 7
                elif "month" in full_match:
                    days = num * 30
                matches.append(f"{int(days)} days")

        if matches:
            periods_by_source[src] = list(set(matches))

    if not periods_by_source:
        return "No notice periods found in retrieved context."

    result = "Extracted notice periods:\n\n"
    for src, periods in periods_by_source.items():
        result += f"- **{src}**: {', '.join(periods)}\n"

    return result


@tool
def calculate_effective_date(
    notice_date: str, notice_period_days: str
) -> Union[Tuple[str, Dict[str, Any]], str]:
    """Calculate effective date based on notice date and notice period.

    Calculates the effective date (e.g., date to vacate the property, date when
    the new rent becomes effective, etc.) given a notice date and notice period
    from the Renters' Rights Act.

    Args:
        notice_date (str): The notice date in various formats. Accepts:
            - ISO format: "2026-02-08"
            - Relative: "today"
            - Other date formats: "1 Jan 2026", "08/02/2026", etc.
        notice_period_days (str): The notice period in days as a string (e.g., "30", "120").

    Returns:
        tuple or str: A tuple containing the effective date (str) and a dictionary
            with keys "date" and "days" on success. An error message string if the
            input is invalid or calculation fails.
    """
    try:
        if notice_date.lower() == "today":
            start_date = date.today()
        else:
            start_date = parser.parse(notice_date).date()

        days = int(notice_period_days)
        end_date = start_date + timedelta(days=days)
        effective_date = end_date.strftime("%Y-%m-%d")
        return effective_date, {"date": effective_date, "days": days}

    except ValueError as e:
        return f"Invalid input: {e}. Use format like '2026-02-08' or 'today' for dates, and use numbers for days."
    except Exception as e:
        return f"Calculation error: {e}"


checkpointer = InMemorySaver()

agent = create_agent(
    model=init_chat_model(LLM_MODEL),
    tools=[retrieve_context, extract_notice_period, calculate_effective_date],
    system_prompt=CHATBOT_PROMPT,
    checkpointer=checkpointer,
)


def renters_rights_assistant(query: str, thread_id: str) -> str:
    """Process a user query using the Renters' Rights assistant agent.

    Invokes the LangGraph agent with the provided query and thread ID to maintain
    conversation context. The agent uses tools to retrieve context, extract notice
    periods, and calculate effective dates from the Renters' Rights Act.

    Args:
        query (str): The user's question or query about renters' rights.
        thread_id (str): A unique identifier for the conversation thread to maintain
            context across multiple interactions.

    Returns:
        str: The assistant's response content as a string. This is extracted from
            the last message in the agent's response.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    return content
