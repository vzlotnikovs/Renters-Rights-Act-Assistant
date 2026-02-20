import os
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
from datetime import datetime, timedelta
from dateutil import parser
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

load_dotenv(dotenv_path=DOTENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

chatbot_model = init_chat_model(LLM_MODEL)

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / SUB_DIR / PDF_FILENAME

bs4_strainer = bs4.SoupStrainer(TAG)
web_loader = WebBaseLoader(
    web_paths=(URL,),
    bs_kwargs={"parse_only": bs4_strainer},
)

pdf_loader = PyPDFLoader(str(PDF_PATH))

web_docs = web_loader.load()
if not web_docs:
    raise RuntimeError("No web documents loaded")

pdf_docs = pdf_loader.load()
if not pdf_docs:
    raise RuntimeError("No PDF documents loaded")

for doc in web_docs:
    doc.metadata["source"] = URL

for doc in pdf_docs:
    doc.metadata["source"] = str(PDF_FILENAME)

print(f"Total characters (web source): {len(web_docs[0].page_content)}")
print(f"Total characters (PDF source): {len(pdf_docs[0].page_content)}")

docs = web_docs + pdf_docs
if len(docs) == 0:
    raise RuntimeError("Error loading source content")
else:
    print("Source content loaded successfully.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

embed_model = OpenAIEmbeddings(
    model=EMBEDDINGS_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embed_model,
    persist_directory=PERSIST_DIR,
)

ids = vector_store.add_documents(documents=all_splits)

@tool
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=K_CONSTANT)
    bullet_points = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown")
        content = doc.page_content.replace("\n", " ")
        bullet_points.append(f"Source: {src}\n {content}")
    return "\n".join(bullet_points)

@tool
def extract_notice_period(query: str) -> str:
    """
    Extract all notice periods (in days) from retrieved Renters' Rights Act context.
    
    Automatically detects "2 months", "4 weeks", "120 days", etc. and converts to days.
    Returns: List of periods with sources (or "No periods found").
    """
    retrieved_docs = vector_store.similarity_search(query, k=K_CONSTANT)
    periods_by_source = {}
    
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:days?|day)',
        r'(\d+(?:\.\d+)?)\s*(?:weeks?|week)s?',
        r'(\d+(?:\.\d+)?)\s*(?:months?|month)s?',
    ]

    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown")
        text = doc.page_content.lower()        
        matches = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                num_str = match.group(1)
                num = float(num_str)

                days = 0
                full_match = match.group(0)

                if 'day' in full_match:
                    days = num
                elif 'week' in full_match:
                    days = num * 7
                elif 'month' in full_match:
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
def calculate_effective_date(notice_date: str, notice_period_days: str):
    """
    Calculate effective date (e.g. date to vacate the property, date when the new rent becomes effective, etc.) given notice date (notice_date)
    and notice period (notice_period_days) from Renters' Rights Act.

    Args:
        notice_date: Notice date ("2026-02-08", "today", "1 Jan 2026", "08/02/2026")
        notice_period_days: Days as string from RAG ("30", "120", etc.)
    """
    try:
        if notice_date.lower() == "today":
            start_date = datetime.now()
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

prompt = CHATBOT_PROMPT

checkpointer = InMemorySaver()

agent = create_agent(
    model=chatbot_model,
    tools=[retrieve_context, extract_notice_period, calculate_effective_date],
    system_prompt=prompt,
    checkpointer=checkpointer,
)

def renters_rights_assistant(query: str, thread_id: str = None) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    return content