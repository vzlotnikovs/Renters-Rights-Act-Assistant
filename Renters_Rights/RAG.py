import os
import uuid
import bs4
from datetime import datetime, timedelta
from dateutil import parser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import AgentState, create_agent

load_dotenv(dotenv_path="../keys.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

chatbot_model = init_chat_model("gpt-5-nano")

URL = "https://www.gov.uk/government/publications/guide-to-the-renters-rights-act/guide-to-the-renters-rights-act"
K_CONSTANT = 2

bs4_strainer = bs4.SoupStrainer("main")
loader = WebBaseLoader(
    web_paths=(URL,),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()
if not docs:
    raise RuntimeError("No documents loaded")

print(f"Total characters: {len(docs[0].page_content)}")
if len(docs[0].page_content) == 0:
    raise RuntimeError("Error loading document content")

for d in docs:
    d.metadata["source"] = URL

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = Chroma(
    collection_name="renters_rights",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",
)

ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=K_CONSTANT)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def calculate_effective_date(notice_date: str, notice_period_days: str):
    """
    Calculate effective date (new rent start date, date to vacate the property, etc.) given notice date (notice_date) 
    and notice period (notice_period_days) from Renters' Rights Act.
    
    RAG determines notice_period_days based on different scenarios:
    - Rent increases
    - Possession grounds (e.g. landlord wants to sell property)
    - Other grounds
    
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
        return f"Invalid input: {e}. Use format like '2026-02-08' or 'today' for date, number for days."
    except Exception as e:
        return f"Calculation error: {e}"

prompt = (
    "You are an assistant answering questions ONLY about the Renters' Rights Act (applicable to England only).\n"
    "if the question is not related to the Renters' Rights Act, say that you don't know and that you can only answer questions about the Renters' Rights Act.\n"
    "To ensure an accurate response, call some or all of the tools available to you before answering a question.\n"
    "Where appropriate, mention which part of the Act or section you are referring to."
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=chatbot_model,
    tools=[retrieve_context, calculate_effective_date],
    system_prompt=prompt,
    checkpointer=checkpointer,
)

def renters_rights_chatbot(query: str, thread_id: str = None) -> str:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    last = result["messages"][-1]
    content = last.content if hasattr(last, 'content') else str(last)
    return content