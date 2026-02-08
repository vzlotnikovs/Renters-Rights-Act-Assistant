import os
from datetime import datetime, timedelta
from dateutil import parser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import AgentState, create_agent
import bs4

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

@tool
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=K_CONSTANT)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def calculate_effective_date(input_date: str, notice_period_days: str):
    """
    Calculate effective date (new rent start, vacate date, etc.) as an output given notice date (input_date) 
    and notice period (notice_period_days) from Renters' Rights Act.
    
    RAG determines notice_period_days based on different scenarios:
    - Rent increases: 2 months
    - Possession (landlord wants to sell property, >12mo tenancy): 4 months
    - Other grounds: 2 weeks, 1 month, 2 months, etc. as per RAG
    
    Args:
        input_date: Notice date ("2026-02-08", "today", "1 Jan 2026", "08/02/2026")
        notice_period_days: Days as string from RAG ("30", "120", etc.)
    """
    try:
        if input_date.lower() == "today":
            start_date = datetime.now()
        else:
            start_date = parser.parse(input_date).date()
        
        days = int(notice_period_days)
        end_date = start_date + timedelta(days=days)
        return end_date.strftime("%Y-%m-%d")
        
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

agent = create_agent(
    model=chatbot_model,
    tools=[retrieve_context, calculate_effective_date],
    system_prompt=prompt,
)

def renters_rights_chatbot(query: str) -> str:
    """Run the agent on a single-turn user query and return the final text reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # `result["messages"][-1]` should be the assistant message object
    last = result["messages"][-1]
    # Adapt depending on your message type; often `.content` or `.text`
    content = getattr(last, "content", None) or getattr(last, "text", "")
    return content

'''
Example query
query = "What is the notice period for evicting a tenant assuming 1A grounds (sale of dwelling-house)?"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
'''