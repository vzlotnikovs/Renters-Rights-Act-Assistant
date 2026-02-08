import os
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

model = init_chat_model("gpt-5-nano")

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

tools = [retrieve_context]

prompt = (
    "You are an assistant answering questions ONLY about the UK Renters' Rights Act (applicable to England only).\n"
    "if the question is not related to the Renters' Rights Act, say you don't know and that you can only answer questions about the Renters' Rights Act.\n"
    "To ensure an accurate response, call the tool(s) available to you before answering a question.\n"
    "When possible, mention which part of the Act or section you are referring to.\n"
    "Do not answer questions that are not related to the Renters' Rights Act."
)
agent = create_agent(model, tools, system_prompt=prompt)

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