import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import bs4

# Step 0: Load environment variables from keys.env
load_dotenv(dotenv_path="../keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Load the raw data or documents
loader = WebBaseLoader(
    web_paths=("https://www.gov.uk/government/publications/guide-to-the-renters-rights-act/guide-to-the-renters-rights-act",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()

# Step 2: Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Step 3: Generate embeddings
# Use OpenAI Embeddings (requires OpenAI API key)
openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Step 4: Store embeddings in a vector database (FAISS in this case)
vector_store = FAISS.from_documents(chunks, openai_embeddings)
