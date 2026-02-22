import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

os.environ["OPENAI_API_KEY"] = "fake-key"
os.environ["USER_AGENT"] = "fake-agent"

from langchain_core.documents import Document

patch('langchain_community.document_loaders.WebBaseLoader.load', return_value=[Document(page_content="web content", metadata={"source": "web"})]).start()
patch('langchain_community.document_loaders.PyPDFLoader.load', return_value=[Document(page_content="pdf content", metadata={"source": "pdf"})]).start()
patch('langchain_chroma.Chroma.add_documents').start()
patch('langchain.chat_models.init_chat_model', return_value=MagicMock()).start()
patch('langchain.agents.create_agent', return_value=MagicMock()).start()
