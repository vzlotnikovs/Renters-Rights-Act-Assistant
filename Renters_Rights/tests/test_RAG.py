import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from Renters_Rights.RAG import (
    load_source_content,
    create_vector_store,
    retrieve_context,
    extract_notice_period,
    calculate_effective_date,
    renters_rights_assistant
)

@patch('Renters_Rights.RAG.WebBaseLoader')
@patch('Renters_Rights.RAG.PyPDFLoader')
@patch('Renters_Rights.RAG.bs4.SoupStrainer')
def test_load_source_content(mock_soup_strainer, mock_pdf_loader, mock_web_loader):
    """Test loading content from web URL and PDF loader."""
    mock_web_instance = MagicMock()
    mock_web_instance.load.return_value = [Document(page_content="dummy web content", metadata={"source": "web"})]
    mock_web_loader.return_value = mock_web_instance
    
    mock_pdf_instance = MagicMock()
    mock_pdf_instance.load.return_value = [Document(page_content="dummy pdf content", metadata={"source": "pdf"})]
    mock_pdf_loader.return_value = mock_pdf_instance
    
    sources = load_source_content("dummy_dir", "dummy.pdf", "http://dummy.com", "div")
    
    assert len(sources) == 2
    mock_web_loader.assert_called_once()
    mock_pdf_loader.assert_called_once()

@patch('Renters_Rights.RAG.Chroma')
@patch('Renters_Rights.RAG.OpenAIEmbeddings')
@patch('Renters_Rights.RAG.RecursiveCharacterTextSplitter')
def test_create_vector_store(mock_splitter, mock_embeddings, mock_chroma):
    """Test creating vector store with documents splitting and embeddings."""
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = ["doc1", "doc2"]
    mock_splitter.return_value = mock_splitter_instance
    
    mock_chroma_instance = MagicMock()
    mock_chroma.return_value = mock_chroma_instance
    
    store = create_vector_store(100, 10, "dummy_model", "dummy_collection", "dummy_dir")
    
    assert store == mock_chroma_instance
    mock_chroma_instance.add_documents.assert_called_once_with(documents=["doc1", "doc2"])

@patch('Renters_Rights.RAG.vector_store')
def test_retrieve_context(mock_vector_store):
    """Test context retrieval from query similarity search."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test_src"}
    mock_doc.page_content = "test content\nwith newline"
    mock_vector_store.similarity_search.return_value = [mock_doc]
    
    result = retrieve_context.invoke({"query": "test query"})
    assert "Source: test_src" in result
    assert "test content with newline" in result

@patch('Renters_Rights.RAG.vector_store')
def test_extract_notice_period(mock_vector_store):
    """Test extracting notice period returns text based on finding time patterns."""
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test_src"}
    mock_doc.page_content = "The notice period is 2 months."
    mock_vector_store.similarity_search.return_value = [mock_doc]
    
    result = extract_notice_period.invoke({"query": "notice period"})
    assert "Extracted notice periods:" in result
    assert "60 days" in result

def test_calculate_effective_date():
    """Test effective date calculations handles varying inputs and errors correctly."""
    result, data = calculate_effective_date.invoke({"notice_date": "2026-10-08", "notice_period_days": "30"})
    assert result == "2026-11-07"
    assert data["date"] == "2026-11-07"
    assert data["days"] == 30
    
    invalid_result = calculate_effective_date.invoke({"notice_date": "invalid-date", "notice_period_days": "30"})
    assert "Invalid input:" in invalid_result

@patch('Renters_Rights.RAG.agent')
def test_renters_rights_assistant(mock_agent):
    """Test that the chatbot invocation gets standard message response from the agent model."""
    mock_last_message = MagicMock()
    mock_last_message.content = "This is a dummy response."
    mock_agent.invoke.return_value = {"messages": [mock_last_message]}
    
    result = renters_rights_assistant("What are my rights?", "thread_123")
    assert result == "This is a dummy response."
    mock_agent.invoke.assert_called_once()
