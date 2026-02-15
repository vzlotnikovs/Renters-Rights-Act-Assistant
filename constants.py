"""Renters Rights Act Chatbot constants."""

# Env
DOTENV_PATH = "../keys.env"
USER_AGENT = "Renters Rights Bot (https://github.com/TuringCollegeSubmissions/vzlotn-AE.2.5)"

# Paths
URL = "https://www.gov.uk/government/publications/guide-to-the-renters-rights-act/guide-to-the-renters-rights-act"
PDF_FILENAME = "JLL-News-Renters-Rights-Act.pdf"
SUB_DIR = "Database"

# LLM Model
LLM_MODEL = "gpt-5-nano"
EMBEDDINGS_MODEL = "text-embedding-3-small"

# HTML Tags
TAG = "main" 

# Text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store
COLLECTION_NAME = "renters_rights"
PERSIST_DIR = "./chroma_langchain_db"
K_CONSTANT = 2

# Gradio Chat Interface Constants
PLACEHOLDER = "Ask me any question about the Renters' Rights Act"
TITLE = "Renters' Rights Act Assistant"
DESCRIPTION = "Ask me any question about the Renters' Rights Act"
EXAMPLES = [
    "What are the key changes introduced by the Renters' Rights Act?",
    "How long is the notice period for rent arrears?",
    "What happens to landlords if they don't comply with the Act?"
]