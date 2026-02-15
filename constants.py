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

# Chatbot Prompt
CHATBOT_PROMPT = (
    "You are an assistant answering questions ONLY about the Renters' Rights Act (applicable to England only).\n"
    "if the question is not related to the Renters' Rights Act, say that you don't know and that you can only answer questions about the Renters' Rights Act.\n"
    "To ensure an accurate response, call some or all of the tools available to you before answering a question.\n"
    "Where appropriate, mention the source of the information (for example, part or section of the Act). \n"
    "Be concise and do not repeat yourself. Use bullet points where appropriate."
)

# Gradio Chat Interface Constants
PLACEHOLDER = "Ask me any question about the Renters' Rights Act"
TITLE = "Renters' Rights Act Assistant"
DESCRIPTION = "Ask me any question about the Renters' Rights Act"
EXAMPLES = [
    "What are the key changes introduced by the Renters' Rights Act?",
    "How long is the notice period for rent arrears?",
    "What happens to landlords if they don't comply with the Act?"
]