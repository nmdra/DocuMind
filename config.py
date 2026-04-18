"""Shared configuration for the local FastMCP + ChromaDB agent."""

OLLAMA_BASE_URL = "http://localhost:11434"

CHAT_MODEL = "phi4-mini:3.8b-q4_K_M"
EMBED_MODEL = "embeddinggemma:300m-qat-q8_0"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
CONVERSATION_COLLECTION_NAME = "conversation_memory"
DEFAULT_CONVERSATION_SESSION_ID = "default"

TOP_K = 5
MAX_RESULT_PREVIEW_LENGTH = 400

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

SUPPORTED_INGEST_EXTENSIONS = (".txt", ".md", ".markdown")

DEFAULT_TRANSPORT = "stdio"
SSE_HOST = "127.0.0.1"
SSE_PORT = 8000
