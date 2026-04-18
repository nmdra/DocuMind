"""Shared configuration for the local FastMCP + ChromaDB agent."""

OLLAMA_BASE_URL = "http://localhost:11434"

CHAT_MODEL = "phi4-mini:3.8b-q4_K_M"
EMBED_MODEL = "embeddinggemma:300m-qat-q8_0"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

TOP_K = 5

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
