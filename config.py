import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    LLM_API_URL = os.getenv("LLM_API_URL", "https://aichat.iacpaas.dvo.ru/api")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-1e4b3879f93a4c5d88380aceff94d0ad")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/home/atarasov/LLM/base_models/Qwen--Qwen3.5-27B-FP8")
    
    # Embedding Configuration
    EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000/v1/embeddings")
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-mysecretkey123")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-embedding")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
    
    # Server Configuration
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8080"))
    
    # Files
    KNOWLEDGE_BASE_FILE = os.getenv("KNOWLEDGE_BASE_FILE", "knowledge_base.json")
    
    # Extraction Settings
    MAX_COLUMNS = int(os.getenv("MAX_COLUMNS", "20"))
    CLUSTER_THRESHOLD = float(os.getenv("CLUSTER_THRESHOLD", "0.6"))