import argparse
import os
import torch
import torch.nn.functional as F
from typing import List, Union, Optional
import base64

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import uvicorn

# Используем sentence-transformers (стабильный бэкенд)e
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embeddings Server")

# Глобальные переменные e
model = None
MODEL_NAME = None
EMBEDDING_DIM = None

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-ada-002"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict

def get_api_key(authorization: Optional[str] = Header(None, alias="Authorization")) -> bool:
    required_key = os.getenv("API_KEY")
    if not required_key:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    provided_key = authorization.split("Bearer ")[1].strip()
    if provided_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, authorized: bool = Depends(get_api_key)):
    global model, MODEL_NAME
    
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Получаем эмбединги
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    embedding_list = embeddings.tolist()
    
    # Применяем ограничение dimensions
    if request.dimensions is not None:
        dim = min(request.dimensions, len(embedding_list[0]) if embedding_list else 0)
        for i in range(len(embedding_list)):
            embedding_list[i] = embedding_list[i][:dim]
    
    data = [EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embedding_list)]
    
    total_tokens = len(texts) * 512  # приблизительно
    
    response = EmbeddingResponse(
        data=data,
        model=MODEL_NAME,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )
    
    if request.encoding_format == "base64":
        for item in response.data:
            arr = torch.tensor(item.embedding, dtype=torch.float32).numpy()
            item.embedding = base64.b64encode(arr.tobytes()).decode("utf-8")
    
    return response

@app.get("/v1/models")
async def list_models(authorized: bool = Depends(get_api_key)):
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 1740000000, "owned_by": "local"}]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embeddings Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--model-name", type=str, default="bge-embedding")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["API_KEY"] = args.api_key
        print(f"✅ API-ключ установлен")
    
    MODEL_NAME = args.model_name
    
    print(f"🚀 Запуск сервера эмбеддингов")
    print(f"   Модель: {args.model_path}")
    print(f"   Имя в API: {MODEL_NAME}")
    print(f"   Устройство: {args.device}")
    
    # Загружаем модель
    print("📥 Загрузка модели...")
    model = SentenceTransformer(args.model_path, device=args.device)
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    
    print(f"✅ Модель загружена!")
    print(f"   Размер эмбеддинга: {EMBEDDING_DIM}")
    print(f"   Сервер: http://{args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)