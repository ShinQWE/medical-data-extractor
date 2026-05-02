#!/usr/bin/env python3
"""
Сервер эмбедингов для MedExtractor
Запуск: python embedding_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Server")

# Глобальная модель
model = None

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str = "bge-embedding"
    encoding_format: str = "float"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Загрузка модели эмбедингов...")
    # Используем легковесную модель
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Модель загружена!")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    global model
    
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Модель не загружена"})
    
    try:
        # Генерируем эмбединги
        embeddings = model.encode(request.input, normalize_embeddings=True)
        
        # Формируем ответ
        data = []
        for i, emb in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": emb.tolist()
            })
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in request.input),
                "total_tokens": sum(len(text.split()) for text in request.input)
            }
        )
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    # Установка устройства
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Запуск сервера эмбедингов на {args.host}:{args.port}")
    logger.info(f"Устройство: {device}")
    
    uvicorn.run(app, host=args.host, port=args.port)