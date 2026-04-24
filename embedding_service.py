"""
Сервис для работы с эмбедингами через API сервер
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import requests
import pickle
import os

class EmbeddingService:
    def __init__(self, 
                 api_url: str = "http://localhost:8000/v1/embeddings",
                 api_key: str = "sk-mysecretkey123",
                 model_name: str = "bge-embedding",
                 embedding_dim: int = 384):
        
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.cache = {}
        self.cache_file = "embeddings_cache.pkl"
        self._load_cache()
        
        print(f"✅ EmbeddingService инициализирован")
        print(f"   API URL: {self.api_url}")
        print(f"   Модель: {self.model_name}")
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"📦 Загружено {len(self.cache)} эмбедингов из кэша")
            except:
                self.cache = {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            else:
                print(f"❌ Ошибка API: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return []
    
    def encode_text(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        
        embeddings = self._call_api([text])
        if embeddings:
            embedding = np.array(embeddings[0])
            embedding = embedding / np.linalg.norm(embedding)
            self.cache[text] = embedding
            if len(self.cache) % 100 == 0:
                self._save_cache()
            return embedding
        
        return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        uncached = []
        indices = []
        embeddings = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings[i] = self.cache[text]
            else:
                uncached.append(text)
                indices.append(i)
        
        if uncached:
            new_embeddings = self._call_api(uncached)
            for i, idx in enumerate(indices):
                emb = np.array(new_embeddings[i])
                emb = emb / np.linalg.norm(emb)
                embeddings[idx] = emb
                self.cache[uncached[i]] = emb
        
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = np.zeros(self.embedding_dim)
        
        self._save_cache()
        return np.array(embeddings)
    
    def find_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = self.encode_text(query)
        candidate_embs = self.encode_batch(candidates)
        similarities = cosine_similarity([query_emb], candidate_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(candidates[i], float(similarities[i])) for i in top_indices]
    
    def search_medical_similar(self, query: str, medical_texts: List[str], top_k: int = 5) -> List[Tuple[str, float, int]]:
        results = self.find_similar(query, medical_texts, top_k)
        indexed = []
        for text, score in results:
            try:
                idx = medical_texts.index(text)
                indexed.append((text, score, idx))
            except:
                indexed.append((text, score, -1))
        return indexed