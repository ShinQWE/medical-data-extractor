import re
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI библиотека не установлена")

from embedding_service import EmbeddingService
from config import Config

class DataExtractor:
    def __init__(self):
        # Загрузка конфигурации из config.py
        self.client = None
        self.model = None
        
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url=Config.LLM_API_URL,
                    api_key=Config.LLM_API_KEY
                )
                self.model = Config.LLM_MODEL_PATH
                print(f"✅ LLM (Qwen) инициализирован")
                print(f"   API URL: {Config.LLM_API_URL}")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации LLM: {e}")
                self.client = None
        else:
            print("⚠️ OpenAI библиотека не установлена, LLM будет недоступна")
        
        # Подключение к серверу эмбедингов
        try:
            print("🔧 Подключение к серверу эмбедингов...")
            self.embedder = EmbeddingService(
                api_url=Config.EMBEDDING_API_URL,
                api_key=Config.EMBEDDING_API_KEY,
                model_name=Config.EMBEDDING_MODEL_NAME,
                embedding_dim=Config.EMBEDDING_DIM
            )
            print("✅ Эмбединги доступны")
        except Exception as e:
            print(f"⚠️ Ошибка подключения к эмбедингам: {e}")
            self.embedder = None
        
        print("✅ DataExtractor инициализирован")
        print("="*50)
    
    def _call_qwen(self, prompt: str) -> str:
        """Отправляет запрос к Qwen через API"""
        if not self.client:
            return ""
        
        print("📤 Отправляем запрос к Qwen...")
        try:
            messages = [
                {"role": "system", "content": "Ты - медицинский эксперт. Отвечай только JSON, без пояснений."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            result = response.choices[0].message.content
            print(f"✅ Получили ответ от Qwen ({len(result)} символов)")
            return result
            
        except Exception as e:
            print(f"❌ Ошибка Qwen API: {e}")
            return ""
    
    async def discover_columns(self, texts: List[str], max_cols: int, domain_description: str = "") -> List[Dict]:
        """Автоматически определяет колонки с помощью LLM и эмбедингов"""
        
        print("\n" + "="*50)
        print("🚀 АВТОМАТИЧЕСКОЕ ФОРМИРОВАНИЕ БАЗЫ ЗНАНИЙ")
        print("="*50)
        print(f"📝 Описание области: {domain_description}")
        print(f"📊 Количество текстов для анализа: {len(texts)}")
        
        # Анализ с помощью эмбедингов для выявления паттернов
        if self.embedder and texts:
            try:
                print("🔍 Анализ текстов с помощью эмбедингов...")
                # Кластеризация для выявления тем
                from sklearn.cluster import KMeans
                embeddings = self.embedder.encode_batch(texts[:min(100, len(texts))])
                
                # Определяем количество кластеров
                n_clusters = min(5, len(embeddings) // 10 + 1)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(embeddings)
                    print(f"📊 Выявлено {n_clusters} смысловых кластеров в данных")
            except Exception as e:
                print(f"⚠️ Ошибка при анализе эмбедингов: {e}")
        
        # Формируем промпт для LLM
        sample_texts = texts[:5] if len(texts) > 5 else texts
        sample_text = "\n---\n".join(sample_texts)
        
        prompt = f"""
        На основе следующих медицинских текстов определи, какие числовые показатели должны быть извлечены.

        Описание предметной области: {domain_description}

        Примеры текстов:
        {sample_text}

        Определи ТОЛЬКО числовые параметры, которые могут быть в таких текстах (например: возраст, дозировка лекарства, размер образования, количество лимфоузлов, давление, пульс, показатели анализов и т.д.).

        Верни ответ строго в формате JSON:
        {{
            "columns": [
                {{"name": "Название_параметра", "description": "единицы измерения или пояснение"}},
                ...
            ]
        }}
        
        Максимум {max_cols} параметров.
        """
        
        response = self._call_qwen(prompt)
        
        columns = []
        if response:
            try:
                # Извлекаем JSON из ответа
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    columns = data.get("columns", [])
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка парсинга JSON: {e}")
                print(f"Ответ: {response[:500]}")
        
        # Если LLM не сработала, используем стандартные колонки
        if not columns:
            print("⚠️ LLM не вернула результат, используем стандартные колонки")
            columns = [
                {"name": "Возраст", "description": "лет"},
                {"name": "Дозировка_лекарства_мг", "description": "мг/сут"},
                {"name": "Размер_образования_мм", "description": "мм"},
                {"name": "Количество_лимфоузлов", "description": "штук"},
                {"name": "Размер_лимфоузла_см", "description": "см"},
                {"name": "Кровопотеря_мл", "description": "мл"},
                {"name": "Давление_систолическое", "description": "мм рт.ст."},
                {"name": "Пульс", "description": "уд/мин"},
                {"name": "Гемоглобин", "description": "г/л"},
                {"name": "Лейкоциты", "description": "10⁹/л"},
            ]
        
        columns = columns[:max_cols]
        print(f"\n✅ Сформировано {len(columns)} колонок:")
        for col in columns:
            print(f"   - {col['name']}: {col.get('description', '')}")
        
        return columns
    
    async def extract_values(self, text: str, columns: List[Dict], row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Извлекает числовые значения с помощью регулярных выражений"""
        result = {}
        text_lower = text.lower()
        
        # Возраст из колонки AGE если есть
        if row is not None and 'AGE' in row and pd.notna(row['AGE']):
            try:
                result["Возраст"] = float(row['AGE'])
            except:
                pass
        
        for col in columns:
            name = col["name"]
            
            if name in result and result[name] is not None:
                continue
            
            value = None
            
            if name == "Возраст" and not value:
                value = self._extract_age(text, text_lower)
            elif name == "Дозировка_лекарства_мг":
                value = self._extract_dosage(text, text_lower)
            elif name == "Размер_образования_мм":
                value = self._extract_size_mm(text, text_lower)
            elif name == "Количество_лимфоузлов":
                value = self._extract_lymph_nodes_count(text, text_lower)
            elif name == "Размер_лимфоузла_см":
                value = self._extract_lymph_node_size(text, text_lower)
            elif name == "Кровопотеря_мл":
                value = self._extract_blood_loss(text, text_lower)
            elif name == "Давление_систолическое":
                value = self._extract_pressure(text, text_lower)
            elif name == "Пульс":
                value = self._extract_pulse(text, text_lower)
            elif name in ["Гемоглобин", "Лейкоциты", "Тромбоциты", "АСТ", "АЛТ", "Билирубин", "Креатинин"]:
                value = self._extract_lab_value(text, text_lower, name)
            else:
                # Общий поиск чисел с единицами измерения
                value = self._extract_generic_value(text, text_lower, name)
            
            if value is not None:
                result[name] = value
        
        return result
    
    def _extract_age(self, text: str, text_lower: str) -> Optional[float]:
        patterns = [r'(\d+)\s*лет', r'age[:\s]*(\d+)', r'возраст[:\s]*(\d+)']
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def _extract_dosage(self, text: str, text_lower: str) -> Optional[float]:
        patterns = [r'(\d+[.,]?\d*)\s*(?:мг|mg)', r'преднизолон\s*(\d+[.,]?\d*)']
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(match.group(1).replace(',', '.'))
                except:
                    pass
        return None
    
    def _extract_size_mm(self, text: str, text_lower: str) -> Optional[float]:
        match = re.search(r'(\d+)\s*[хx]\s*\d+\s*(?:мм|mm)', text_lower)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+[.,]?\d*)\s*(?:мм|mm)', text_lower)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except:
                pass
        return None
    
    def _extract_lymph_nodes_count(self, text: str, text_lower: str) -> Optional[float]:
        if re.search(r'лимфоузл.*\d+[.,]?\d*\s*(?:см|mm|мм)', text_lower):
            return None
        match = re.search(r'лимфоузл[а-я]*\s+(\d+)', text_lower)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_lymph_node_size(self, text: str, text_lower: str) -> Optional[float]:
        match = re.search(r'лимфоузл[а-я]*\s*(\d+[.,]?\d*)\s*(?:см|cm)', text_lower)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except:
                pass
        return None
    
    def _extract_blood_loss(self, text: str, text_lower: str) -> Optional[float]:
        match = re.search(r'кровопотер[яи][:\s]*(\d+)\s*(?:мл|ml)', text_lower)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_pressure(self, text: str, text_lower: str) -> Optional[float]:
        match = re.search(r'(\d{2,3})\s*[\/\-]\s*(\d{2,3})', text_lower)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_pulse(self, text: str, text_lower: str) -> Optional[float]:
        match = re.search(r'пульс[^\d]*(\d{2,3})', text_lower)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_lab_value(self, text: str, text_lower: str, param_name: str) -> Optional[float]:
        param_lower = param_name.lower()
        match = re.search(rf'{param_lower}[^\d]*(\d+[.,]?\d*)', text_lower)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except:
                pass
        return None
    
    def _extract_generic_value(self, text: str, text_lower: str, param_name: str) -> Optional[float]:
        """Общий поиск чисел в контексте параметра"""
        param_lower = param_name.lower()
        # Ищем число после названия параметра
        match = re.search(rf'{param_lower}[^\d]*(\d+[.,]?\d*)', text_lower)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except:
                pass
        return None