import re
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional

# Проверка наличия OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI библиотека не установлена")

from embedding_service import EmbeddingService

class DataExtractor:
    def __init__(self):
        # Настройки для Qwen через API (только если библиотека есть)
        self.client = None
        self.model = None
        
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url="https://aichat.iacpaas.dvo.ru/api",
                    api_key="sk-1e4b3879f93a4c5d88380aceff94d0ad"
                )
                self.model = "/home/atarasov/LLM/base_models/Qwen--Qwen3.5-27B-FP8"
                print("✅ LLM (Qwen) инициализирован")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации LLM: {e}")
                self.client = None
        else:
            print("⚠️ OpenAI библиотека не установлена, LLM будет недоступна")
        
        # Подключение к серверу эмбедингов
        try:
            print("🔧 Подключение к серверу эмбедингов...")
            self.embedder = EmbeddingService(
                api_url="http://localhost:8000/v1/embeddings",
                api_key="sk-mysecretkey123",
                model_name="bge-embedding",
                embedding_dim=384
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
        """Определяет колонки"""
        
        print("\n" + "="*50)
        print("🚀 ОПРЕДЕЛЕНИЕ КОЛОНОК")
        print("="*50)
        print(f"📝 Описание области: {domain_description}")
        print(f"📊 Количество текстов: {len(texts)}")
        
        # Используем эмбединги для анализа
        if self.embedder and texts:
            try:
                print("🔍 Анализ текстов с помощью эмбедингов...")
                clusters = self.embedder.cluster_texts(texts[:50], threshold=0.6)
                print(f"📊 Найдено {len(clusters)} кластеров")
            except Exception as e:
                print(f"⚠️ Ошибка эмбедингов: {e}")
        
        # Фиксированный список колонок
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
            {"name": "Тромбоциты", "description": "10⁹/л"},
            {"name": "АСТ", "description": "Ед/л"},
            {"name": "АЛТ", "description": "Ед/л"},
            {"name": "Билирубин", "description": "мкмоль/л"},
            {"name": "Креатинин", "description": "мкмоль/л"},
        ]
        
        print(f"✅ Используем {len(columns)} колонок")
        for col in columns[:10]:
            print(f"   - {col['name']}: {col['description']}")
        
        return columns[:max_cols]
    
    async def extract_values(self, text: str, columns: List[Dict], row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Извлекает числовые значения"""
        result = {}
        text_lower = text.lower()
        
        # Возраст из колонки AGE
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
            
            if name == "Дозировка_лекарства_мг":
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
            
            if value is not None:
                result[name] = value
        
        return result
    
    # Методы извлечения
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


if __name__ == "__main__":
    import asyncio
    async def test():
        extractor = DataExtractor()
        test_text = "преднизолон 5мг/сут, лимфоузлы 0,5см, кровопотеря 50 мл"
        columns = await extractor.discover_columns([test_text], 15, "Медицинские карты")
        values = await extractor.extract_values(test_text, columns, None)
        print(f"\n📊 Результат: {values}")
    asyncio.run(test())