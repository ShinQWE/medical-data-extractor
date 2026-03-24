import re
import pandas as pd
import json
import os
import requests
from typing import List, Dict, Any, Optional

class DataExtractor:
    def __init__(self):
        # Настройки
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:1b"
        
        # Словарь синонимов
        self.synonyms = {
            "вес": ["вес", "масса", "масса тела", "weight"],
            "возраст": ["возраст", "лет", "год", "age"],
            "рост": ["рост", "длина", "height"],
            "давление": ["давление", "ад", "артериальное давление", "pressure"],
            "пульс": ["пульс", "чсс", "heart rate"],
            "температура": ["температура", "t", "temperature"],
            "лейкоциты": ["лейкоциты", "wbc", "white blood"],
            "гемоглобин": ["гемоглобин", "hb", "hgb", "hemoglobin"],
            "тромбоциты": ["тромбоциты", "plt", "platelets"],
            "аст": ["аст", "ast", "аспартатаминотрансфераза"],
            "алт": ["алт", "alt", "аланинаминотрансфераза"],
            "длительность": ["длительность", "продолжительность", "течение", "duration"],
        }
        
        # Веса важности
        self.importance_weights = {
            "возраст": 100,
            "вес": 90,
            "давление": 85,
            "пульс": 80,
            "температура": 80,
            "лейкоциты": 75,
            "гемоглобин": 75,
            "тромбоциты": 70,
            "аст": 70,
            "алт": 70,
            "длительность": 65,
        }
        
        print("✅ DataExtractor инициализирован")
        print(f"🤖 Используем модель: {self.model}")
        print("="*50)
    
    def _call_ollama(self, prompt: str) -> str:
        """Отправляет запрос в Ollama"""
        print("📤 Отправляем запрос в Ollama...")
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()["response"]
                print(f"✅ Получили ответ ({len(result)} символов)")
                return result
            else:
                print(f"❌ Ошибка Ollama: {response.status_code}")
                return ""
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return ""
    
    def _normalize_column_name(self, name: str) -> str:
        """Нормализует название колонки"""
        name_lower = name.lower().strip()
        for canonical, variants in self.synonyms.items():
            if name_lower in variants or any(v in name_lower for v in variants):
                return canonical.capitalize()
        return name
    
    def _merge_columns(self, columns: List[Dict]) -> List[Dict]:
        """Объединяет похожие колонки"""
        normalized = {}
        for col in columns:
            original_name = col["name"]
            normalized_name = self._normalize_column_name(original_name)
            
            if normalized_name not in normalized:
                normalized[normalized_name] = {
                    "name": normalized_name,
                    "description": col["description"],
                    "original_names": [original_name],
                    "importance": self.importance_weights.get(
                        normalized_name.lower(), 50
                    )
                }
            else:
                if original_name not in normalized[normalized_name]["original_names"]:
                    normalized[normalized_name]["original_names"].append(original_name)
        
        result = sorted(normalized.values(), key=lambda x: x["importance"], reverse=True)
        
        for col in result:
            if len(col["original_names"]) > 1:
                print(f"   🔗 Объединено: {col['original_names']} → {col['name']}")
        
        return result
    
    async def discover_columns_with_llm(self, texts: List[str], max_cols: int, domain_description: str) -> List[Dict]:
        """Определяет колонки через LLM"""
        
        print("\n🔍 Анализируем тексты с помощью LLM...")
        
        sample_texts = texts[:3] if len(texts) > 3 else texts
        print(f"📊 Анализируем {len(sample_texts)} текстов")
        
        prompt = f"""Ты - медицинский эксперт. Проанализируй тексты и верни JSON с ЧИСЛОВЫМИ показателями.

Предметная область: {domain_description}

Примеры текстов:
{chr(10).join([f'Текст {i+1}: {t[:300]}' for i, t in enumerate(sample_texts)])}

Верни JSON: {{"columns": [{{"name": "Название", "description": "Описание"}}]}}

Только JSON."""
        
        response = self._call_ollama(prompt)
        
        if not response:
            return []
        
        # Парсим JSON
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            try:
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                data = json.loads(json_str)
                if "columns" in data:
                    columns = data["columns"]
                    print(f"📊 До объединения: {len(columns)} колонок")
                    columns = self._merge_columns(columns)
                    print(f"📊 После объединения: {len(columns)} колонок")
                    return columns
            except Exception as e:
                print(f"❌ Ошибка парсинга: {e}")
        
        return []
    
    def _get_fallback_columns(self) -> List[Dict]:
        """Базовый набор колонок"""
        print("📋 Используем базовый набор колонок")
        return [
            {"name": "Возраст", "description": "Возраст пациента в годах"},
            {"name": "Вес", "description": "Вес пациента в кг"},
            {"name": "Рост", "description": "Рост пациента в см"},
            {"name": "Лейкоциты", "description": "Уровень лейкоцитов (10⁹/л)"},
            {"name": "Гемоглобин", "description": "Уровень гемоглобина (г/л)"},
            {"name": "Тромбоциты", "description": "Уровень тромбоцитов (10⁹/л)"},
            {"name": "АСТ", "description": "Аспартатаминотрансфераза (Ед/л)"},
            {"name": "АЛТ", "description": "Аланинаминотрансфераза (Ед/л)"},
            {"name": "Длительность", "description": "Длительность заболевания (дни)"},
        ]
    
    async def discover_columns(self, texts: List[str], max_cols: int, domain_description: str = "") -> List[Dict]:
        """Определяет колонки"""
        
        print("\n" + "="*50)
        print("🚀 НАЧАЛО ОПРЕДЕЛЕНИЯ КОЛОНОК")
        print("="*50)
        print(f"📝 Описание области: {domain_description}")
        print(f"📊 Всего текстов: {len(texts)}")
        
        if not texts:
            return self._get_fallback_columns()[:max_cols]
        
        if domain_description:
            print("🤖 Пытаемся использовать LLM...")
            columns = await self.discover_columns_with_llm(texts, max_cols, domain_description)
            if columns:
                print(f"✅ УСПЕШНО! Используем {len(columns)} колонок от LLM")
                return columns[:max_cols]
            else:
                print("⚠️ LLM не дала результата")
        else:
            print("⚠️ Нет описания области")
        
        return self._get_fallback_columns()[:max_cols]
    
    async def extract_values(self, text: str, columns: List[Dict], row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Извлекает значения из текста"""
        result = {}
        text_lower = text.lower()
        
        for col in columns:
            name = col["name"]
            value = None
            
            if row is not None:
                if name == "Возраст" and 'AGE' in row and pd.notna(row['AGE']):
                    try:
                        value = float(row['AGE'])
                        result[name] = value
                        continue
                    except:
                        pass
            
            value = self._extract_by_name(text, text_lower, name)
            result[name] = value
        
        return result
    
    def _extract_by_name(self, text: str, text_lower: str, name: str) -> Optional[float]:
        name_lower = name.lower()
        
        for canonical, variants in self.synonyms.items():
            if name_lower == canonical or any(v in name_lower for v in variants):
                if canonical == "возраст":
                    return self._extract_age(text, text_lower)
                elif canonical == "вес":
                    return self._extract_weight(text, text_lower)
                elif canonical in ["лейкоциты", "гемоглобин", "тромбоциты", "аст", "алт"]:
                    return self._extract_lab(text, text_lower, variants)
                elif canonical == "длительность":
                    return self._extract_duration(text, text_lower)
                elif canonical == "давление":
                    return self._extract_pressure(text, text_lower)
                elif canonical == "пульс":
                    return self._extract_pulse(text, text_lower)
        
        return None
    
    def _extract_age(self, text: str, text_lower: str) -> Optional[float]:
        patterns = [r'(\d+)\s*(?:лет|год|года)', r'возраст[:\s]*(\d+)']
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    age = float(match.group(1))
                    if 0 < age < 130:
                        return age
                except:
                    pass
        return None
    
    def _extract_weight(self, text: str, text_lower: str) -> Optional[float]:
        patterns = [r'(\d+[.,]?\d*)\s*(?:кг|килограмм)', r'вес[:\s]*(\d+[.,]?\d*)']
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    weight = float(match.group(1).replace(',', '.'))
                    if 1 < weight < 300:
                        return weight
                except:
                    pass
        return None
    
    def _extract_lab(self, text: str, text_lower: str, keywords: List[str]) -> Optional[float]:
        if self._is_date(text):
            return None
        for keyword in keywords:
            if keyword in text_lower:
                pattern = rf'{keyword}[^\d]*(\d+[.,]?\d*)'
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        val = float(match.group(1).replace(',', '.'))
                        if 0 < val < 1000:
                            return val
                    except:
                        pass
        return None
    
    def _extract_duration(self, text: str, text_lower: str) -> Optional[float]:
        if self._is_date(text):
            return None
        patterns = [
            (r'(\d{1,3})\s*(?:дней|дня|день|дн)', 1),
            (r'(\d{1,2})\s*(?:недел[юи]|нед)', 7),
            (r'(\d{1,2})\s*(?:месяц|месяца|месяцев|мес)', 30),
            (r'(\d{1,2})\s*(?:год|года|лет|г)', 365),
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    num = float(match.group(1))
                    if num < 100:
                        return num * multiplier
                except:
                    pass
        if 'в течение дня' in text_lower:
            return 1
        return None
    
    def _extract_pressure(self, text: str, text_lower: str) -> Optional[float]:
        if self._is_date(text):
            return None
        pattern = r'(\d{2,3})\s*[\/\-]\s*(\d{2,3})'
        match = re.search(pattern, text_lower)
        if match:
            try:
                systolic = float(match.group(1))
                if 70 < systolic < 250:
                    return systolic
            except:
                pass
        return None
    
    def _extract_pulse(self, text: str, text_lower: str) -> Optional[float]:
        if self._is_date(text):
            return None
        patterns = [r'пульс[^\d]*(\d{2,3})', r'чсс[^\d]*(\d{2,3})']
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    pulse = float(match.group(1))
                    if 30 < pulse < 220:
                        return pulse
                except:
                    pass
        return None
    
    def _is_date(self, text: str) -> bool:
        date_patterns = [
            r'\d{2}[./]\d{2}[./]\d{2,4}',
            r'\d{2}\.\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}\s*г',
        ]
        text_lower = text.lower()
        for pattern in date_patterns:
            if re.search(pattern, text_lower):
                return True
        return False