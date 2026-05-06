import re
import pandas as pd
from typing import List, Dict, Any, Optional

class DataExtractor:
    def __init__(self):
        print("✅ DataExtractor инициализирован")
    
    async def extract_value(self, text: str, param_name: str, param_type: str = "numeric", mapping: dict = None) -> Optional[Any]:
        """Извлекает значение параметра из текста"""
        if not text or pd.isna(text):
            return None
        
        text_lower = text.lower()
        
        # Категориальные параметры (преобразуем в числа)
        if param_type == "categorical":
            return self._extract_categorical(text_lower, param_name, mapping)
        
        # Числовые параметры
        return self._extract_numeric(text_lower, param_name)
    
    def _extract_categorical(self, text: str, param_name: str, mapping: dict = None) -> Optional[float]:
        """Извлекает категориальный параметр и преобразует в число"""
        if mapping is None:
            mapping = {}
        
        # Стандартные маппинги
        default_mappings = {
            "пол": {"м": 1, "ж": 0, "муж": 1, "жен": 0, "male": 1, "female": 0, "мужской": 1, "женский": 0},
            "курит": {"да": 1, "нет": 0, "курит": 1, "не курит": 0},
            "диабет": {"есть": 1, "нет": 0, "диабет": 1, "сахарный диабет": 1},
            "гипертензия": {"есть": 1, "нет": 0, "гипертензия": 1, "артериальная гипертензия": 1},
        }
        
        # Ищем ключевые слова в тексте
        for key, value in default_mappings.get(param_name.lower(), {}).items():
            if key in text:
                return value
        
        for key, value in mapping.items():
            if key.lower() in text:
                return value
        
        # Поиск по названию параметра
        patterns = [
            rf'{param_name}[:\s]*([^\s,.]+)',
            rf'{param_name.lower()}[:\s]*([^\s,.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = match.group(1).lower()
                for key, mapped in {**default_mappings.get(param_name.lower(), {}), **mapping}.items():
                    if key in val:
                        return mapped
        
        return None
    
    def _extract_numeric(self, text: str, param_name: str) -> Optional[float]:
        """Извлекает числовое значение параметра"""
        
        # Специфичные паттерны для разных параметров
        patterns = {
            "возраст": [
                r'(\d+)\s*лет',
                r'возраст[:\s]*(\d+)',
                r'age[:\s]*(\d+)',
            ],
            "дозировка_лекарства_мг": [
                r'(\d+[.,]?\d*)\s*(?:мг|mg)',
                r'преднизолон\s*(\d+[.,]?\d*)',
            ],
            "размер_образования_мм": [
                r'(\d+)\s*[хx]\s*\d+\s*(?:мм|mm)',
                r'(\d+[.,]?\d*)\s*(?:мм|mm)',
            ],
            "количество_лимфоузлов": [
                r'лимфоузл[а-я]*\s+(\d+)',
            ],
            "размер_лимфоузла_см": [
                r'лимфоузл[а-я]*\s*(\d+[.,]?\d*)\s*(?:см|cm)',
            ],
            "кровопотеря_мл": [
                r'кровопотер[яи][:\s]*(\d+)\s*(?:мл|ml)',
            ],
            "давление_систолическое": [
                r'(\d{2,3})\s*[\/\-]\s*(\d{2,3})',
            ],
            "пульс": [
                r'пульс[^\d]*(\d{2,3})',
            ],
            "гемоглобин": [r'гемоглобин[^\d]*(\d+[.,]?\d*)'],
            "лейкоциты": [r'лейкоцит[^\d]*(\d+[.,]?\d*)'],
            "тромбоциты": [r'тромбоцит[^\d]*(\d+[.,]?\d*)'],
            "аст": [r'аст[^\d]*(\d+[.,]?\d*)'],
            "алт": [r'алт[^\d]*(\d+[.,]?\d*)'],
            "билирубин": [r'билирубин[^\d]*(\d+[.,]?\d*)'],
            "креатинин": [r'креатинин[^\d]*(\d+[.,]?\d*)'],
        }
        
        param_lower = param_name.lower()
        
        # Проверяем специфичные паттерны
        if param_lower in patterns:
            for pattern in patterns[param_lower]:
                match = re.search(pattern, text)
                if match:
                    try:
                        val = match.group(1).replace(',', '.')
                        return float(val)
                    except:
                        pass
        
        # Общий поиск: название параметра + число
        general_patterns = [
            rf'{param_lower}[^\d]*(\d+[.,]?\d*)',
            rf'{param_lower}\s*[-:]\s*(\d+[.,]?\d*)',
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = match.group(1).replace(',', '.')
                    return float(val)
                except:
                    pass
        
        return None