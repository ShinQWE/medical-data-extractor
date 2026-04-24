import re
import pandas as pd
import json
import os
import requests
from typing import List, Dict, Any, Optional

class DataExtractor:
    def __init__(self):
        # Настройки - используем мощную модель 8b
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.1:8b"
        
        # ===== РАСШИРЕННЫЙ СЛОВАРЬ СИНОНИМОВ =====
        self.synonyms = {
            "Возраст": ["возраст", "лет", "год", "года", "age"],
            "Вес": ["вес", "масса", "weight", "кг"],
            "Рост": ["рост", "height", "см"],
            "Давление": ["давление", "ад", "артериальное давление"],
            "Пульс": ["пульс", "чсс", "heart rate"],
            "Температура": ["температура", "t", "temperature"],
            "Лейкоциты": ["лейкоциты", "wbc"],
            "Гемоглобин": ["гемоглобин", "hb", "hgb"],
            "Тромбоциты": ["тромбоциты", "plt"],
            "АСТ": ["аст", "ast"],
            "АЛТ": ["алт", "alt"],
            "Глюкоза": ["глюкоза", "сахар", "glucose"],
            "Холестерин": ["холестерин", "cholesterol"],
            "Креатинин": ["креатинин", "creatinine"],
            "Мочевина": ["мочевина", "urea"],
            "Калий": ["калий", "k"],
        }
        
        print("✅ DataExtractor инициализирован")
        print(f"🤖 Используем модель: {self.model}")
        print("="*50)
    
    def _call_ollama(self, prompt: str) -> str:
        """Отправляет запрос в Ollama"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=180
            )
            if response.status_code == 200:
                return response.json()["response"]
            return ""
        except Exception as e:
            print(f"❌ Ошибка Ollama: {e}")
            return ""
    
    async def discover_columns(self, texts: List[str], max_cols: int, domain_description: str = "") -> List[Dict]:
        """Определяет колонки для извлечения (числовые + текстовые)"""
        
        print("\n" + "="*50)
        print("🚀 НАЧАЛО ОПРЕДЕЛЕНИЯ КОЛОНОК")
        print("="*50)
        print(f"📝 Описание области: {domain_description}")
        
        # Расширенный набор колонок - ВАЖНО: добавляем текстовые поля!
        columns = [
            # Числовые показатели
            {"name": "Возраст", "description": "Возраст пациента (лет)"},
            {"name": "Давление", "description": "Артериальное давление (мм рт.ст.)"},
            {"name": "Пульс", "description": "Частота пульса (уд/мин)"},
            {"name": "Гемоглобин", "description": "Гемоглобин (г/л)"},
            {"name": "Лейкоциты", "description": "Лейкоциты (10⁹/л)"},
            {"name": "Тромбоциты", "description": "Тромбоциты (10⁹/л)"},
            {"name": "Креатинин", "description": "Креатинин (мкмоль/л)"},
            {"name": "Холестерин", "description": "Холестерин (ммоль/л)"},
            {"name": "Калий", "description": "Калий (ммоль/л)"},
            {"name": "Мочевина", "description": "Мочевина (ммоль/л)"},
            
            # ТЕКСТОВЫЕ ПОЛЯ - вот что добавили!
            {"name": "Диагноз_MKB", "description": "Код диагноза МКБ (из MCardMKB или MKBCode_Ref)"},
            {"name": "Диагноз_текст", "description": "Клинический диагноз (текстовое описание)"},
            {"name": "Заключение", "description": "Заключение врача/результат исследования"},
            {"name": "Анамнез", "description": "Анамнез заболевания (история болезни)"},
            {"name": "Операция", "description": "Название и протокол хирургической операции"},
            {"name": "Жалобы", "description": "Жалобы пациента при обращении"},
            {"name": "Пол", "description": "Пол пациента (М/Ж)"},
            {"name": "Дата_приема", "description": "Дата медицинского приема/услуги"},
        ]
        
        print(f"✅ Используем {len(columns)} колонок (числовые + текстовые)")
        for col in columns:
            print(f"   - {col['name']}: {col['description']}")
        
        return columns[:max_cols]
    
    async def extract_values(self, text: str, columns: List[Dict], row: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Извлекает значения из текста и из строки Excel"""
        result = {}
        text_lower = text.lower()
        
        for col in columns:
            name = col["name"]
            value = None
            
            # === СНАЧАЛА ПРОВЕРЯЕМ ДАННЫЕ ИЗ EXCEL (они приоритетнее) ===
            if row is not None:
                # Пол из колонки Sex
                if name == "Пол" and 'Sex' in row and pd.notna(row['Sex']):
                    value = str(row['Sex']).strip()
                    if value in ['М', 'Ж', 'M', 'F']:
                        result[name] = value
                        continue
                
                # Возраст из колонки AGE
                if name == "Возраст" and 'AGE' in row and pd.notna(row['AGE']):
                    try:
                        value = float(row['AGE'])
                        result[name] = value
                        continue
                    except:
                        pass
                
                # Диагноз МКБ из MCardMKB или MKBCode_Ref
                if name == "Диагноз_MKB":
                    if 'MCardMKB' in row and pd.notna(row['MCardMKB']):
                        value = str(row['MCardMKB']).strip()
                    elif 'MKBCode_Ref' in row and pd.notna(row['MKBCode_Ref']):
                        value = str(row['MKBCode_Ref']).strip()
                    if value and value != 'nan':
                        result[name] = value
                        continue
                
                # Операция из ServiceName
                if name == "Операция" and 'ServiceName' in row and pd.notna(row['ServiceName']):
                    value = str(row['ServiceName']).strip()
                    if value and value != 'nan':
                        result[name] = value
                        continue
                
                # Дата приема из StartDate
                if name == "Дата_приема" and 'StartDate' in row and pd.notna(row['StartDate']):
                    value = str(row['StartDate']).split()[0]  # Берем только дату
                    result[name] = value
                    continue
            
            # === ЕСЛИ В EXCEL НЕТ, ИЗВЛЕКАЕМ ИЗ ТЕКСТА ===
            if name == "Возраст":
                value = self._extract_age(text, text_lower)
            elif name == "Давление":
                value = self._extract_pressure(text, text_lower)
            elif name == "Пульс":
                value = self._extract_pulse(text, text_lower)
            elif name in ["Гемоглобин", "Лейкоциты", "Тромбоциты", "Креатинин", "Холестерин", "Калий", "Мочевина"]:
                value = self._extract_lab_value(text, text_lower, name)
            elif name == "Диагноз_текст":
                value = self._extract_text_section(text, ["диагноз", "клинический диагноз", "диагноз:", "disease"] , 300)
            elif name == "Заключение":
                value = self._extract_text_section(text, ["заключение", "вывод", "результат", "conclusion"], 400)
            elif name == "Анамнез":
                value = self._extract_text_section(text, ["анамнез", "история заболевания", "anamnesis", "анамнез заболевания"], 600)
            elif name == "Операция":
                if not result.get("Операция"):  # Если не нашли в Excel
                    value = self._extract_text_section(text, ["операция", "протокол операции", "surgery", "хирургическое"], 400)
            elif name == "Жалобы":
                value = self._extract_text_section(text, ["жалобы", "жалуется", "complaints", "предъявляет"], 300)
            elif name == "Пол":
                value = self._extract_gender(text, text_lower)
            elif name == "Дата_приема":
                value = self._extract_date(text, text_lower)
            
            result[name] = value
        
        return result
    
    # === МЕТОДЫ ДЛЯ ИЗВЛЕЧЕНИЯ ЧИСЛОВЫХ ЗНАЧЕНИЙ ===
    
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
    
    def _extract_pressure(self, text: str, text_lower: str) -> Optional[float]:
        patterns = [r'(\d{2,3})\s*[\/\-]\s*(\d{2,3})']
        for pattern in patterns:
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
    
    def _extract_lab_value(self, text: str, text_lower: str, param_name: str) -> Optional[float]:
        param_lower = param_name.lower()
        patterns = [
            rf'{param_lower}[^\d]*(\d+[.,]?\d*)',
            rf'{param_lower}[:\s]*(\d+[.,]?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1).replace(',', '.'))
                    if 0 < val < 10000:
                        return val
                except:
                    pass
        return None
    
    # === НОВЫЕ МЕТОДЫ ДЛЯ ИЗВЛЕЧЕНИЯ ТЕКСТОВЫХ ПОЛЕЙ ===
    
    def _extract_text_section(self, text: str, keywords: List[str], max_len: int = 300) -> Optional[str]:
        """Извлекает текстовую секцию по ключевым словам"""
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Ищем строку, содержащую ключевое слово
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if keyword_lower in line.lower():
                    # Берем эту строку и следующую (если есть)
                    extracted = line.strip()
                    if i + 1 < len(lines) and len(lines[i+1].strip()) > 20:
                        extracted += " " + lines[i+1].strip()
                    
                    # Очищаем от лишних пробелов
                    extracted = re.sub(r'\s+', ' ', extracted)
                    
                    if len(extracted) > 10 and len(extracted) < max_len:
                        return extracted[:max_len]
            
            # Ищем в одном абзаце
            pattern = rf'{keyword_lower}[:\s]*([^.!?]+[.!?])'
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 10:
                    return extracted[:max_len]
        
        # Если текст не слишком длинный, возвращаем начало
        if 20 < len(text) < max_len:
            return text.strip()[:max_len]
        
        return None
    
    def _extract_gender(self, text: str, text_lower: str) -> Optional[str]:
        """Извлекает пол пациента"""
        # Ищем явные указания
        if re.search(r'\b[Мм]ужчина\b|\b[Мм]ужской\b|\b[Мм]\b', text):
            return "М"
        if re.search(r'\b[Жж]енщина\b|\b[Жж]енский\b|\b[Жж]\b', text):
            return "Ж"
        
        # Проверяем по местоимениям
        if re.search(r'\bон\b|\bему\b|\bего\b', text_lower):
            return "М"
        if re.search(r'\bона\b|\bей\b|\bеё\b', text_lower):
            return "Ж"
        
        return None
    
    def _extract_date(self, text: str, text_lower: str) -> Optional[str]:
        """Извлекает дату"""
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}\.\d{2}\.\d{4})',
            r'(\d{2}/\d{2}/\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None