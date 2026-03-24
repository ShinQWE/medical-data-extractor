import json
import os
from typing import List, Dict
from datetime import datetime

class KnowledgeBase:
    """База знаний для хранения информации о колонках"""
    
    def __init__(self, file_path: str = "knowledge_base.json"):
        self.file_path = file_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Загрузка базы знаний из файла"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"columns": [], "stats": {"processed": 0}}
        return {"columns": [], "stats": {"processed": 0}}
    
    def save(self, columns: List[Dict]):
        """Сохранение колонок в базу знаний"""
        self.data["columns"] = columns
        self.data["stats"]["processed"] = self.data["stats"].get("processed", 0) + 1
        self.data["stats"]["last_update"] = str(datetime.now())
        
        # Сохраняем в файл
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def get_all(self) -> List[Dict]:
        """Получение всех колонок"""
        return self.data.get("columns", [])