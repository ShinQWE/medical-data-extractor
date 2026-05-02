import json
import os
from typing import List, Dict
from datetime import datetime

class KnowledgeBase:
    def __init__(self, file_path: str = "knowledge_base.json"):
        self.file_path = file_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"columns": [], "stats": {"processed": 0, "auto_generated": False}}
        return {"columns": [], "stats": {"processed": 0, "auto_generated": False}}
    
    def save(self, columns: List[Dict]):
        self.data["columns"] = columns
        self.data["stats"]["processed"] = self.data["stats"].get("processed", 0) + 1
        self.data["stats"]["last_update"] = str(datetime.now())
        self.data["stats"]["auto_generated"] = True
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ База знаний сохранена: {len(columns)} колонок")
    
    def clear(self):
        """Очищает базу знаний"""
        self.data = {"columns": [], "stats": {"processed": 0, "auto_generated": False}}
        self.save([])
        print("🗑️ База знаний очищена")
    
    def get_all(self) -> List[Dict]:
        return self.data.get("columns", [])
    
    def add_column(self, column: Dict):
        """Добавляет новую колонку"""
        columns = self.get_all()
        columns.append(column)
        self.save(columns)
    
    def remove_column(self, column_name: str):
        """Удаляет колонку по имени"""
        columns = self.get_all()
        columns = [c for c in columns if c.get("name") != column_name]
        self.save(columns)