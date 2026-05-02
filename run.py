#!/usr/bin/env python3
"""
MedExtractor - Запуск приложения
"""

import os
import sys
import subprocess
from config import Config

def check_requirements():
    """Проверка установки зависимостей"""
    try:
        import pandas
        import fastapi
        import uvicorn
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False

def start_embedding_server():
    """Запуск сервера эмбедингов"""
    print("\n🔧 Проверка сервера эмбедингов...")
    # Здесь можно добавить проверку, запущен ли сервер
    print(f"   Ожидается сервер на {Config.EMBEDDING_API_URL}")
    print("   Если сервер не запущен, запустите: python embedding_server.py")

def main():
    """Главная функция запуска"""
    print("="*50)
    print("🧬 MedExtractor - Медицинский экстрактор данных")
    print("="*50)
    
    if not check_requirements():
        sys.exit(1)
    
    start_embedding_server()
    
    print(f"\n🚀 Запуск сервера на http://{Config.HOST}:{Config.PORT}")
    print("="*50)
    
    import uvicorn
    uvicorn.run("app:app", host=Config.HOST, port=Config.PORT, reload=True)

if __name__ == "__main__":
    main()

    # http://127.0.0.1:8000/



    # ollama serve
    # python -m uvicorn app:app --host 127.0.0.1 --port 8080 --reload
    # python embedding_server.py --host 0.0.0.0 --port 8000 --device cpu --api-key sk-mysecretkey123
    
    # python run.py
    # Медицинские карты пациентов с онкологическими заболеваниямиe