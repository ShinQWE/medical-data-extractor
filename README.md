# Medical Data Extractor

Сервис для извлечения числовых медицинских показателей из текстов с использованием LLM (Qwen) и эмбедингов.

---

##  Проект

Сервис анализирует медицинские тексты, определяет числовые показатели (возраст, дозировки, размеры, давление, лабораторные данные) и формирует структурированный датасет с извлеченными числами.

**Технологии:**
- **LLM**: Qwen3.5-27B-FP8 (анализ текстов)
- **Эмбединги**: BAAI/bge-small-en-v1.5 (поиск похожих записей)
- **Веб-сервер**: FastAPI + Uvicorn

---

##  Быстрый запускe

### 1. Клонирование проекта

```bash
git clone https://github.com/ShinQWE/medical-data-extractor.git
cd medical-data-extractor
2. Загрузка модели
bash
ollama pull llama3.1:8b
3. Запуск Ollama сервера
bash
ollama serve
4. Установка зависимостей
bash
pip install -r requirements.txt
5. Запуск сервера эмбедингов (Окно 1)
bash
python embedding_server.py --host 0.0.0.0 --port 8000
6. Запуск основного сервера (Окно 2)
bash
python run.py
7. Открыть в браузере

http://127.0.0.1:8080