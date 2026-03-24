from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import uuid
import shutil
import json
from datetime import datetime
from extractor import DataExtractor
from knowledge_base import KnowledgeBase

app = FastAPI(title="Medical Data Extractor")

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Создание необходимых папок
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Инициализация компонентов
# api опенAi а olamma сервер был
extractor = DataExtractor()
kb = KnowledgeBase("knowledge_base.json")
tasks = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    id_col: str = Form("PersonID_Ref"),
    text_col: str = Form("PropertyValue"),
    max_cols: int = Form(15),
    domain_desc: str = Form("")  # Важно! Это поле для описания
):
    """Загрузка и обработка файла"""
    task_id = str(uuid.uuid4())[:8]
    file_path = f"uploads/{task_id}_{file.filename}"
    
    print(f"\n📤 Получен файл: {file.filename}")
    print(f"📝 Описание области: {domain_desc}")
    
    # Сохраняем файл
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Инициализируем статус задачи
    tasks[task_id] = {
        "status": "processing", 
        "progress": 0, 
        "result": None,
        "filename": None
    }
    
    # Запускаем обработку в фоне
    background_tasks.add_task(
        process_file, task_id, file_path, id_col, text_col, max_cols, domain_desc
    )
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "Задача не найдена"})
    return tasks[task_id]

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    if task_id not in tasks or not tasks[task_id]["result"]:
        return JSONResponse(status_code=404, content={"error": "Результат не найден"})
    
    result_path = tasks[task_id]["result"]
    if os.path.exists(result_path):
        filename = tasks[task_id].get("filename", f"result_{task_id}.xlsx")
        
        return FileResponse(
            result_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename
        )
    return JSONResponse(status_code=404, content={"error": "Файл не найден"})

@app.get("/columns")
async def get_columns():
    return {"columns": kb.get_all()}

async def process_file(task_id: str, file_path: str, id_col: str, text_col: str, max_cols: int, domain_desc: str = ""):
    """Фоновая обработка файла"""
    try:
        print(f"\n🔧 Начинаем обработку задачи {task_id}")
        print(f"📝 Описание области в process_file: {domain_desc}")
        
        tasks[task_id]["progress"] = 0.1
        
        # Читаем файл
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        tasks[task_id]["progress"] = 0.3
        
        # Проверяем колонки
        if id_col not in df.columns:
            raise ValueError(f"Колонка ID '{id_col}' не найдена")
        if text_col not in df.columns:
            raise ValueError(f"Текстовая колонка '{text_col}' не найдена")
        
        # Получаем тексты
        texts = df[text_col].dropna().astype(str).tolist()
        
        # Получаем или создаем колонки
        columns = kb.get_all()
        if not columns:
            sample_texts = texts[:10] if len(texts) > 10 else texts
            print(f"🤖 Запускаем discover_columns с описанием: {domain_desc}")
            columns = await extractor.discover_columns(sample_texts, max_cols, domain_desc)
            if columns:
                print(f"💾 Сохраняем {len(columns)} колонок в базу знаний")
                kb.save(columns)
        
        tasks[task_id]["progress"] = 0.5
        
        # Извлекаем значения
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            values = await extractor.extract_values(text, columns, row)
            
            row_result = {id_col: row[id_col]}
            row_result.update(values)
            results.append(row_result)
            
            if idx % 5 == 0:
                tasks[task_id]["progress"] = 0.5 + 0.4 * (idx / total_rows)
        
        # Создаем DataFrame с результатами
        result_df = pd.DataFrame(results)
        
        # Сохраняем результат
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(file_path)
        if original_name.startswith(f"{task_id}_"):
            original_name = original_name[len(task_id)+1:]
        original_name = original_name.replace('.xlsx', '').replace('.csv', '')
        
        filename = f"result_{original_name}_{timestamp}.xlsx"
        out_path = f"outputs/{filename}"
        
        result_df.to_excel(out_path, index=False)
        
        tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "result": out_path,
            "filename": filename
        })
        
        print(f"✅ Обработка завершена: {filename}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)