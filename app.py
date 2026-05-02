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
from config import Config

app = FastAPI(title="Medical Data Extractor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

extractor = DataExtractor()
kb = KnowledgeBase(Config.KNOWLEDGE_BASE_FILE)
tasks = {}

def find_id_column(df: pd.DataFrame) -> str:
    possible_id_names = ['PersonID_Ref', 'Идентификационный номер', 'ID', 'Id', 'id', 'patient_id', 'PatientID', 'Номер', '№']
    for col in df.columns:
        for id_name in possible_id_names:
            if id_name.lower() in col.lower():
                print(f"🔍 Найдена ID колонка: {col}")
                return col
    return df.columns[0]

def find_text_column(df: pd.DataFrame) -> str:
    possible_text_names = ['PropertyValue', 'Текст', 'Описание', 'Text', 'Диагноз']
    for col in df.columns:
        for text_name in possible_text_names:
            if text_name.lower() in col.lower():
                return col
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna()
            if len(sample) > 0 and sample.astype(str).str.len().mean() > 30:
                return col
    return df.columns[-1]

def find_target_column(df: pd.DataFrame, user_target_col: str) -> str:
    """Поиск колонки IsTarget"""
    if user_target_col and user_target_col in df.columns:
        return user_target_col
    
    possible_target_names = ['IsTarget', 'is_target', 'target', 'Target', 'Цель']
    for col in df.columns:
        for target_name in possible_target_names:
            if target_name.lower() in col.lower():
                print(f"🔍 Найдена Target колонка: {col}")
                return col
    return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    id_col: str = Form(""),
    text_col: str = Form(""),
    target_col: str = Form(""),
    max_cols: int = Form(20),
    domain_desc: str = Form("")
):
    task_id = str(uuid.uuid4())[:8]
    file_path = f"uploads/{task_id}_{file.filename}"
    
    print(f"\n📤 Получен файл: {file.filename}")
    print(f"📝 Описание области: {domain_desc}")
    print(f"🎯 Target колонка: {target_col}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    tasks[task_id] = {"status": "processing", "progress": 0, "result": None, "filename": None}
    
    background_tasks.add_task(process_file, task_id, file_path, id_col, text_col, target_col, max_cols, domain_desc)
    
    return {"task_id": task_id}

@app.post("/generate_knowledge_base")
async def generate_knowledge_base(
    file: UploadFile = File(...),
    domain_desc: str = Form(""),
    max_cols: int = Form(20)
):
    """Автоматически формирует базу знаний на основе загруженного файла"""
    try:
        file_path = f"uploads/temp_{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Находим текстовую колонку
        text_col = find_text_column(df)
        texts = df[text_col].dropna().astype(str).tolist()[:50]  # Берем первые 50 записей
        
        # Формируем базу знаний
        columns = await extractor.discover_columns(texts, max_cols, domain_desc)
        
        if columns:
            kb.save(columns)
            os.remove(file_path)
            return {"success": True, "columns_count": len(columns), "columns": columns}
        else:
            return {"success": False, "error": "Не удалось сформировать колонки"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/clear_knowledge_base")
async def clear_knowledge_base():
    """Очищает базу знаний"""
    try:
        kb.clear()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, JSONResponse(status_code=404, content={"error": "Задача не найдена"}))

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    if task_id not in tasks or not tasks[task_id]["result"]:
        return JSONResponse(status_code=404, content={"error": "Результат не найден"})
    
    result_path = tasks[task_id]["result"]
    if os.path.exists(result_path):
        filename = tasks[task_id].get("filename", f"result_{task_id}.xlsx")
        return FileResponse(result_path, filename=filename)
    
    return JSONResponse(status_code=404, content={"error": "Файл не найден"})

@app.get("/columns")
async def get_columns():
    return {"columns": kb.get_all()}

async def process_file(task_id: str, file_path: str, id_col: str, text_col: str, target_col: str, max_cols: int, domain_desc: str = ""):
    try:
        print(f"\n🔧 Начинаем обработку задачи {task_id}")
        tasks[task_id]["progress"] = 0.1
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        tasks[task_id]["progress"] = 0.3
        
        # Определяем ID колонку
        if not id_col or id_col not in df.columns:
            id_col = find_id_column(df)
            print(f"📌 Используем ID колонку: {id_col}")
        
        # Определяем текстовую колонку
        if not text_col or text_col not in df.columns:
            text_col = find_text_column(df)
            print(f"📌 Используем текстовую колонку: {text_col}")
        
        texts = df[text_col].dropna().astype(str).tolist() if text_col in df.columns else []
        
        # Загружаем или формируем базу знаний
        columns = kb.get_all()
        if not columns:
            print("📚 База знаний пуста, формируем автоматически...")
            columns = await extractor.discover_columns(texts[:50], max_cols, domain_desc)
            if columns:
                kb.save(columns)
                print(f"✅ Сформировано {len(columns)} колонок")
        
        tasks[task_id]["progress"] = 0.5
        
        results = []
        total_rows = len(df)
        
        print(f"📊 Обработка {total_rows} строк...")
        
        for idx, row in df.iterrows():
            text = str(row[text_col]) if text_col in df and pd.notna(row[text_col]) else ""
            values = await extractor.extract_values(text, columns, row)
            
            # ✅ ИСПРАВЛЕНО: IsTarget всегда копирует значение PersonID_Ref
            row_result = {
                id_col: row[id_col],           # PersonID_Ref
                "IsTarget": row[id_col]        # IsTarget = PersonID_Ref
            }
            
            # Добавляем извлеченные значения
            row_result.update(values)
            results.append(row_result)
            
            if idx % 5 == 0:
                tasks[task_id]["progress"] = 0.5 + 0.4 * (idx / total_rows)
                print(f"   Обработано {idx}/{total_rows} строк...")
        
        # Создаем DataFrame
        result_df = pd.DataFrame(results)
        
        # Переупорядочиваем колонки: PersonID_Ref, IsTarget, затем остальные
        cols_order = [id_col, "IsTarget"] + [c for c in result_df.columns if c not in [id_col, "IsTarget"]]
        result_df = result_df[cols_order]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(file_path)
        if original_name.startswith(f"{task_id}_"):
            original_name = original_name[len(task_id)+1:]
        original_name = original_name.replace('.xlsx', '').replace('.csv', '')
        
        filename = f"result_{original_name}_{timestamp}.xlsx"
        out_path = f"outputs/{filename}"
        
        result_df.to_excel(out_path, index=False)
        
        tasks[task_id].update({"status": "completed", "progress": 1.0, "result": out_path, "filename": filename})
        
        print(f"\n✅ Обработка завершена: {filename}")
        print(f"📊 Колонки в результате: {list(result_df.columns)}")
        print(f"📊 Первые 3 строки результата:")
        print(result_df.head(3).to_string())
        print(f"\n📊 Проверка IsTarget: {result_df['IsTarget'].tolist()[:5]}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, reload=True)