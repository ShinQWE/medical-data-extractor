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
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    tasks[task_id] = {"status": "processing", "progress": 0, "result": None, "filename": None}
    
    background_tasks.add_task(process_file, task_id, file_path, id_col, text_col, target_col, max_cols, domain_desc)
    
    return {"task_id": task_id}

@app.post("/clear_knowledge_base")
async def clear_knowledge_base():
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
        
        # Загрузка файла
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        tasks[task_id]["progress"] = 0.2
        
        # Определяем колонки
        if not id_col or id_col not in df.columns:
            id_col = find_id_column(df)
        
        if not text_col or text_col not in df.columns:
            text_col = find_text_column(df)
        
        print(f"📌 ID колонка: {id_col}")
        print(f"📌 Текстовая колонка: {text_col}")
        
        # Получаем или создаем базу знаний
        columns = kb.get_all()
        
        # ДОБАВЛЯЕМ СТАНДАРТНЫЕ КОЛОНКИ, ЕСЛИ БАЗА ПУСТА
        if not columns:
            print("📚 База знаний пуста, создаем стандартные колонки...")
            columns = [
                {"name": "Возраст", "type": "numeric", "description": "лет"},
                {"name": "Дозировка_лекарства_мг", "type": "numeric", "description": "мг/сут"},
                {"name": "Размер_образования_мм", "type": "numeric", "description": "мм"},
                {"name": "Количество_лимфоузлов", "type": "numeric", "description": "штук"},
                {"name": "Размер_лимфоузла_см", "type": "numeric", "description": "см"},
                {"name": "Кровопотеря_мл", "type": "numeric", "description": "мл"},
                {"name": "Давление_систолическое", "type": "numeric", "description": "мм рт.ст."},
                {"name": "Пульс", "type": "numeric", "description": "уд/мин"},
                {"name": "Гемоглобин", "type": "numeric", "description": "г/л"},
                {"name": "Лейкоциты", "type": "numeric", "description": "10⁹/л"},
                {"name": "Тромбоциты", "type": "numeric", "description": "10⁹/л"},
                {"name": "АСТ", "type": "numeric", "description": "Ед/л"},
                {"name": "АЛТ", "type": "numeric", "description": "Ед/л"},
                {"name": "Билирубин", "type": "numeric", "description": "мкмоль/л"},
                {"name": "Креатинин", "type": "numeric", "description": "мкмоль/л"},
                {"name": "Пол", "type": "categorical", "description": "М/Ж", "mapping": {"м": 1, "ж": 0, "male": 1, "female": 0, "мужской": 1, "женский": 0}},
                {"name": "Курит", "type": "categorical", "description": "да/нет", "mapping": {"да": 1, "нет": 0, "курит": 1, "не курит": 0}},
                {"name": "Диабет", "type": "categorical", "description": "есть/нет", "mapping": {"есть": 1, "нет": 0, "диабет": 1}},
                {"name": "Гипертензия", "type": "categorical", "description": "есть/нет", "mapping": {"есть": 1, "нет": 0}},
            ]
            kb.save(columns)
            print(f"✅ Создано {len(columns)} колонок")
        
        tasks[task_id]["progress"] = 0.4
        
        # ИЗВЛЕКАЕМ ЗНАЧЕНИЯ ДЛЯ КАЖДОЙ СТРОКИ
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            
            # Извлекаем значения для всех колонок
            extracted_values = {}
            for col in columns:
                col_name = col["name"]
                col_type = col.get("type", "numeric")
                
                # Извлекаем значение
                value = await extractor.extract_value(text, col_name, col_type, col.get("mapping", {}))
                
                if value is not None:
                    extracted_values[col_name] = value
            
            # Формируем строку результата
            row_result = {
                "PersonID_Ref": row[id_col],
                "IsTarget": row[id_col]  # Копируем ID
            }
            row_result.update(extracted_values)
            results.append(row_result)
            
            if idx % 5 == 0:
                tasks[task_id]["progress"] = 0.4 + 0.5 * (idx / total_rows)
        
        # СОЗДАЕМ DATAFRAME СО ВСЕМИ КОЛОНКАМИ
        result_df = pd.DataFrame(results)
        
        # Заполняем пропуски
        result_df = result_df.fillna('')
        
        # Сохраняем результат
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
        print(f"📊 Всего колонок в результате: {len(result_df.columns)}")
        print(f"📊 Колонки: {list(result_df.columns)}")
        print(f"📊 Первые 3 строки:")
        print(result_df.head(3).to_string())
        
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