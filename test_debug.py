# test_debug.py
import pandas as pd
import asyncio
from extractor import DataExtractor
from knowledge_base import KnowledgeBase

async def test():
    print("="*50)
    print("ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("="*50)
    
    # 1. Проверка базы знаний
    kb = KnowledgeBase("knowledge_base.json")
    columns = kb.get_all()
    print(f"\n1. База знаний:")
    print(f"   Колонок: {len(columns)}")
    for col in columns:
        print(f"   - {col}")
    
    # 2. Проверка извлечения
    extractor = DataExtractor()
    
    test_texts = [
        "Пациент 45 лет, преднизолон 5мг, давление 120/80, пульс 75",
        "Пол мужской, курит, диабет есть",
        "Гемоглобин 130, лейкоциты 8.5, тромбоциты 250"
    ]
    
    print(f"\n2. Проверка извлечения значений:")
    for text in test_texts:
        print(f"\n   Текст: {text}")
        for col in columns[:5]:  # Проверяем первые 5 колонок
            col_name = col.get("name")
            col_type = col.get("type", "numeric")
            value = await extractor.extract_value(text, col_name, col_type, col.get("mapping", {}))
            print(f"   {col_name} ({col_type}): {value}")
    
    # 3. Проверка полного цикла обработки
    print(f"\n3. Проверка обработки DataFrame:")
    df = pd.DataFrame({
        'PersonID_Ref': [1, 2, 3],
        'PropertyValue': test_texts
    })
    
    results = []
    for idx, row in df.iterrows():
        text = str(row['PropertyValue'])
        row_result = {
            "PersonID_Ref": row['PersonID_Ref'],
            "IsTarget": row['PersonID_Ref']
        }
        
        for col in columns:
            col_name = col.get("name")
            col_type = col.get("type", "numeric")
            value = await extractor.extract_value(text, col_name, col_type, col.get("mapping", {}))
            if value is not None:
                row_result[col_name] = value
        
        results.append(row_result)
    
    result_df = pd.DataFrame(results)
    print(f"\n   Результат:")
    print(result_df.to_string())
    print(f"\n   Колонки в результате: {list(result_df.columns)}")

if __name__ == "__main__":
    asyncio.run(test())