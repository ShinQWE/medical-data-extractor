# %%
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import os
import time

def query_llm_openai(
    messages: List[Dict[str, Any]],
    model: str = "",
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.8,
    presence_penalty: float = 1.5,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    """
    Отправляет запрос к API модели Qwen с полной гибкостью параметров.
    
    Args:
        messages: Список сообщений (история диалога, текст, изображения).
        model: Идентификатор модели.
        max_tokens: Максимальное количество токенов в ответе.
        temperature: Температура генерации.
        top_p: Параметр nucleus sampling.
        presence_penalty: Штраф за повторение тем.
        extra_body: Словарь с дополнительными параметрами, специфичными для модели.
    
    Returns:
        Tuple[str, float]: Кортеж, содержащий текст ответа модели и время выполнения запроса в секундах.
        
    Пример использования:
    if __name__ == "__main__":
        messages = [
            {"role": "system", "content": "Ты — эксперт по Python."},
            {"role": "user", "content": "Напиши функцию для сортировки списка."}
        ]
        response_text, duration = query_llm_openai(messages=messages, max_tokens=1024)
        print(f"Ответ: {response_text}")
        print(f"Время выполнения: {duration:.2f} сек.")
    """
    client = OpenAI(
        base_url="https://aichat.iacpaas.dvo.ru/api", 
        api_key="sk-1e4b3879f93a4c5d88380aceff94d0ad"
    )
    
    if extra_body is None:
        extra_body = {}
    
    start_time = time.perf_counter()
    
    try:
        chat_response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
        )
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return chat_response.choices[0].message.content, duration
        
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        error_msg = f"Ошибка при запросе к API: {str(e)}"
        # В случае ошибки также возвращаем время, потраченное на попытку запроса
        return error_msg, duration

# %%

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Ты — эксперт по Python."},
        {"role": "user", "content": "Напиши функцию для сортировки списка."},
    ]
    
    custom_extra_body = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    
    # Вызов функции и получение кортежа (ответ, время)
    response_text, time_spent = query_llm_openai(
        # model="/home/atarasov/LLM/base_models/Qwen--Qwen3.5-27B-FP8",
        model="/home/atarasov/LLM/base_models/Qwen--Qwen3.5-9B",
        messages=messages,
        extra_body=custom_extra_body,
        temperature=0.5,
        max_tokens=1024,
    )
    
    print("Ответ модели:")
    print(response_text)
    print("-" * 30)
    print(f"Время, затраченное на ответ: {time_spent:.4f} секунд")
# %%
