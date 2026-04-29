import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    # http://127.0.0.1:8000/



    # ollama serve
    # python -m uvicorn app:app --host 127.0.0.1 --port 8080 --reload
    # python embedding_server.py --host 0.0.0.0 --port 8000 --device cpu --api-key sk-mysecretkey123
    
    # python run.py
    # Медицинские карты пациентов с онкологическими заболеваниямиe