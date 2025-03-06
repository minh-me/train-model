# source venv/bin/activate  # Kích hoạt virtual environment (nếu có)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4