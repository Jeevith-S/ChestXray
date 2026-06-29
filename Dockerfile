FROM python:3.11-slim

RUN apt-get update && apt-get install -y 
libgl1 
libglib2.0-0 
libsm6 
libxext6 
libxrender-dev

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD uvicorn fapi:app --host 0.0.0.0 --port ${PORT:-8080}
