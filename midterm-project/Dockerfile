FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py predict.py model.bin model_metadata.json ./

EXPOSE 7860

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-7860}"]
