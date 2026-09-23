FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY params.yaml .
COPY src/ ./src/
COPY artifacts/models/flan-t5-base-lora/ ./artifacts/models/flan-t5-base-lora/

EXPOSE 8000

CMD ["uvicorn", "summarization.api:app", "--host", "0.0.0.0", "--port", "8000"]
