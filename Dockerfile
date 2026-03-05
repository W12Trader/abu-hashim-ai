FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

RUN mkdir -p dataset_raw/qalam_exports \
    dataset_processed/qalam_processed \
    learning_buffer \
    model_base \
    model_finetune \
    model_inference

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
