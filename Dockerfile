FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl postgresql-client && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
COPY requirements_finetune.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_finetune.txt
COPY backend /app/backend
COPY part4 /app/part4
CMD ["bash"]
